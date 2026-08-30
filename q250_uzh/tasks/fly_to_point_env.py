from __future__ import annotations

import math

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.markers import CUBOID_MARKER_CFG, VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from q250_uzh.config import Q250
from q250_uzh.fly_to_point_math import FlyToPointRewardCfg, compute_fly_to_point_reward, curriculum_bounds
from q250_uzh.isaac.torch_dynamics import Q250TorchMotorBank
from q250_uzh.rl_control import CTBRActionCfg, TorchBodyRatePID, TorchMotorAllocator, map_actions_to_ctbr


@configclass
class FlyToPointEnvCfg(DirectRLEnvCfg):
    # RL timing: validated 240 Hz physics inner loop, 60 Hz policy.
    episode_length_s = 8.0
    decimation = 4
    action_space = 4
    observation_space = 12
    state_space = 0
    debug_vis = False

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 240.0,
        render_interval=decimation,
        gravity=(0.0, 0.0, -Q250.gravity_m_s2),
        physx=sim_utils.PhysxCfg(enable_external_forces_every_iteration=True),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512,
        env_spacing=10.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    robot: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.CuboidCfg(
            size=Q250.inertia_equivalent_box_m,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=60.0,
                max_angular_velocity=30.0,
                solver_velocity_iteration_count=1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=Q250.mass_kg),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.12, 0.28, 0.42), metallic=0.25
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.5), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    # Task/control limits.
    target_radius_m = 0.25
    min_z_m = 0.15
    max_z_m = 5.0
    max_xy_from_origin_m = 6.0
    spawn_z_m = 1.5

    action_cfg = CTBRActionCfg()
    reward_cfg = FlyToPointRewardCfg()


class FlyToPointEnv(DirectRLEnv):
    """Q250 privileged-state Fly-to-Point task with CTBR actions.

    Observation (12): [target_pos_b(3), lin_vel_b(3), projected_gravity_b(3), ang_vel_b(3)]
    Action (4): normalized [collective, p_cmd, q_cmd, r_cmd].
    """

    cfg: FlyToPointEnvCfg

    def __init__(self, cfg: FlyToPointEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self._collective_cmd_n = torch.full(
            (self.num_envs,), Q250.mass_kg * Q250.gravity_m_s2, dtype=torch.float32, device=self.device
        )
        self._rate_cmd_rad_s = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self._desired_pos_w = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self._previous_distance = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._last_success = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._last_crashed = torch.zeros_like(self._last_success)
        self._allocator_saturated = torch.zeros_like(self._last_success)

        self._rate_pid = TorchBodyRatePID(self.num_envs, self.device)
        self._allocator = TorchMotorAllocator(self.device)
        self._motors = Q250TorchMotorBank(self.num_envs, self.device)
        self._motors.reset(omega_rad_s=Q250.hover_omega_rad_s)

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            for key in ("progress", "success", "crash", "action")
        }
        self._success_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = RigidObject(self.cfg.robot)
        self.scene.rigid_objects["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2200.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._collective_cmd_n, self._rate_cmd_rad_s = map_actions_to_ctbr(
            self._actions, self.cfg.action_cfg
        )

    def _apply_action(self):
        measured_rates = self._robot.data.root_ang_vel_b
        torque_cmd = self._rate_pid.update(
            self._rate_cmd_rad_s, measured_rates, self.physics_dt
        )
        omega_cmd, self._allocator_saturated = self._allocator.allocate(
            self._collective_cmd_n, torque_cmd
        )
        self._motors.step_omega_command(omega_cmd, self.physics_dt)
        forces_b, torques_b = self._motors.wrench()

        composer = getattr(self._robot, "permanent_wrench_composer", None)
        if composer is not None:
            composer.set_forces_and_torques(forces=forces_b, torques=torques_b)
        else:
            self._robot.set_external_force_and_torque(forces_b, torques_b, is_global=False)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._desired_pos_w,
        )
        obs = torch.cat(
            (
                desired_pos_b,
                self._robot.data.root_lin_vel_b,
                self._robot.data.projected_gravity_b,
                self._robot.data.root_ang_vel_b,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _termination_flags(self) -> tuple[torch.Tensor, torch.Tensor]:
        distance = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=-1)
        success = distance <= self.cfg.target_radius_m

        local_pos = self._robot.data.root_pos_w - self._terrain.env_origins
        too_low = local_pos[:, 2] < self.cfg.min_z_m
        too_high = local_pos[:, 2] > self.cfg.max_z_m
        too_far_xy = torch.linalg.norm(local_pos[:, :2], dim=-1) > self.cfg.max_xy_from_origin_m
        crashed = too_low | too_high | too_far_xy
        return success, crashed

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._last_success, self._last_crashed = self._termination_flags()
        terminated = self._last_success | self._last_crashed
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _get_rewards(self) -> torch.Tensor:
        current_distance = torch.linalg.norm(
            self._desired_pos_w - self._robot.data.root_pos_w, dim=-1
        )
        reward, parts = compute_fly_to_point_reward(
            current_distance=current_distance,
            previous_distance=self._previous_distance,
            success=self._last_success,
            crashed=self._last_crashed,
            actions=self._actions,
            cfg=self.cfg.reward_cfg,
        )
        self._previous_distance.copy_(current_distance)
        for key, value in parts.items():
            self._episode_sums[key] += value
        return reward

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        # Episode logging before buffers are cleared.
        if self.common_step_counter > 0 and len(env_ids) > 0:
            self.extras["log"] = {}
            for key, value in self._episode_sums.items():
                self.extras["log"][f"Episode_Reward/{key}"] = torch.mean(value[env_ids]).item()
            final_dist = torch.linalg.norm(
                self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=-1
            )
            self.extras["log"]["Metrics/final_distance_m"] = torch.mean(final_dist).item()
            self.extras["log"]["Metrics/success_rate"] = torch.mean(
                self._last_success[env_ids].float()
            ).item()
            self.extras["log"]["Metrics/allocator_saturation_rate"] = torch.mean(
                self._allocator_saturated[env_ids].float()
            ).item()
            self.extras["log"]["Curriculum/stage"] = float(
                curriculum_bounds(self.common_step_counter).stage
            )

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        self._actions[env_ids] = 0.0
        self._collective_cmd_n[env_ids] = Q250.mass_kg * Q250.gravity_m_s2
        self._rate_cmd_rad_s[env_ids] = 0.0
        self._rate_pid.reset(env_ids)
        self._motors.reset(env_ids, omega_rad_s=Q250.hover_omega_rad_s)
        self._allocator_saturated[env_ids] = False
        self._last_success[env_ids] = False
        self._last_crashed[env_ids] = False
        for value in self._episode_sums.values():
            value[env_ids] = 0.0

        # Reset vehicle level at local (0, 0, spawn_z). This is intentionally simple in Phase 2.
        n = len(env_ids)
        root_pos = self._terrain.env_origins[env_ids].clone()
        root_pos[:, 2] += self.cfg.spawn_z_m
        root_quat = torch.zeros((n, 4), dtype=torch.float32, device=self.device)
        root_quat[:, 0] = 1.0
        root_pose = torch.cat((root_pos, root_quat), dim=-1)
        root_vel = torch.zeros((n, 6), dtype=torch.float32, device=self.device)
        self._robot.write_root_pose_to_sim(root_pose, env_ids)
        self._robot.write_root_velocity_to_sim(root_vel, env_ids)

        # Curriculum target in local environment coordinates.
        bounds = curriculum_bounds(self.common_step_counter)
        target_local = torch.empty((n, 3), dtype=torch.float32, device=self.device)
        target_local[:, 0:2].uniform_(-bounds.xy_extent_m, bounds.xy_extent_m)
        target_local[:, 2].uniform_(bounds.z_min_m, bounds.z_max_m)

        # Avoid trivial targets sitting almost exactly on the reset pose.
        spawn_local = torch.tensor(
            [0.0, 0.0, self.cfg.spawn_z_m], dtype=torch.float32, device=self.device
        )
        too_close = torch.linalg.norm(target_local - spawn_local, dim=-1) < 0.60
        target_local[too_close, 0] = 0.80

        self._desired_pos_w[env_ids] = self._terrain.env_origins[env_ids] + target_local
        self._previous_distance[env_ids] = torch.linalg.norm(
            self._desired_pos_w[env_ids] - root_pos, dim=-1
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Q250FlyToPoint/goal"
                marker_cfg.markers["cuboid"].size = (0.18, 0.18, 0.18)
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            self.goal_pos_visualizer.set_visibility(True)
        elif hasattr(self, "goal_pos_visualizer"):
            self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if hasattr(self, "goal_pos_visualizer"):
            self.goal_pos_visualizer.visualize(self._desired_pos_w)
