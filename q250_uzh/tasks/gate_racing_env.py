from __future__ import annotations

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
from q250_uzh.gate_racing_math import (
    GateRacingRewardCfg,
    advance_gate_indices,
    compute_gate_racing_reward,
    detect_gate_crossing,
    gate_curriculum,
    signed_gate_distance,
)
from q250_uzh.isaac.torch_dynamics import Q250TorchMotorBank
from q250_uzh.rl_control import CTBRActionCfg, TorchBodyRatePID, TorchMotorAllocator, map_actions_to_ctbr


@configclass
class GateRacingEnvCfg(DirectRLEnvCfg):
    """Privileged-state gate racing curriculum.

    v0.4.0 deliberately keeps the same 12-D observation and 4-D CTBR action
    interface as Fly-to-Point. Gates remain vertical and point along +X so the
    first gate-racing phase focuses on plane crossing and multi-gate sequencing.
    """

    episode_length_s = 10.0
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
        env_spacing=20.0,
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

    max_gates = 3
    gate_frame_thickness_m = 0.08
    spawn_z_m = 1.5
    min_z_m = 0.15
    max_z_m = 5.0
    max_xy_from_origin_m = 9.5

    action_cfg = CTBRActionCfg()
    reward_cfg = GateRacingRewardCfg()


class GateRacingEnv(DirectRLEnv):
    """Q250 gate racing task with a large-gate -> standard-gate -> three-gate curriculum.

    Observation (12): [current_gate_center_b(3), lin_vel_b(3), projected_gravity_b(3), ang_vel_b(3)]
    Action (4): normalized [collective, p_cmd, q_cmd, r_cmd].

    A gate is passed only when the Q250 crosses its plane from the negative side
    to the positive side while the vehicle center lies inside the rectangular
    opening. Crossing the plane outside the opening terminates the episode as a
    missed gate. In v0.4.0 the gate frames are visual markers; plane-crossing
    geometry is the authoritative task rule.
    """

    cfg: GateRacingEnvCfg

    def __init__(self, cfg: GateRacingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self._collective_cmd_n = torch.full(
            (self.num_envs,), Q250.mass_kg * Q250.gravity_m_s2, dtype=torch.float32, device=self.device
        )
        self._rate_cmd_rad_s = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

        self._gate_centers_w = torch.zeros(
            (self.num_envs, self.cfg.max_gates, 3), dtype=torch.float32, device=self.device
        )
        self._gate_count = torch.ones((self.num_envs,), dtype=torch.long, device=self.device)
        self._gate_width_m = torch.full((self.num_envs,), 3.0, dtype=torch.float32, device=self.device)
        self._gate_height_m = torch.full((self.num_envs,), 3.0, dtype=torch.float32, device=self.device)
        self._current_gate_idx = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._previous_signed_distance = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._previous_gate_distance = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._step_gate_progress = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

        self._last_gate_passed = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._last_gate_missed = torch.zeros_like(self._last_gate_passed)
        self._last_race_finished = torch.zeros_like(self._last_gate_passed)
        self._last_crashed = torch.zeros_like(self._last_gate_passed)
        self._allocator_saturated = torch.zeros_like(self._last_gate_passed)
        self._allocator_saturated_policy = torch.zeros_like(self._last_gate_passed)

        self._rate_pid = TorchBodyRatePID(self.num_envs, self.device)
        self._allocator = TorchMotorAllocator(self.device)
        self._motors = Q250TorchMotorBank(self.num_envs, self.device)
        self._motors.reset(omega_rad_s=Q250.hover_omega_rad_s)

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            for key in ("progress", "gate", "finish", "crash", "action", "time")
        }
        self._gates_passed_episode = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._saturation_policy_steps = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._policy_steps_episode = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

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
        self._allocator_saturated_policy.zero_()
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
        self._allocator_saturated_policy |= self._allocator_saturated
        self._motors.step_omega_command(omega_cmd, self.physics_dt)
        forces_b, torques_b = self._motors.wrench()

        composer = getattr(self._robot, "permanent_wrench_composer", None)
        if composer is not None:
            composer.set_forces_and_torques(forces=forces_b, torques=torques_b)
        else:
            self._robot.set_external_force_and_torque(forces_b, torques_b, is_global=False)

    def _current_gate_center_w(self) -> torch.Tensor:
        rows = torch.arange(self.num_envs, device=self.device)
        return self._gate_centers_w[rows, self._current_gate_idx]

    def _get_observations(self) -> dict[str, torch.Tensor]:
        current_gate_w = self._current_gate_center_w()
        current_gate_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            current_gate_w,
        )
        obs = torch.cat(
            (
                current_gate_b,
                self._robot.data.root_lin_vel_b,
                self._robot.data.projected_gravity_b,
                self._robot.data.root_ang_vel_b,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos_w = self._robot.data.root_pos_w
        current_center = self._current_gate_center_w()
        normal_w = torch.zeros_like(current_center)
        normal_w[:, 0] = 1.0
        signed = signed_gate_distance(pos_w, current_center, normal_w)
        lateral_y = pos_w[:, 1] - current_center[:, 1]
        lateral_z = pos_w[:, 2] - current_center[:, 2]

        half_w = self._gate_width_m * 0.5
        half_h = self._gate_height_m * 0.5
        self._last_gate_passed, self._last_gate_missed, _ = detect_gate_crossing(
            self._previous_signed_distance,
            signed,
            lateral_y,
            lateral_z,
            half_width_m=half_w,
            half_height_m=half_h,
        )
        current_gate_distance = torch.linalg.norm(pos_w - current_center, dim=-1)
        self._step_gate_progress = self._previous_gate_distance - current_gate_distance

        new_idx, self._last_race_finished = advance_gate_indices(
            self._current_gate_idx, self._gate_count, self._last_gate_passed
        )
        self._gates_passed_episode += self._last_gate_passed.float()
        self._current_gate_idx.copy_(new_idx)

        # Prime both plane-crossing and center-progress state against the next gate after a clean pass.
        self._previous_signed_distance.copy_(signed)
        self._previous_gate_distance.copy_(current_gate_distance)
        continue_ids = torch.nonzero(self._last_gate_passed & (~self._last_race_finished), as_tuple=False).squeeze(-1)
        if continue_ids.numel() > 0:
            rows = continue_ids
            next_center = self._gate_centers_w[rows, self._current_gate_idx[rows]]
            next_normal = torch.zeros_like(next_center)
            next_normal[:, 0] = 1.0
            self._previous_signed_distance[rows] = signed_gate_distance(
                pos_w[rows], next_center, next_normal
            )
            self._previous_gate_distance[rows] = torch.linalg.norm(pos_w[rows] - next_center, dim=-1)

        local_pos = pos_w - self._terrain.env_origins
        too_low = local_pos[:, 2] < self.cfg.min_z_m
        too_high = local_pos[:, 2] > self.cfg.max_z_m
        too_far_xy = torch.linalg.norm(local_pos[:, :2], dim=-1) > self.cfg.max_xy_from_origin_m
        self._last_crashed = self._last_gate_missed | too_low | too_high | too_far_xy

        terminated = self._last_race_finished | self._last_crashed
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _get_rewards(self) -> torch.Tensor:
        reward, parts = compute_gate_racing_reward(
            gate_progress_m=self._step_gate_progress,
            gate_passed=self._last_gate_passed,
            race_finished=self._last_race_finished,
            crashed=self._last_crashed,
            actions=self._actions,
            cfg=self.cfg.reward_cfg,
        )
        for key, value in parts.items():
            self._episode_sums[key] += value
        self._saturation_policy_steps += self._allocator_saturated_policy.float()
        self._policy_steps_episode += 1.0
        return reward

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        if self.common_step_counter > 0 and len(env_ids) > 0:
            self.extras["log"] = {}
            for key, value in self._episode_sums.items():
                self.extras["log"][f"Episode_Reward/{key}"] = torch.mean(value[env_ids]).item()

            gate_fraction = self._gates_passed_episode[env_ids] / self._gate_count[env_ids].clamp_min(1).float()
            sat_rate = self._saturation_policy_steps[env_ids] / self._policy_steps_episode[env_ids].clamp_min(1.0)
            self.extras["log"]["Metrics/race_success_rate"] = torch.mean(
                self._last_race_finished[env_ids].float()
            ).item()
            self.extras["log"]["Metrics/gate_completion_fraction"] = torch.mean(gate_fraction).item()
            self.extras["log"]["Metrics/missed_gate_rate"] = torch.mean(
                self._last_gate_missed[env_ids].float()
            ).item()
            self.extras["log"]["Metrics/allocator_saturation_rate"] = torch.mean(sat_rate).item()
            self.extras["log"]["Metrics/episode_time_s"] = torch.mean(
                self.episode_length_buf[env_ids].float() * self.step_dt
            ).item()
            self.extras["log"]["Curriculum/stage"] = float(
                gate_curriculum(self.common_step_counter).stage
            )

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        self._actions[env_ids] = 0.0
        self._collective_cmd_n[env_ids] = Q250.mass_kg * Q250.gravity_m_s2
        self._rate_cmd_rad_s[env_ids] = 0.0
        self._rate_pid.reset(env_ids)
        self._motors.reset(env_ids, omega_rad_s=Q250.hover_omega_rad_s)
        self._allocator_saturated[env_ids] = False
        self._allocator_saturated_policy[env_ids] = False
        self._last_gate_passed[env_ids] = False
        self._last_gate_missed[env_ids] = False
        self._last_race_finished[env_ids] = False
        self._last_crashed[env_ids] = False
        self._step_gate_progress[env_ids] = 0.0
        self._gates_passed_episode[env_ids] = 0.0
        self._saturation_policy_steps[env_ids] = 0.0
        self._policy_steps_episode[env_ids] = 0.0
        for value in self._episode_sums.values():
            value[env_ids] = 0.0

        n = len(env_ids)
        root_pos = self._terrain.env_origins[env_ids].clone()
        root_pos[:, 2] += self.cfg.spawn_z_m
        root_quat = torch.zeros((n, 4), dtype=torch.float32, device=self.device)
        root_quat[:, 0] = 1.0
        root_pose = torch.cat((root_pos, root_quat), dim=-1)
        root_vel = torch.zeros((n, 6), dtype=torch.float32, device=self.device)
        self._robot.write_root_pose_to_sim(root_pose, env_ids)
        self._robot.write_root_velocity_to_sim(root_vel, env_ids)

        curriculum = gate_curriculum(self.common_step_counter)
        self._gate_count[env_ids] = curriculum.gate_count
        self._gate_width_m[env_ids] = curriculum.width_m
        self._gate_height_m[env_ids] = curriculum.height_m
        self._current_gate_idx[env_ids] = 0

        centers_local = torch.zeros((n, self.cfg.max_gates, 3), dtype=torch.float32, device=self.device)
        base_x = torch.tensor([3.0, 5.5, 8.0], dtype=torch.float32, device=self.device)
        centers_local[:, :, 0] = base_x
        centers_local[:, :, 0] += torch.empty((n, self.cfg.max_gates), device=self.device).uniform_(-0.25, 0.25)
        centers_local[:, :, 1].uniform_(-curriculum.y_extent_m, curriculum.y_extent_m)
        centers_local[:, :, 2].uniform_(curriculum.z_min_m, curriculum.z_max_m)

        # For single-gate stages only gate 0 is active. In stage 0 keep it especially easy.
        if curriculum.stage == 0:
            centers_local[:, 0, 0].uniform_(2.5, 3.5)
            centers_local[:, 0, 1].uniform_(-0.5, 0.5)
        elif curriculum.stage == 1:
            centers_local[:, 0, 0].uniform_(3.0, 4.0)
        else:
            # Smooth randomized zig-zag: adjacent gates move progressively rather than
            # teleporting between opposite sides of the track. This makes Stage 2 learnable
            # before v0.5.0 introduces orientation and aggressive corner geometry.
            y0 = torch.empty((n,), device=self.device).uniform_(-1.0, 1.0)
            y1 = torch.clamp(y0 + torch.empty((n,), device=self.device).uniform_(-1.0, 1.0), -1.6, 1.6)
            y2 = torch.clamp(y1 + torch.empty((n,), device=self.device).uniform_(-1.0, 1.0), -1.6, 1.6)
            centers_local[:, 0, 1] = y0
            centers_local[:, 1, 1] = y1
            centers_local[:, 2, 1] = y2

            z0 = torch.empty((n,), device=self.device).uniform_(1.1, 2.2)
            z1 = torch.clamp(z0 + torch.empty((n,), device=self.device).uniform_(-0.6, 0.6), 0.9, 2.7)
            z2 = torch.clamp(z1 + torch.empty((n,), device=self.device).uniform_(-0.6, 0.6), 0.9, 2.7)
            centers_local[:, 0, 2] = z0
            centers_local[:, 1, 2] = z1
            centers_local[:, 2, 2] = z2

        self._gate_centers_w[env_ids] = self._terrain.env_origins[env_ids, None, :] + centers_local

        normal = torch.zeros((n, 3), dtype=torch.float32, device=self.device)
        normal[:, 0] = 1.0
        first_gate = self._gate_centers_w[env_ids, 0]
        self._previous_signed_distance[env_ids] = signed_gate_distance(
            root_pos,
            first_gate,
            normal,
        )
        self._previous_gate_distance[env_ids] = torch.linalg.norm(root_pos - first_gate, dim=-1)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "gate_frame_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Q250GateRacing/frames"
                marker_cfg.markers["cuboid"].size = (1.0, 1.0, 1.0)
                marker_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.35, 0.03), metallic=0.0
                )
                self.gate_frame_visualizer = VisualizationMarkers(marker_cfg)
            if not hasattr(self, "current_gate_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Q250GateRacing/current_center"
                marker_cfg.markers["cuboid"].size = (0.12, 0.12, 0.12)
                marker_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.15, 1.0, 0.15), metallic=0.0
                )
                self.current_gate_visualizer = VisualizationMarkers(marker_cfg)
            self.gate_frame_visualizer.set_visibility(True)
            self.current_gate_visualizer.set_visibility(True)
        else:
            if hasattr(self, "gate_frame_visualizer"):
                self.gate_frame_visualizer.set_visibility(False)
            if hasattr(self, "current_gate_visualizer"):
                self.current_gate_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not hasattr(self, "gate_frame_visualizer"):
            return

        gate_slots = torch.arange(self.cfg.max_gates, device=self.device)[None, :]
        active = gate_slots < self._gate_count[:, None]
        centers = self._gate_centers_w[active]
        if centers.numel() == 0:
            return

        widths = self._gate_width_m[:, None].expand(-1, self.cfg.max_gates)[active]
        heights = self._gate_height_m[:, None].expand(-1, self.cfg.max_gates)[active]
        t = float(self.cfg.gate_frame_thickness_m)

        left = centers.clone(); left[:, 1] -= widths * 0.5 + t * 0.5
        right = centers.clone(); right[:, 1] += widths * 0.5 + t * 0.5
        top = centers.clone(); top[:, 2] += heights * 0.5 + t * 0.5
        bottom = centers.clone(); bottom[:, 2] -= heights * 0.5 + t * 0.5
        translations = torch.cat((left, right, top, bottom), dim=0)

        vertical_scale = torch.stack(
            (torch.full_like(widths, t), torch.full_like(widths, t), heights + 2.0 * t), dim=-1
        )
        horizontal_scale = torch.stack(
            (torch.full_like(widths, t), widths + 2.0 * t, torch.full_like(widths, t)), dim=-1
        )
        scales = torch.cat((vertical_scale, vertical_scale, horizontal_scale, horizontal_scale), dim=0)
        self.gate_frame_visualizer.visualize(translations=translations, scales=scales)
        self.current_gate_visualizer.visualize(translations=self._current_gate_center_w())
