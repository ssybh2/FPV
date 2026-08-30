from __future__ import annotations

import math

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
)
from q250_uzh.isaac.torch_dynamics import Q250TorchMotorBank
from q250_uzh.lookahead_racing_math import (
    build_lookahead_observation,
    gate_basis_from_yaw_pitch,
    gate_local_coordinates,
    gate_quat_wxyz_from_yaw_pitch,
    lookahead_alignment_reward,
    lookahead_curriculum,
    quat_rotate_inverse_wxyz,
)
from q250_uzh.rl_control import CTBRActionCfg, TorchBodyRatePID, TorchMotorAllocator, map_actions_to_ctbr


@configclass
class LookAheadRacingEnvCfg(DirectRLEnvCfg):
    """v0.5.0 true-racing task with current+next gate look-ahead.

    Observation keeps the v0.4.0 12-D prefix intact, then appends 9 new
    features so the old model can be transferred without changing its initial
    behavior:
      [current_gate_b, lin_vel_b, gravity_b, ang_vel_b, next_gate_b,
       current_normal_b, next_normal_b] -> 21D.
    """

    episode_length_s = 12.0
    decimation = 4
    action_space = 4
    observation_space = 21
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
        env_spacing=24.0,
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
                diffuse_color=(0.10, 0.34, 0.52), metallic=0.25
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.5), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    max_gates = 5
    gate_frame_thickness_m = 0.08
    spawn_z_m = 1.5
    min_z_m = 0.15
    max_z_m = 5.5
    max_xy_from_origin_m = 15.5

    action_cfg = CTBRActionCfg()
    reward_cfg = GateRacingRewardCfg(
        progress_scale=4.0,
        gate_bonus=5.0,
        finish_bonus=16.0,
        crash_penalty=-15.0,
        action_penalty_scale=0.0007,
        time_penalty=-0.02,
        max_progress_per_step_m=0.50,
    )


class LookAheadRacingEnv(DirectRLEnv):
    cfg: LookAheadRacingEnvCfg

    def __init__(self, cfg: LookAheadRacingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self._collective_cmd_n = torch.full(
            (self.num_envs,), Q250.mass_kg * Q250.gravity_m_s2, dtype=torch.float32, device=self.device
        )
        self._rate_cmd_rad_s = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

        shape = (self.num_envs, self.cfg.max_gates, 3)
        self._gate_centers_w = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self._gate_normals_w = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self._gate_right_w = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self._gate_up_w = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self._gate_quats_w = torch.zeros((self.num_envs, self.cfg.max_gates, 4), dtype=torch.float32, device=self.device)

        self._gate_count = torch.full((self.num_envs,), 3, dtype=torch.long, device=self.device)
        self._gate_width_m = torch.full((self.num_envs,), 1.8, dtype=torch.float32, device=self.device)
        self._gate_height_m = torch.full((self.num_envs,), 1.8, dtype=torch.float32, device=self.device)
        self._current_gate_idx = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._previous_signed_distance = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._previous_gate_distance = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._step_gate_progress = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._step_lookahead_reward = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

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
            for key in ("progress", "gate", "finish", "crash", "action", "time", "lookahead")
        }
        self._gates_passed_episode = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._saturation_policy_steps = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._policy_steps_episode = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._speed_sum_episode = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

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
        self._collective_cmd_n, self._rate_cmd_rad_s = map_actions_to_ctbr(self._actions, self.cfg.action_cfg)

    def _apply_action(self):
        torque_cmd = self._rate_pid.update(
            self._rate_cmd_rad_s, self._robot.data.root_ang_vel_b, self.physics_dt
        )
        omega_cmd, self._allocator_saturated = self._allocator.allocate(self._collective_cmd_n, torque_cmd)
        self._allocator_saturated_policy |= self._allocator_saturated
        self._motors.step_omega_command(omega_cmd, self.physics_dt)
        forces_b, torques_b = self._motors.wrench()
        composer = getattr(self._robot, "permanent_wrench_composer", None)
        if composer is not None:
            composer.set_forces_and_torques(forces=forces_b, torques=torques_b)
        else:
            self._robot.set_external_force_and_torque(forces_b, torques_b, is_global=False)

    def _rows(self) -> torch.Tensor:
        return torch.arange(self.num_envs, device=self.device)

    def _current_indices(self):
        rows = self._rows()
        return rows, self._current_gate_idx

    def _next_gate_idx(self) -> torch.Tensor:
        return torch.minimum(self._current_gate_idx + 1, self._gate_count - 1)

    def _current_gate_center_w(self) -> torch.Tensor:
        rows, idx = self._current_indices()
        return self._gate_centers_w[rows, idx]

    def _next_gate_center_w(self) -> torch.Tensor:
        rows = self._rows()
        return self._gate_centers_w[rows, self._next_gate_idx()]

    def _current_gate_geometry(self):
        rows, idx = self._current_indices()
        return (
            self._gate_centers_w[rows, idx],
            self._gate_normals_w[rows, idx],
            self._gate_right_w[rows, idx],
            self._gate_up_w[rows, idx],
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        rows = self._rows()
        current_idx = self._current_gate_idx
        next_idx = self._next_gate_idx()
        current_w = self._gate_centers_w[rows, current_idx]
        next_w = self._gate_centers_w[rows, next_idx]

        current_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, current_w
        )
        next_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, next_w
        )
        current_normal_b = quat_rotate_inverse_wxyz(
            self._robot.data.root_quat_w, self._gate_normals_w[rows, current_idx]
        )
        next_normal_b = quat_rotate_inverse_wxyz(
            self._robot.data.root_quat_w, self._gate_normals_w[rows, next_idx]
        )
        obs = build_lookahead_observation(
            current_b,
            self._robot.data.root_lin_vel_b,
            self._robot.data.projected_gravity_b,
            self._robot.data.root_ang_vel_b,
            next_b,
            current_normal_b,
            next_normal_b,
        )
        return {"policy": obs}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos_w = self._robot.data.root_pos_w
        current_center, normal_w, right_w, up_w = self._current_gate_geometry()
        signed, lateral, vertical = gate_local_coordinates(pos_w, current_center, normal_w, right_w, up_w)

        half_w = self._gate_width_m * 0.5
        half_h = self._gate_height_m * 0.5
        self._last_gate_passed, self._last_gate_missed, _ = detect_gate_crossing(
            self._previous_signed_distance,
            signed,
            lateral,
            vertical,
            half_width_m=half_w,
            half_height_m=half_h,
        )

        current_distance = torch.linalg.norm(pos_w - current_center, dim=-1)
        self._step_gate_progress = self._previous_gate_distance - current_distance

        has_next = self._current_gate_idx < (self._gate_count - 1)
        self._step_lookahead_reward = lookahead_alignment_reward(
            self._robot.data.root_lin_vel_w,
            current_center,
            self._next_gate_center_w(),
            current_distance,
            has_next,
        )

        new_idx, self._last_race_finished = advance_gate_indices(
            self._current_gate_idx, self._gate_count, self._last_gate_passed
        )
        self._gates_passed_episode += self._last_gate_passed.float()
        self._current_gate_idx.copy_(new_idx)

        self._previous_signed_distance.copy_(signed)
        self._previous_gate_distance.copy_(current_distance)
        continue_ids = torch.nonzero(
            self._last_gate_passed & (~self._last_race_finished), as_tuple=False
        ).squeeze(-1)
        if continue_ids.numel() > 0:
            idx = self._current_gate_idx[continue_ids]
            next_center = self._gate_centers_w[continue_ids, idx]
            next_normal = self._gate_normals_w[continue_ids, idx]
            delta = pos_w[continue_ids] - next_center
            self._previous_signed_distance[continue_ids] = torch.sum(delta * next_normal, dim=-1)
            self._previous_gate_distance[continue_ids] = torch.linalg.norm(delta, dim=-1)

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
        parts["lookahead"] = self._step_lookahead_reward
        reward = reward + self._step_lookahead_reward
        for key, value in parts.items():
            self._episode_sums[key] += value
        self._saturation_policy_steps += self._allocator_saturated_policy.float()
        self._policy_steps_episode += 1.0
        self._speed_sum_episode += torch.linalg.norm(self._robot.data.root_lin_vel_w, dim=-1)
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
            mean_speed = self._speed_sum_episode[env_ids] / self._policy_steps_episode[env_ids].clamp_min(1.0)
            episode_time = self.episode_length_buf[env_ids].float() * self.step_dt
            gates_per_second = self._gates_passed_episode[env_ids] / episode_time.clamp_min(self.step_dt)

            self.extras["log"]["Metrics/race_success_rate"] = torch.mean(
                self._last_race_finished[env_ids].float()
            ).item()
            self.extras["log"]["Metrics/gate_completion_fraction"] = torch.mean(gate_fraction).item()
            self.extras["log"]["Metrics/missed_gate_rate"] = torch.mean(
                self._last_gate_missed[env_ids].float()
            ).item()
            self.extras["log"]["Metrics/allocator_saturation_rate"] = torch.mean(sat_rate).item()
            self.extras["log"]["Metrics/episode_time_s"] = torch.mean(episode_time).item()
            self.extras["log"]["Metrics/mean_speed_m_s"] = torch.mean(mean_speed).item()
            self.extras["log"]["Metrics/gates_per_second"] = torch.mean(gates_per_second).item()
            self.extras["log"]["Curriculum/stage"] = float(
                lookahead_curriculum(self.common_step_counter).stage
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
        self._step_lookahead_reward[env_ids] = 0.0
        self._gates_passed_episode[env_ids] = 0.0
        self._saturation_policy_steps[env_ids] = 0.0
        self._policy_steps_episode[env_ids] = 0.0
        self._speed_sum_episode[env_ids] = 0.0
        for value in self._episode_sums.values():
            value[env_ids] = 0.0

        n = len(env_ids)
        root_pos = self._terrain.env_origins[env_ids].clone()
        root_pos[:, 2] += self.cfg.spawn_z_m
        root_quat = torch.zeros((n, 4), dtype=torch.float32, device=self.device)
        root_quat[:, 0] = 1.0
        self._robot.write_root_pose_to_sim(torch.cat((root_pos, root_quat), dim=-1), env_ids)
        self._robot.write_root_velocity_to_sim(
            torch.zeros((n, 6), dtype=torch.float32, device=self.device), env_ids
        )

        curriculum = lookahead_curriculum(self.common_step_counter)
        self._gate_count[env_ids] = curriculum.gate_count
        self._gate_width_m[env_ids] = curriculum.width_m
        self._gate_height_m[env_ids] = curriculum.height_m
        self._current_gate_idx[env_ids] = 0

        centers_local = torch.zeros((n, self.cfg.max_gates, 3), dtype=torch.float32, device=self.device)
        spacing = torch.empty((n, self.cfg.max_gates), device=self.device).uniform_(
            curriculum.min_spacing_m, curriculum.max_spacing_m
        )
        centers_local[:, :, 0] = torch.cumsum(spacing, dim=1)

        y = torch.empty((n,), device=self.device).uniform_(-0.8, 0.8)
        z = torch.empty((n,), device=self.device).uniform_(1.2, 2.2)
        for gate_idx in range(self.cfg.max_gates):
            if gate_idx > 0:
                y = torch.clamp(
                    y + torch.empty((n,), device=self.device).uniform_(-1.0, 1.0),
                    -curriculum.y_extent_m,
                    curriculum.y_extent_m,
                )
                z = torch.clamp(
                    z + torch.empty((n,), device=self.device).uniform_(-0.65, 0.65),
                    curriculum.z_min_m,
                    curriculum.z_max_m,
                )
            centers_local[:, gate_idx, 1] = y
            centers_local[:, gate_idx, 2] = z

        if curriculum.stage == 0:
            # Closest to the v0.4 three-gate distribution for rapid transfer adaptation.
            centers_local[:, 0, 0] = 3.0
            centers_local[:, 1, 0] = 5.5
            centers_local[:, 2, 0] = 8.0

        self._gate_centers_w[env_ids] = self._terrain.env_origins[env_ids, None, :] + centers_local

        yaw = torch.zeros((n, self.cfg.max_gates), dtype=torch.float32, device=self.device)
        pitch = torch.zeros_like(yaw)
        prev_local = torch.zeros((n, 3), dtype=torch.float32, device=self.device)
        prev_local[:, 2] = self.cfg.spawn_z_m
        yaw_jitter = math.radians(curriculum.yaw_jitter_deg)
        pitch_jitter = math.radians(curriculum.pitch_jitter_deg)
        for gate_idx in range(self.cfg.max_gates):
            delta = centers_local[:, gate_idx] - prev_local
            base_yaw = torch.atan2(delta[:, 1], delta[:, 0])
            horizontal = torch.sqrt(delta[:, 0].square() + delta[:, 1].square()).clamp_min(1e-6)
            base_pitch = torch.atan2(delta[:, 2], horizontal)
            if curriculum.stage == 0:
                yaw[:, gate_idx] = 0.0
                pitch[:, gate_idx] = 0.0
            else:
                yaw[:, gate_idx] = base_yaw + torch.empty((n,), device=self.device).uniform_(-yaw_jitter, yaw_jitter)
                pitch[:, gate_idx] = base_pitch + torch.empty((n,), device=self.device).uniform_(-pitch_jitter, pitch_jitter)
            prev_local = centers_local[:, gate_idx]

        normals, rights, ups = gate_basis_from_yaw_pitch(yaw, pitch)
        self._gate_normals_w[env_ids] = normals
        self._gate_right_w[env_ids] = rights
        self._gate_up_w[env_ids] = ups
        self._gate_quats_w[env_ids] = gate_quat_wxyz_from_yaw_pitch(yaw, pitch)

        first_center = self._gate_centers_w[env_ids, 0]
        first_normal = self._gate_normals_w[env_ids, 0]
        first_delta = root_pos - first_center
        self._previous_signed_distance[env_ids] = torch.sum(first_delta * first_normal, dim=-1)
        self._previous_gate_distance[env_ids] = torch.linalg.norm(first_delta, dim=-1)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "gate_frame_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Q250LookAhead/frames"
                marker_cfg.markers["cuboid"].size = (1.0, 1.0, 1.0)
                marker_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.35, 0.03), metallic=0.0
                )
                self.gate_frame_visualizer = VisualizationMarkers(marker_cfg)
            if not hasattr(self, "current_gate_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Q250LookAhead/current_center"
                marker_cfg.markers["cuboid"].size = (0.12, 0.12, 0.12)
                marker_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.15, 1.0, 0.15), metallic=0.0
                )
                self.current_gate_visualizer = VisualizationMarkers(marker_cfg)
            if not hasattr(self, "next_gate_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Q250LookAhead/next_center"
                marker_cfg.markers["cuboid"].size = (0.10, 0.10, 0.10)
                marker_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.10, 0.85, 1.0), metallic=0.0
                )
                self.next_gate_visualizer = VisualizationMarkers(marker_cfg)
            self.gate_frame_visualizer.set_visibility(True)
            self.current_gate_visualizer.set_visibility(True)
            self.next_gate_visualizer.set_visibility(True)
        else:
            for name in ("gate_frame_visualizer", "current_gate_visualizer", "next_gate_visualizer"):
                if hasattr(self, name):
                    getattr(self, name).set_visibility(False)

    def _debug_vis_callback(self, event):
        if not hasattr(self, "gate_frame_visualizer"):
            return
        gate_slots = torch.arange(self.cfg.max_gates, device=self.device)[None, :]
        active = gate_slots < self._gate_count[:, None]
        centers = self._gate_centers_w[active]
        if centers.numel() == 0:
            return
        rights = self._gate_right_w[active]
        ups = self._gate_up_w[active]
        quats = self._gate_quats_w[active]
        widths = self._gate_width_m[:, None].expand(-1, self.cfg.max_gates)[active]
        heights = self._gate_height_m[:, None].expand(-1, self.cfg.max_gates)[active]
        t = float(self.cfg.gate_frame_thickness_m)

        left = centers - rights * (widths * 0.5 + t * 0.5)[:, None]
        right = centers + rights * (widths * 0.5 + t * 0.5)[:, None]
        top = centers + ups * (heights * 0.5 + t * 0.5)[:, None]
        bottom = centers - ups * (heights * 0.5 + t * 0.5)[:, None]
        translations = torch.cat((left, right, top, bottom), dim=0)
        orientations = torch.cat((quats, quats, quats, quats), dim=0)

        vertical_scale = torch.stack(
            (torch.full_like(widths, t), torch.full_like(widths, t), heights + 2.0 * t), dim=-1
        )
        horizontal_scale = torch.stack(
            (torch.full_like(widths, t), widths + 2.0 * t, torch.full_like(widths, t)), dim=-1
        )
        scales = torch.cat((vertical_scale, vertical_scale, horizontal_scale, horizontal_scale), dim=0)
        self.gate_frame_visualizer.visualize(
            translations=translations, orientations=orientations, scales=scales
        )
        self.current_gate_visualizer.visualize(translations=self._current_gate_center_w())
        self.next_gate_visualizer.visualize(translations=self._next_gate_center_w())
