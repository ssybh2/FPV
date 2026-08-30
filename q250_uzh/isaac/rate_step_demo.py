from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from isaaclab.app import AppLauncher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Q250 body-rate PID step/pulse validation")
    parser.add_argument("--axis", choices=("roll", "pitch", "yaw"), default="roll")
    parser.add_argument("--rate-deg-s", type=float, default=100.0, help="commanded body rate during the pulse")
    parser.add_argument("--duration", type=float, default=2.0, help="simulation seconds")
    parser.add_argument("--step-start", type=float, default=0.50, help="pulse start time")
    parser.add_argument("--step-end", type=float, default=0.90, help="pulse end time")
    parser.add_argument("--physics-dt", type=float, default=1.0 / 240.0, help="physics/control timestep")
    parser.add_argument("--collective-scale", type=float, default=1.0, help="hover collective multiplier")
    parser.add_argument("--log-path", type=str, default="", help="CSV output path")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _set_body_wrench(asset, forces, torques) -> str:
    composer = getattr(asset, "permanent_wrench_composer", None)
    if composer is not None:
        composer.set_forces_and_torques(forces=forces, torques=torques)
        return "permanent_wrench_composer"
    asset.set_external_force_and_torque(forces=forces, torques=torques, is_global=False)
    return "set_external_force_and_torque"


def _default_log_path(axis: str, rate_deg_s: float) -> Path:
    root = Path(__file__).resolve().parents[2]
    rate_tag = f"{rate_deg_s:+.0f}".replace("+", "p").replace("-", "m")
    return root / "logs" / "rate_steps" / f"{axis}_{rate_tag}dps.csv"


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.duration <= 0.0:
        parser.error("--duration must be positive")
    if args.physics_dt <= 0.0:
        parser.error("--physics-dt must be positive")
    if args.step_start < 0.0 or args.step_end <= args.step_start or args.step_end >= args.duration:
        parser.error("require 0 <= step-start < step-end < duration")
    if args.collective_scale <= 0.0:
        parser.error("--collective-scale must be positive")

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg

    from q250_uzh.allocator import MotorAllocator
    from q250_uzh.config import Q250
    from q250_uzh.initial_state import corrected_warm_start_root_state
    from q250_uzh.isaac.torch_dynamics import Q250TorchMotorBank
    from q250_uzh.rate_controller import BodyRatePID
    from q250_uzh.rate_step_profile import RateStepProfile

    sim_cfg = sim_utils.SimulationCfg(
        dt=args.physics_dt,
        device=args.device,
        gravity=(0.0, 0.0, -Q250.gravity_m_s2),
        physx=sim_utils.PhysxCfg(enable_external_forces_every_iteration=True),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(2.7, 2.7, 2.3), target=(0.0, 0.0, 1.6))

    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8)).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
    )

    drone_cfg = RigidObjectCfg(
        prim_path="/World/Q250",
        spawn=sim_utils.CuboidCfg(
            size=Q250.inertia_equivalent_box_m,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=100.0,
                max_angular_velocity=100.0,
                solver_velocity_iteration_count=1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=Q250.mass_kg),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.12, 0.28, 0.42), metallic=0.25
            ),
        ),
        # Higher than the hover demo because a body-rate-only pulse intentionally tilts the craft.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    drone = RigidObject(drone_cfg)

    sim.reset()
    drone.update(args.physics_dt)
    warm_state = corrected_warm_start_root_state(
        drone.data.root_state_w, position_xyz=(0.0, 0.0, 2.0)
    )
    drone.write_root_state_to_sim(warm_state)
    drone.update(args.physics_dt)

    motors = Q250TorchMotorBank(num_envs=1, device=args.device)
    motors.reset(omega_rad_s=Q250.hover_omega_rad_s)
    controller = BodyRatePID()
    allocator = MotorAllocator()
    profile = RateStepProfile(
        axis=args.axis,
        rate_rad_s=math.radians(args.rate_deg_s),
        start_s=args.step_start,
        end_s=args.step_end,
    )
    collective_n = Q250.mass_kg * Q250.gravity_m_s2 * args.collective_scale

    log_path = Path(args.log_path) if args.log_path else _default_log_path(args.axis, args.rate_deg_s)
    if not log_path.is_absolute():
        log_path = Path.cwd() / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n=== Q250 Body-Rate Step Validation ===")
    print(f"axis                    : {args.axis}")
    print(f"command                 : {args.rate_deg_s:+.1f} deg/s")
    print(f"pulse                   : {args.step_start:.2f} -> {args.step_end:.2f} s")
    print(f"control/physics rate    : {1.0 / args.physics_dt:.1f} Hz")
    print(f"collective              : {collective_n:.4f} N ({args.collective_scale:.3f} x hover)")
    print(f"motor tau (provisional) : {Q250.motor_tau_s:.3f} s")
    print(f"log                     : {log_path}")
    print("NOTE: this is a body-rate inner-loop test, not attitude/altitude hold. Some translation is expected.\n")

    header = [
        "time_s",
        "cmd_p_deg_s", "cmd_q_deg_s", "cmd_r_deg_s",
        "p_deg_s", "q_deg_s", "r_deg_s",
        "tau_x_cmd_nm", "tau_y_cmd_nm", "tau_z_cmd_nm",
        "m1_cmd_rpm", "m2_cmd_rpm", "m3_cmd_rpm", "m4_cmd_rpm",
        "m1_rpm", "m2_rpm", "m3_rpm", "m4_rpm",
        "fz_actual_n", "tau_x_actual_nm", "tau_y_actual_nm", "tau_z_actual_nm",
        "allocator_saturated", "z_m", "vz_m_s",
    ]

    sim_time = 0.0
    next_print = 0.0
    wrench_api = "unknown"
    rpm_factor = 60.0 / (2.0 * math.pi)

    with log_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)

        while simulation_app.is_running() and sim_time < args.duration:
            rate_b_tensor = drone.data.root_ang_vel_b[0]
            rate_b = tuple(float(v) for v in rate_b_tensor.detach().cpu().tolist())
            rate_cmd = profile.command_at(sim_time)
            torque_cmd = controller.update(rate_cmd, rate_b, args.physics_dt)
            allocation = allocator.allocate(collective_n, torque_cmd)

            omega_cmd = torch.tensor(
                [allocation.omega_cmd_rad_s], dtype=torch.float32, device=args.device
            )
            motors.step_omega_command(omega_cmd, args.physics_dt)
            force_b, torque_b = motors.wrench()
            wrench_api = _set_body_wrench(drone, force_b, torque_b)
            drone.write_data_to_sim()
            sim.step()
            drone.update(args.physics_dt)
            sim_time += args.physics_dt

            measured = tuple(
                float(v) for v in drone.data.root_ang_vel_b[0].detach().cpu().tolist()
            )
            pos_z = float(drone.data.root_pos_w[0, 2].detach().cpu())
            vz = float(drone.data.root_lin_vel_w[0, 2].detach().cpu())
            actual_force = tuple(float(v) for v in force_b[0, 0].detach().cpu().tolist())
            actual_torque = tuple(float(v) for v in torque_b[0, 0].detach().cpu().tolist())
            actual_rpm = tuple(float(v) * rpm_factor for v in motors.omega[0].detach().cpu().tolist())
            cmd_rpm = tuple(float(v) * rpm_factor for v in allocation.omega_cmd_rad_s)
            cmd_deg = tuple(math.degrees(v) for v in rate_cmd)
            measured_deg = tuple(math.degrees(v) for v in measured)

            writer.writerow(
                [
                    sim_time,
                    *cmd_deg,
                    *measured_deg,
                    *torque_cmd,
                    *cmd_rpm,
                    *actual_rpm,
                    actual_force[2],
                    *actual_torque,
                    int(allocation.saturated),
                    pos_z,
                    vz,
                ]
            )

            if sim_time + 1e-9 >= next_print:
                print(
                    f"t={sim_time:5.2f}s cmd=({cmd_deg[0]:+7.1f},{cmd_deg[1]:+7.1f},{cmd_deg[2]:+7.1f}) "
                    f"rate=({measured_deg[0]:+7.1f},{measured_deg[1]:+7.1f},{measured_deg[2]:+7.1f}) deg/s "
                    f"tau=({torque_cmd[0]:+6.3f},{torque_cmd[1]:+6.3f},{torque_cmd[2]:+6.3f}) Nm "
                    f"sat={int(allocation.saturated)} z={pos_z:5.2f}m"
                )
                next_print += 0.10

    print(f"\nwrench API              : {wrench_api}")
    print(f"CSV saved               : {log_path}")
    print("Next: plot the CSV and inspect rise time, overshoot, settling, saturation, and motor RPM.\n")
    simulation_app.close()


if __name__ == "__main__":
    main()
