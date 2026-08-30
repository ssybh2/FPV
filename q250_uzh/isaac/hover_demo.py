from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Q250 PhysX hover validation")
    parser.add_argument("--duration", type=float, default=8.0, help="simulation seconds")
    parser.add_argument("--physics-dt", type=float, default=1.0 / 240.0, help="physics timestep")
    parser.add_argument(
        "--cold-start",
        action="store_true",
        help="start motors at zero instead of hover omega; the vehicle will initially drop",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _set_body_wrench(asset, forces, torques) -> str:
    """Support Isaac Lab 2.3.0/2.3.1 and the 2.3.2 wrench composer API."""
    composer = getattr(asset, "permanent_wrench_composer", None)
    if composer is not None:
        composer.set_forces_and_torques(forces=forces, torques=torques)
        return "permanent_wrench_composer"
    asset.set_external_force_and_torque(forces=forces, torques=torques, is_global=False)
    return "set_external_force_and_torque"


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg

    from q250_uzh.config import Q250
    from q250_uzh.initial_state import corrected_warm_start_root_state
    from q250_uzh.isaac.torch_dynamics import Q250TorchMotorBank

    sim_cfg = sim_utils.SimulationCfg(
        dt=args.physics_dt,
        device=args.device,
        gravity=(0.0, 0.0, -Q250.gravity_m_s2),
        physx=sim_utils.PhysxCfg(enable_external_forces_every_iteration=True),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(2.4, 2.4, 1.8), target=(0.0, 0.0, 1.2))

    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8)).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
    )

    # First milestone: an inertia-equivalent cuboid. Its uniform-box inertia is exactly
    # the measured Ix/Iy/Iz when mass=1.0006 kg. A detailed Q250 visual USD comes later.
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
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.20, 0.25), metallic=0.25),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.5), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    drone = RigidObject(drone_cfg)

    sim.reset()
    drone.update(args.physics_dt)

    # IMPORTANT: sim.reset()/asset initialization may already have advanced gravity before
    # the first commanded wrench reaches PhysX.  In the observed failure this left exactly
    # vz=-0.0817 m/s; the hover thrust then balanced gravity and preserved that velocity.
    # Re-write the desired pose and zero velocity *after* reset so the validation starts
    # from a true stationary state.
    warm_state = corrected_warm_start_root_state(
        drone.data.root_state_w, position_xyz=(0.0, 0.0, 1.5)
    )
    drone.write_root_state_to_sim(warm_state)
    drone.update(args.physics_dt)

    motors = Q250TorchMotorBank(num_envs=1, device=args.device)
    initial_omega = 0.0 if args.cold_start else Q250.hover_omega_rad_s
    motors.reset(omega_rad_s=initial_omega)
    omega_cmd = torch.full((1, 4), Q250.hover_omega_rad_s, dtype=torch.float32, device=args.device)

    print("\n=== Q250 Hover Validation ===")
    print(f"mass                    : {Q250.mass_kg:.6f} kg")
    print(f"inertia Ix/Iy/Iz        : {Q250.inertia_x_kg_m2:.6f}, {Q250.inertia_y_kg_m2:.6f}, {Q250.inertia_z_kg_m2:.6f} kg m^2")
    print(f"equivalent PhysX box    : {Q250.inertia_equivalent_box_m}")
    print(f"model hover omega       : {Q250.hover_omega_rad_s:.3f} rad/s")
    print(f"model hover RPM         : {Q250.hover_rpm_model:.1f} rpm")
    print(f"motor tau (provisional) : {Q250.motor_tau_s:.3f} s")
    print("warm-start correction   : pose rewritten + 6-D velocity zeroed after sim.reset()")

    sim_time = 0.0
    next_print = 0.0
    wrench_api = "unknown"
    while simulation_app.is_running() and sim_time < args.duration:
        motors.step_omega_command(omega_cmd, args.physics_dt)
        force_b, torque_b = motors.wrench()
        wrench_api = _set_body_wrench(drone, force_b, torque_b)
        drone.write_data_to_sim()
        sim.step()
        drone.update(args.physics_dt)
        sim_time += args.physics_dt

        if sim_time + 1e-9 >= next_print:
            pos = drone.data.root_pos_w[0].detach().cpu().tolist()
            lin = drone.data.root_lin_vel_w[0].detach().cpu().tolist()
            ang = drone.data.root_ang_vel_w[0].detach().cpu().tolist()
            total_t = float(force_b[0, 0, 2].detach().cpu())
            print(
                f"t={sim_time:6.2f}s z={pos[2]:7.4f}m vz={lin[2]:+8.4f}m/s "
                f"|w|={math.sqrt(sum(v*v for v in ang)):7.4f}rad/s T={total_t:7.4f}N"
            )
            next_print += 0.5

    print(f"wrench API              : {wrench_api}")
    print("Expected warm-start behavior: z stays close to 1.5 m with near-zero vertical velocity.")
    print("If it accelerates strongly, stop and inspect coordinate/wrench conventions before any RL training.\n")
    simulation_app.close()


if __name__ == "__main__":
    main()
