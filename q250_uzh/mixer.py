from __future__ import annotations

from collections.abc import Sequence

from .config import Q250


def wrench_from_motor_omega(
    motor_omega_rad_s: Sequence[float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return body-frame force and torque for M1..M4 in FLU/Z-up.

    Force is expressed at the center of mass. Roll and pitch moments include the
    arm cross thrust contribution; yaw is the propeller reaction torque.
    """
    if len(motor_omega_rad_s) != 4:
        raise ValueError("Exactly four motor speeds are required (M1..M4).")

    thrust = [Q250.kt_n_per_rad_s_sq * max(0.0, float(w)) ** 2 for w in motor_omega_rad_s]
    reaction = [
        sign * Q250.kq_nm_per_rad_s_sq * max(0.0, float(w)) ** 2
        for sign, w in zip(Q250.rotor_reaction_sign, motor_omega_rad_s, strict=True)
    ]

    tau_x = 0.0
    tau_y = 0.0
    for (x, y, _), t in zip(Q250.motor_positions_m, thrust, strict=True):
        # r x F for F=[0,0,T] -> [yT, -xT, 0]
        tau_x += y * t
        tau_y += -x * t
    tau_z = sum(reaction)

    return (0.0, 0.0, sum(thrust)), (tau_x, tau_y, tau_z)


def mixer_matrix() -> tuple[tuple[float, float, float, float], ...]:
    a = Q250.arm_xy_m
    kt = Q250.kt_n_per_rad_s_sq
    kq = Q250.kq_nm_per_rad_s_sq
    return (
        (kt, kt, kt, kt),
        (a * kt, -a * kt, -a * kt, a * kt),
        (-a * kt, -a * kt, a * kt, a * kt),
        (-kq, +kq, -kq, +kq),
    )
