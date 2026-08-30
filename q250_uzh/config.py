from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Q250Parameters:
    """Physical parameters for Teddy's Q250 first-principles model.

    Body frame convention used throughout this project: FLU / Z-up
      +x: forward
      +y: left
      +z: up
    Rotor numbering viewed from above:
      M1 front-left  CCW
      M2 front-right CW
      M3 rear-right CCW
      M4 rear-left   CW
    """

    mass_kg: float = 1.0006
    gravity_m_s2: float = 9.80665

    inertia_x_kg_m2: float = 0.00517
    inertia_y_kg_m2: float = 0.00484
    inertia_z_kg_m2: float = 0.00750

    wheelbase_diagonal_m: float = 0.250

    # Mean of the three identified fits in the uploaded motor/propeller tests.
    kt_n_per_rad_s_sq: float = 1.3287717252618608e-6
    kq_nm_per_rad_s_sq: float = 1.772957417327994e-8

    # The uploaded high-frequency files contain no usable time-series samples.
    # 0.10 s is therefore intentionally provisional and should be re-identified.
    motor_tau_s: float = 0.10

    # Conservative limits from the measured table.
    pwm_min_us: float = 1000.0
    pwm_max_us: float = 1800.0

    arm_xy_m: float = field(init=False)
    motor_positions_m: tuple[tuple[float, float, float], ...] = field(init=False)
    rotor_reaction_sign: tuple[float, float, float, float] = field(init=False)
    hover_thrust_per_motor_n: float = field(init=False)
    hover_omega_rad_s: float = field(init=False)
    hover_rpm_model: float = field(init=False)
    inertia_equivalent_box_m: tuple[float, float, float] = field(init=False)

    def __post_init__(self) -> None:
        a = self.wheelbase_diagonal_m / (2.0 * math.sqrt(2.0))
        object.__setattr__(self, "arm_xy_m", a)
        object.__setattr__(
            self,
            "motor_positions_m",
            (
                (+a, +a, 0.0),  # M1 front-left, CCW
                (+a, -a, 0.0),  # M2 front-right, CW
                (-a, -a, 0.0),  # M3 rear-right, CCW
                (-a, +a, 0.0),  # M4 rear-left, CW
            ),
        )
        # Reaction torque on the BODY in FLU/Z-up.
        object.__setattr__(self, "rotor_reaction_sign", (-1.0, +1.0, -1.0, +1.0))

        hover_t = self.mass_kg * self.gravity_m_s2 / 4.0
        hover_w = math.sqrt(hover_t / self.kt_n_per_rad_s_sq)
        object.__setattr__(self, "hover_thrust_per_motor_n", hover_t)
        object.__setattr__(self, "hover_omega_rad_s", hover_w)
        object.__setattr__(self, "hover_rpm_model", hover_w * 60.0 / (2.0 * math.pi))

        # A uniform cuboid with these dimensions has exactly the measured diagonal inertia.
        # This lets PhysX reproduce Ix/Iy/Iz without requiring a custom USD inertia authoring step.
        m = self.mass_kg
        ix, iy, iz = self.inertia_x_kg_m2, self.inertia_y_kg_m2, self.inertia_z_kg_m2
        x2 = 6.0 * (iy + iz - ix) / m
        y2 = 6.0 * (ix + iz - iy) / m
        z2 = 6.0 * (ix + iy - iz) / m
        if min(x2, y2, z2) <= 0.0:
            raise ValueError("Measured inertia is not realizable by an equivalent cuboid.")
        object.__setattr__(self, "inertia_equivalent_box_m", (math.sqrt(x2), math.sqrt(y2), math.sqrt(z2)))


Q250 = Q250Parameters()
