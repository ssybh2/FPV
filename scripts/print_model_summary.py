from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from q250_uzh.config import Q250
from q250_uzh.mixer import mixer_matrix
from q250_uzh.motor_model import default_motor_model


def main() -> None:
    motor = default_motor_model()
    hover_pwm = motor.rpm_to_pwm(Q250.hover_rpm_model)
    print("Q250 model summary")
    print("------------------")
    print(f"mass: {Q250.mass_kg:.6f} kg")
    print(f"I: [{Q250.inertia_x_kg_m2:.6f}, {Q250.inertia_y_kg_m2:.6f}, {Q250.inertia_z_kg_m2:.6f}] kg m^2")
    print(f"arm xy: {Q250.arm_xy_m:.9f} m")
    print(f"Kt: {Q250.kt_n_per_rad_s_sq:.12e}")
    print(f"Kq: {Q250.kq_nm_per_rad_s_sq:.12e}")
    print(f"hover thrust/motor: {Q250.hover_thrust_per_motor_n:.6f} N")
    print(f"hover omega (Kt model): {Q250.hover_omega_rad_s:.3f} rad/s")
    print(f"hover rpm (Kt model): {Q250.hover_rpm_model:.1f} rpm")
    print(f"PWM giving that RPM by measured LUT: {hover_pwm:.2f} us")
    print("\nMixer B for [Fz, tx, ty, tz]^T = B [w1^2..w4^2]^T:")
    for row in mixer_matrix():
        print("  [" + ", ".join(f"{v:+.12e}" for v in row) + "]")


if __name__ == "__main__":
    main()
