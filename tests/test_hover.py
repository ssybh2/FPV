import unittest

from q250_uzh.config import Q250


class TestHover(unittest.TestCase):
    def test_hover_per_motor_thrust(self):
        self.assertAlmostEqual(Q250.hover_thrust_per_motor_n, Q250.mass_kg * Q250.gravity_m_s2 / 4.0, places=12)

    def test_inertia_equivalent_cuboid_matches_measured_inertia(self):
        x, y, z = Q250.inertia_equivalent_box_m
        m = Q250.mass_kg
        ix = m * (y * y + z * z) / 12.0
        iy = m * (x * x + z * z) / 12.0
        iz = m * (x * x + y * y) / 12.0
        self.assertAlmostEqual(ix, Q250.inertia_x_kg_m2, places=12)
        self.assertAlmostEqual(iy, Q250.inertia_y_kg_m2, places=12)
        self.assertAlmostEqual(iz, Q250.inertia_z_kg_m2, places=12)


if __name__ == "__main__":
    unittest.main()
