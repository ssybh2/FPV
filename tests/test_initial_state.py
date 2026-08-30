import unittest

import torch

from q250_uzh.initial_state import corrected_warm_start_root_state


class TestInitialState(unittest.TestCase):
    def test_warm_start_rewrites_pose_and_zeros_all_velocities(self):
        # Mimic a state after sim.reset() has already allowed gravity to advance the body.
        state = torch.tensor(
            [[0.0, 0.0, 1.4991, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0817, 0.01, -0.02, 0.03]],
            dtype=torch.float32,
        )

        corrected = corrected_warm_start_root_state(state, position_xyz=(0.0, 0.0, 1.5))

        self.assertAlmostEqual(float(corrected[0, 2]), 1.5, places=6)
        self.assertTrue(torch.equal(corrected[0, 7:], torch.zeros(6, dtype=corrected.dtype)))
        # Do not mutate the input tensor in place.
        self.assertAlmostEqual(float(state[0, 2]), 1.4991, places=6)
        self.assertAlmostEqual(float(state[0, 9]), -0.0817, places=6)


if __name__ == "__main__":
    unittest.main()
