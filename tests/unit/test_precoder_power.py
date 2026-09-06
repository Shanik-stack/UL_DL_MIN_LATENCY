import unittest

import numpy as np
import torch

from latency_optimization.precoders.power import (
    POWER_PROJECTION_SAFETY_MARGIN,
    cap_matrix_power_numpy,
    cap_matrix_power_torch,
    joint_power_scale_numpy,
    joint_power_scale_torch,
    normalize_matrix_power_numpy,
    normalize_matrix_power_torch,
)


class PrecoderPowerTests(unittest.TestCase):
    def test_normalize_numpy_and_torch_match(self) -> None:
        matrix_np = np.asarray([[1 + 2j, 3 - 1j], [0.5j, -2]], dtype=np.complex64)
        matrix_torch = torch.as_tensor(matrix_np)

        normalized_np = normalize_matrix_power_numpy(matrix_np, 7.0)
        normalized_torch = normalize_matrix_power_torch(matrix_torch, 7.0)

        np.testing.assert_allclose(normalized_torch.detach().numpy(), normalized_np, rtol=1e-6, atol=1e-6)
        expected_power = 7.0 * (1.0 - POWER_PROJECTION_SAFETY_MARGIN) ** 2
        self.assertAlmostEqual(float(np.linalg.norm(normalized_np, ord="fro") ** 2), expected_power, places=5)

    def test_cap_leaves_feasible_matrix_unchanged(self) -> None:
        matrix_np = np.asarray([[0.25 + 0.1j]], dtype=np.complex64)
        matrix_torch = torch.as_tensor(matrix_np)

        self.assertIs(cap_matrix_power_numpy(matrix_np, 1.0), matrix_np)
        self.assertIs(cap_matrix_power_torch(matrix_torch, 1.0), matrix_torch)

    def test_joint_scale_matches_numpy_and_torch(self) -> None:
        matrices_np = [
            np.asarray([[1 + 1j]], dtype=np.complex64),
            np.asarray([[2 - 1j]], dtype=np.complex64),
        ]
        scale_np = joint_power_scale_numpy(matrices_np, 5.0)
        scale_torch = joint_power_scale_torch([torch.as_tensor(value) for value in matrices_np], 5.0)

        self.assertIsNotNone(scale_np)
        self.assertIsNotNone(scale_torch)
        self.assertAlmostEqual(float(scale_torch.detach().cpu()), float(scale_np), places=6)


if __name__ == "__main__":
    unittest.main()
