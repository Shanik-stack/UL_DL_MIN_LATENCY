import unittest

import numpy as np
import torch

from latency_optimization.physics.finite_blocklength import (
    finite_blocklength_mimo,
    q_inverse,
)


class FiniteBlocklengthRateTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(17)
        self.channel = (
            rng.standard_normal((3, 2)) + 1j * rng.standard_normal((3, 2))
        ) / np.sqrt(2.0)
        self.precoder = (
            rng.standard_normal((2, 1)) + 1j * rng.standard_normal((2, 1))
        ) / np.sqrt(2.0)
        base = (
            rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
        ) / np.sqrt(2.0)
        self.covariance = base @ base.conj().T + 0.4 * np.eye(3)

    def test_rate_returns_finite_torch_values(self) -> None:
        torch_result = finite_blocklength_mimo(
            torch.tensor(self.channel, dtype=torch.complex128),
            torch.tensor(self.precoder, dtype=torch.complex128),
            torch.tensor(self.covariance, dtype=torch.complex128),
            n_kl=37,
            epsilon=1.0e-14,
        )

        self.assertTrue(torch.isfinite(torch_result.capacity))
        self.assertTrue(torch.isfinite(torch_result.dispersion))
        self.assertTrue(torch.isfinite(torch_result.penalty))
        self.assertTrue(torch.isfinite(torch_result.rate))

    def test_rate_matches_fixed_numerical_reference(self) -> None:
        result = finite_blocklength_mimo(
            torch.tensor(self.channel, dtype=torch.complex128),
            torch.tensor(self.precoder, dtype=torch.complex128),
            torch.tensor(self.covariance, dtype=torch.complex128),
            n_kl=37,
            epsilon=1.0e-14,
        )

        self.assertAlmostEqual(float(result.capacity), 4.13436886051956, places=12)
        self.assertAlmostEqual(float(result.dispersion), 2.07462042280595, places=12)
        self.assertAlmostEqual(float(result.penalty), 1.81163786555407, places=12)
        self.assertAlmostEqual(float(result.rate), 2.32273099496549, places=12)

    def test_torch_rate_is_differentiable_with_respect_to_precoder(self) -> None:
        precoder = torch.tensor(self.precoder, dtype=torch.complex128, requires_grad=True)
        result = finite_blocklength_mimo(
            torch.tensor(self.channel, dtype=torch.complex128),
            precoder,
            torch.tensor(self.covariance, dtype=torch.complex128),
            n_kl=37,
            epsilon=1.0e-14,
        )
        result.rate.backward()

        self.assertIsNotNone(precoder.grad)
        self.assertTrue(torch.isfinite(precoder.grad).all())
        self.assertGreater(float(torch.linalg.vector_norm(precoder.grad)), 0.0)

    def test_small_epsilon_is_not_clamped_to_a_different_target(self) -> None:
        self.assertGreater(q_inverse(1.0e-14), q_inverse(1.0e-12))

    def test_non_positive_blocklength_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            finite_blocklength_mimo(
                torch.tensor(self.channel, dtype=torch.complex128),
                torch.tensor(self.precoder, dtype=torch.complex128),
                torch.tensor(self.covariance, dtype=torch.complex128),
                n_kl=0,
                epsilon=1.0e-5,
            )


if __name__ == "__main__":
    unittest.main()
