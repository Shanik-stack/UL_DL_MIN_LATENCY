import unittest

import numpy as np
import torch

from latency_optimization.physics.finite_blocklength import (
    finite_blocklength_mimo_numpy,
    finite_blocklength_mimo_torch,
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

    def test_numpy_and_torch_results_match(self) -> None:
        numpy_result = finite_blocklength_mimo_numpy(
            self.channel,
            self.precoder,
            self.covariance,
            n_kl=37,
            epsilon=1.0e-14,
        )
        torch_result = finite_blocklength_mimo_torch(
            torch.tensor(self.channel, dtype=torch.complex128),
            torch.tensor(self.precoder, dtype=torch.complex128),
            torch.tensor(self.covariance, dtype=torch.complex128),
            n_kl=37,
            epsilon=1.0e-14,
        )

        self.assertAlmostEqual(numpy_result.capacity, torch_result.capacity.item(), places=10)
        self.assertAlmostEqual(numpy_result.dispersion, torch_result.dispersion.item(), places=10)
        self.assertAlmostEqual(numpy_result.penalty, torch_result.penalty.item(), places=10)
        self.assertAlmostEqual(numpy_result.rate, torch_result.rate.item(), places=10)

    def test_torch_rate_is_differentiable_with_respect_to_precoder(self) -> None:
        precoder = torch.tensor(self.precoder, dtype=torch.complex128, requires_grad=True)
        result = finite_blocklength_mimo_torch(
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
            finite_blocklength_mimo_numpy(
                self.channel,
                self.precoder,
                self.covariance,
                n_kl=0,
                epsilon=1.0e-5,
            )


if __name__ == "__main__":
    unittest.main()
