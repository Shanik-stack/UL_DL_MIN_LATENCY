"""Uplink finite-blocklength objective shared by all optimization methods."""

from __future__ import annotations

import torch
from torch import nn

from latency_optimization.physics.rate_law import NORMAL_APPROXIMATION_RATE_LAW, RateLaw

from ..physics.rate import evaluate_uplink_rate_tensor


class UplinkPrecoderObjective(nn.Module):
    """Canonical differentiable uplink rate objective and feasibility diagnostics."""

    def __init__(
        self,
        channel: torch.Tensor,
        noise_variance: float,
        epsilon: float,
        payload_bits: float,
        power_limit: float,
        blocklength: int,
        noise_plus_interference_covariance: torch.Tensor | None = None,
        rate_law: RateLaw = NORMAL_APPROXIMATION_RATE_LAW,
    ) -> None:
        """Bind one user/channel/block optimization problem to the shared rate law."""
        super().__init__()
        self.H_kl = channel
        self.sigma2 = float(noise_variance)
        self.epsilon = float(epsilon)
        self.B = float(payload_bits)
        self.P = float(power_limit)
        self.n_kl = int(blocklength)
        self.noise_plus_interference_cov = noise_plus_interference_covariance
        self.rate_law = rate_law

    def set_blocklength(self, blocklength: int) -> None:
        """Retarget the reusable objective to a new candidate n_kl."""
        self.n_kl = int(blocklength)

    def set_payload(self, bits: float) -> None:
        """Retarget the rate constraint to the bits assigned to this block."""
        self.B = float(bits)

    def finite_blocklength_rate(self, precoder: torch.Tensor) -> torch.Tensor:
        """Compute differentiable FBL rate for the bound channel, covariance, and n_kl."""
        identity = torch.eye(self.H_kl.shape[0], dtype=torch.complex64, device=self.H_kl.device)
        covariance = (
            self.sigma2 * identity
            if self.noise_plus_interference_cov is None
            else self.noise_plus_interference_cov.to(device=self.H_kl.device, dtype=torch.complex64)
        )
        return evaluate_uplink_rate_tensor(
            self.H_kl,
            precoder,
            self.sigma2,
            self.epsilon,
            self.n_kl,
            covariance,
            rate_law=self.rate_law,
        ).rate

    def forward(self, precoder: torch.Tensor) -> dict[str, torch.Tensor]:
        """Evaluate one user's beam objective and constraint residuals.

        What: return loss ``-R_fbl``, rate gap ``B/n-R_fbl``, power gap
        ``||F||_F^2-P``, and their positive ReLU violations. Why: the solver maximizes
        physical rate while separately deciding whether the resulting beam can serve
        the requested bits within its blocklength and power budget.
        """
        rate = self.finite_blocklength_rate(precoder)
        power = (torch.linalg.norm(precoder, ord="fro") ** 2).real
        rate_gap = (self.B / self.n_kl) - rate
        power_gap = power - self.P
        return {
            "loss": -rate,
            "rate": rate,
            "reward": rate,
            "power": power,
            "rate_gap": rate_gap,
            "power_gap": power_gap,
            "rate_violation": torch.relu(rate_gap),
            "power_violation": torch.relu(power_gap),
        }


__all__ = ["UplinkPrecoderObjective"]
