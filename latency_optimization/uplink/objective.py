"""Uplink finite-blocklength objective shared by all optimization methods."""

from __future__ import annotations

import torch
from torch import nn

from latency_optimization.physics.rate_law import NORMAL_APPROXIMATION_RATE_LAW, RateLaw

from .uplink_rate_model import evaluate_uplink_rate_tensor


class UplinkPrecoderObjective(nn.Module):
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
        self.n_kl = int(blocklength)

    def set_payload(self, bits: float) -> None:
        self.B = float(bits)

    def finite_blocklength_rate(self, precoder: torch.Tensor) -> torch.Tensor:
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
