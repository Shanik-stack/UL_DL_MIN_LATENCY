"""Coupled downlink finite-blocklength precoder objective."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from latency_optimization.physics.rate_law import NORMAL_APPROXIMATION_RATE_LAW, RateLaw

from ..physics.rate import evaluate_downlink_rate_tensor


@dataclass(frozen=True)
class DownlinkBlockContext:
    """Immutable Torch inputs shared by every objective evaluation of one block."""

    channels: Mapping[int, torch.Tensor]
    noise_variances: Mapping[int, float]
    error_probabilities: Mapping[int, float]
    block_power_budget: float
    rate_law: RateLaw = NORMAL_APPROXIMATION_RATE_LAW


class DownlinkPrecoderObjective(nn.Module):
    """Evaluate one joint BS precoder for an active-user block."""

    def __init__(
        self,
        context: DownlinkBlockContext,
        blocklengths: Mapping[int, int],
        active_users: Sequence[int],
        requested_bits: Mapping[int, int],
        user_weights: Mapping[int, float],
    ) -> None:
        """Bind immutable block physics to candidate n, bits, users, and weights."""
        super().__init__()
        self.context = context
        self.active_users = [int(user) for user in active_users]
        self.blocklengths = {int(user): int(value) for user, value in blocklengths.items()}
        self.requested_bits = {int(user): int(bits) for user, bits in requested_bits.items()}
        self.user_weights = {int(user): float(weight) for user, weight in user_weights.items()}

    def blocklength_for(self, user: int) -> int:
        """Return the candidate n_kl assigned to an active user."""
        return int(self.blocklengths[int(user)])

    def finite_blocklength_rate(
        self,
        user: int,
        precoders: Mapping[int, torch.Tensor],
    ) -> torch.Tensor:
        """Compute one active user's differentiable interference-coupled FBL rate.

        What: form ``Sigma_k`` from all other candidate beams and evaluate the common
        rate law at that user's ``n_kl``. Why: gradients must include how changing any
        BS beam changes desired signal and interference within the same joint block.
        """
        user = int(user)
        channel = self.context.channels[user]
        covariance = self.context.noise_variances[user] * torch.eye(
            channel.shape[0],
            dtype=channel.dtype,
            device=channel.device,
        )
        for interferer in self.active_users:
            if interferer == user:
                continue
            interference = channel @ precoders[interferer]
            covariance = covariance + interference @ interference.conj().transpose(1, 0)
        return evaluate_downlink_rate_tensor(
            channel,
            precoders[user],
            covariance,
            self.blocklength_for(user),
            self.context.error_probabilities[user],
            rate_law=self.context.rate_law,
        )

    def forward(self, precoders: Mapping[int, torch.Tensor]) -> dict[str, object]:
        """Evaluate the complete differentiable objective for one downlink block.

        What: compute per-user rates, required rates ``B_k/n_k``, ReLU rate violations,
        per-user powers, total BS power, and weighted sum rate; return ``-weighted_rate``
        as the loss. Why: direct-beam and neural convergence modes need one identical
        objective and one identical set of feasibility diagnostics.
        """
        rates: dict[int, torch.Tensor] = {}
        powers: dict[int, torch.Tensor] = {}
        required_rates: dict[int, float] = {}
        rate_gaps: dict[int, torch.Tensor] = {}
        rate_violations: dict[int, torch.Tensor] = {}

        for user in self.active_users:
            rate = self.finite_blocklength_rate(user, precoders)
            power = torch.linalg.norm(precoders[user], ord="fro").square().real
            required_rate = float(self.requested_bits.get(user, 0)) / max(
                self.blocklength_for(user), 1
            )
            gap = rate.new_tensor(required_rate) - rate
            rates[user] = rate
            powers[user] = power
            required_rates[user] = required_rate
            rate_gaps[user] = gap
            rate_violations[user] = torch.relu(gap)

        reference = next(iter(precoders.values()), None)
        zero = torch.zeros(()) if reference is None else reference.real.new_zeros(())
        total_rate = torch.stack(list(rates.values())).sum() if rates else zero
        weighted_rate = (
            torch.stack(
                [self.user_weights.get(user, 1.0) * rates[user] for user in self.active_users]
            ).sum()
            if rates
            else zero
        )
        block_power = torch.stack(list(powers.values())).sum() if powers else zero
        block_power_gap = block_power - self.context.block_power_budget
        return {
            "loss": -weighted_rate,
            "rates": rates,
            "powers": powers,
            "required_rates": required_rates,
            "rate_gap": rate_gaps,
            "rate_violation_pos": rate_violations,
            "block_power": block_power,
            "block_power_gap": block_power_gap,
            "block_power_violation_pos": torch.relu(block_power_gap),
            "sum_rate": total_rate,
            "weighted_sum_rate": weighted_rate,
        }


__all__ = ["DownlinkBlockContext", "DownlinkPrecoderObjective"]
