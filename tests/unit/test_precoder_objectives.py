import unittest

import torch

from latency_optimization.downlink.objectives.precoder import DownlinkBlockContext, DownlinkPrecoderObjective
from latency_optimization.runtime import DEVICE
from latency_optimization.uplink.objectives.precoder import UplinkPrecoderObjective


class PrecodersObjectiveTests(unittest.TestCase):
    def test_uplink_objective_returns_named_differentiable_values(self) -> None:
        channel = torch.ones((1, 2), dtype=torch.complex64, device=DEVICE)
        precoder = torch.ones(
            (2, 1), dtype=torch.complex64, device=DEVICE, requires_grad=True
        )
        objective = UplinkPrecoderObjective(
            channel=channel,
            noise_variance=1.0,
            epsilon=1.0e-5,
            payload_bits=1.0,
            power_limit=3.0,
            blocklength=20,
        ).to(DEVICE)

        result = objective(precoder)

        self.assertEqual(
            set(result),
            {
                "loss",
                "rate",
                "reward",
                "power",
                "rate_gap",
                "power_gap",
                "rate_violation",
                "power_violation",
            },
        )
        result["loss"].backward()
        self.assertIsNotNone(precoder.grad)

    def test_downlink_objective_uses_joint_bs_power(self) -> None:
        precoders = {
            0: torch.ones((2, 1), dtype=torch.complex64, device=DEVICE, requires_grad=True),
            1: torch.full(
                (2, 1), 0.5, dtype=torch.complex64, device=DEVICE, requires_grad=True
            ),
        }
        objective = DownlinkPrecoderObjective(
            context=DownlinkBlockContext(
                channels={
                    0: torch.ones((1, 2), dtype=torch.complex64, device=DEVICE),
                    1: torch.tensor([[1.0, -1.0]], dtype=torch.complex64, device=DEVICE),
                },
                noise_variances={0: 1.0, 1: 1.0},
                error_probabilities={0: 1.0e-5, 1: 1.0e-5},
                block_power_budget=3.0,
            ),
            blocklengths={0: 20, 1: 20},
            active_users=[0, 1],
            requested_bits={0: 1, 1: 1},
            user_weights={0: 1.0, 1: 1.0},
        ).to(DEVICE)

        result = objective(precoders)

        self.assertAlmostEqual(float(result["block_power"].detach().cpu()), 2.5, places=5)
        result["loss"].backward()
        self.assertIsNotNone(precoders[0].grad)
        self.assertIsNotNone(precoders[1].grad)


if __name__ == "__main__":
    unittest.main()
