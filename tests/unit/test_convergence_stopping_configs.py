import unittest

import torch

from latency_optimization.downlink.configuration.loader import load_config as load_downlink_config
from latency_optimization.precoders.parameters import complex_parameter
from latency_optimization.uplink.methods.convergence.optimize_precoder import optimize_precoder_for_nl
from latency_optimization.uplink.configuration.loader import load_config as load_uplink_config


class _QuadraticPrecoderObjective(torch.nn.Module):
    P = 100.0

    def forward(self, precoder: torch.Tensor) -> dict[str, torch.Tensor]:
        loss = torch.sum(torch.abs(precoder - (1.0 + 0.0j)) ** 2)
        zero = loss.new_zeros(())
        return {
            "loss": loss,
            "rate": -loss,
            "reward": -loss,
            "power": torch.sum(torch.abs(precoder) ** 2),
            "rate_gap": zero,
            "power_gap": zero,
            "rate_violation": zero,
            "power_violation": zero,
        }


class ConvergenceStoppingConfigTests(unittest.TestCase):
    def test_payload_configs_use_objective_stationarity(self) -> None:
        _, uplink, _ = load_uplink_config("uplink_dispersion_heavy.yaml")
        _, downlink, _ = load_downlink_config("downlink_dispersion_heavy.yaml")
        self.assertEqual(uplink["convergence_stopping_rule"], "objective_stationarity")
        self.assertEqual(downlink["convergence_stopping_rule"], "objective_stationarity")

    def test_streaming_configs_keep_kkt_residual_stopping(self) -> None:
        _, uplink, _ = load_uplink_config("uplink_streaming.yaml")
        _, downlink, _ = load_downlink_config("downlink_streaming.yaml")
        for config in (uplink, downlink):
            self.assertEqual(config["convergence_stopping_rule"], "kkt_residuals")
            self.assertGreaterEqual(config["kkt_primal_tolerance"], 0.0)
            self.assertGreaterEqual(config["kkt_complementarity_tolerance"], 0.0)
            self.assertGreaterEqual(config["kkt_stationarity_tolerance"], 0.0)

    def test_monte_carlo_test_search_settings_survive_config_loading(self) -> None:
        _, uplink, _ = load_uplink_config("uplink_dispersion_heavy.yaml")
        _, downlink, _ = load_downlink_config("downlink_dispersion_heavy.yaml")
        for config in (uplink, downlink):
            self.assertIn("monte_carlo_test_n_search_direction", config)
            self.assertIn("monte_carlo_test_n_search_strategy", config)
            self.assertIn("monte_carlo_test_n_search_coarse_step", config)
            self.assertIn("monte_carlo_test_n_search_exponential_factor", config)

    def test_one_uplink_epoch_performs_one_optimizer_update(self) -> None:
        parameter = complex_parameter(torch.zeros((2, 1), dtype=torch.complex64))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        result = optimize_precoder_for_nl(
            precoder_net=None,
            loss_fn=_QuadraticPrecoderObjective(),
            Nt=2,
            dk=1,
            max_epochs=1,
            optimizer=optimizer,
            stopping_config={
                "convergence_stopping_rule": "objective_stationarity",
                "precoder_change_tolerance": 0.0,
                "kkt_primal_tolerance": 0.0,
                "kkt_complementarity_tolerance": 0.0,
                "kkt_stationarity_tolerance": 0.0,
            },
            verbose=False,
            precoder_param=parameter,
            update_mode="direct_precoder",
        )
        self.assertGreater(float(torch.linalg.norm(result["F"]).item()), 0.0)
        self.assertEqual(len(result["loss_curve"]), 1)


if __name__ == "__main__":
    unittest.main()
