import unittest

from latency_optimization.downlink.config import load_config as load_downlink_config
from latency_optimization.uplink.config import load_config as load_uplink_config


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


if __name__ == "__main__":
    unittest.main()
