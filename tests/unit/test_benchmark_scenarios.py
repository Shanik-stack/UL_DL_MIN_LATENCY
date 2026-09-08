import unittest
from pathlib import Path
from unittest.mock import patch

from latency_optimization.downlink.benchmarks.evaluate_test_dataset import _evaluate_episode
from latency_optimization.downlink.configuration.loader import load_config as load_downlink_config
from latency_optimization.uplink.benchmarks.linear_beamforming import run_uplink_closed_form_benchmark


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


class BenchmarkScenarioTests(unittest.TestCase):
    def test_downlink_zf_supports_streaming_without_payload_carryover(self) -> None:
        system_params, sim_params, _ = load_downlink_config(
            str(FIXTURES / "downlink_streaming_smoke.yaml")
        )
        result = _evaluate_episode(system_params, sim_params, 9, [4.0, 4.0], "zf")

        self.assertEqual(result["scenario_mode"], "streaming")
        self.assertEqual([len(row) for row in result["n_kl_per_user"]], [1, 1])

    def test_uplink_rzf_supports_streaming(self) -> None:
        with patch(
            "latency_optimization.uplink.benchmarks.linear_beamforming.build_uplink_convergence_result_dirs",
            return_value={},
        ):
            experiment = run_uplink_closed_form_benchmark(
                method_key="rzf",
                cfg_name=str(FIXTURES / "uplink_streaming_smoke.yaml"),
                seed=9,
                verbose=False,
            )

        self.assertEqual(experiment["result"]["scenario_mode"], "streaming")
        self.assertEqual(experiment["benchmark_data"]["scenario_mode"], "streaming")


if __name__ == "__main__":
    unittest.main()
