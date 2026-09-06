import argparse
import tempfile
import unittest
from pathlib import Path

from latency_optimization.experiments.monte_carlo_testing import (
    build_test_dataset_summary,
    build_test_sample_dirs,
    build_test_search_overrides,
    build_test_search_tag,
)


class MonteCarloSupportTests(unittest.TestCase):
    def test_binary_search_override_and_tag(self) -> None:
        args = argparse.Namespace(
            test_n_search_strategy="binary",
            test_n_search_direction=None,
            test_n_search_coarse_step=None,
            test_n_search_exponential_factor=None,
        )
        overrides = build_test_search_overrides(args)
        self.assertEqual(overrides, {"monte_carlo_test_n_search_strategy": "binary"})
        self.assertEqual(build_test_search_tag(overrides), "ntest_bin")

    def test_links_use_the_same_episode_directory_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            roots = {"testing_root": temp_dir}
            uplink = build_test_sample_dirs(roots, 3, link="uplink")
            downlink = build_test_sample_dirs(roots, 4, link="downlink")
            self.assertIn("optimization_history", uplink)
            self.assertIn("optimization_history", downlink)
            self.assertTrue(Path(uplink["test_data"]).is_dir())
            self.assertTrue(Path(downlink["test_data"]).is_dir())

    def test_test_dataset_summary_aggregates_samples(self) -> None:
        results = [
            {
                "seed": 3,
                "summary_metrics": {
                    "final_total_latency": 0.2,
                    "total_latency_reduction_percent": 10.0,
                    "final_asynchronality_sum": 0.01,
                },
            },
            {
                "seed": 4,
                "summary_metrics": {
                    "final_total_latency": 0.4,
                    "total_latency_reduction_percent": 20.0,
                    "final_asynchronality_sum": 0.03,
                },
            },
        ]
        summary = build_test_dataset_summary(results, {3: [1.0], 4: [2.0]})
        self.assertEqual(summary["total_test_samples"], 2)
        self.assertAlmostEqual(summary["final_total_latency_seconds"]["mean"], 0.3)


if __name__ == "__main__":
    unittest.main()
