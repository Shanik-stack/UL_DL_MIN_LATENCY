import math
import unittest

from latency_optimization.results.metrics import (
    build_schedule_reference,
    pairwise_latency_differences,
    reference_latency_metrics,
)


class ResultMetricsTests(unittest.TestCase):
    def test_schedule_reference_uses_one_common_schema(self) -> None:
        reference = build_schedule_reference(
            "random",
            [0.1],
            {
                "completed": False,
                "failure_reason": "incomplete",
                "remaining_bits": [2],
                "n_kl": [[5]],
                "B_kl": [[1]],
                "R_alloc": [[0.2]],
                "skipped_blocks_per_user": [1],
            },
            snr_db=[3.0],
            sinr_db=[2.0],
        )
        self.assertFalse(reference["completed"])
        self.assertEqual(reference["n_kl"], [[5]])
        self.assertEqual(reference["remaining_bits"], [2])

    def test_pairwise_asynchronality_uses_each_pair_once(self) -> None:
        matrix, pairs, total = pairwise_latency_differences([1.0, 3.0, 4.0])
        self.assertEqual(matrix, [[0.0, 2.0, 3.0], [2.0, 0.0, 1.0], [3.0, 1.0, 0.0]])
        self.assertEqual(len(pairs), 3)
        self.assertEqual(total, 6.0)

    def test_incomplete_reference_has_no_comparison_percentage(self) -> None:
        metrics = reference_latency_metrics([math.nan, 1.0], [0.5, 0.5])
        self.assertFalse(metrics["baseline_completed"])
        self.assertTrue(math.isnan(metrics["total_latency_reduction_percent"]))

    def test_completed_reference_reports_latency_reduction(self) -> None:
        metrics = reference_latency_metrics([2.0, 2.0], [1.0, 1.0])
        self.assertTrue(metrics["baseline_completed"])
        self.assertEqual(metrics["total_latency_reduction_percent"], 50.0)


if __name__ == "__main__":
    unittest.main()
