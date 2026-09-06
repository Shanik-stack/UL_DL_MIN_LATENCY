import unittest

from latency_optimization.optimization.stopping import objective_convergence_status


class OptimizationStoppingTests(unittest.TestCase):
    def test_stops_after_a_previous_state_when_change_is_small(self) -> None:
        self.assertEqual(
            objective_convergence_status(3e-4, 1e-3, has_previous_state=True),
            "objective_stationary",
        )

    def test_first_state_never_stops(self) -> None:
        self.assertEqual(
            objective_convergence_status(0.0, 1e-3, has_previous_state=False),
            "running",
        )

    def test_change_above_tolerance_continues(self) -> None:
        self.assertEqual(
            objective_convergence_status(2e-3, 1e-3, has_previous_state=True),
            "running",
        )

    def test_negative_tolerance_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            objective_convergence_status(0.0, -1.0, has_previous_state=True)


if __name__ == "__main__":
    unittest.main()
