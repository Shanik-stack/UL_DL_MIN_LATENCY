import unittest

from latency_optimization.optimization.convergence_criteria import (
    KktResiduals,
    KktTolerances,
    convergence_status,
    objective_convergence_status,
)


class ConvergenceCriteriaTests(unittest.TestCase):
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

    def test_kkt_rule_rejects_stationary_infeasible_state(self) -> None:
        self.assertEqual(
            convergence_status(
                "kkt_residuals",
                precoder_change=1e-4,
                precoder_change_tolerance=1e-3,
                has_previous_state=True,
                residuals=KktResiduals(2e-3, 0.0, 1e-4),
                kkt_tolerances=KktTolerances(1e-3, 1e-6, 1e-3),
            ),
            "stationary_infeasible",
        )

    def test_kkt_rule_accepts_feasible_stationary_state(self) -> None:
        self.assertEqual(
            convergence_status(
                "kkt_residuals",
                precoder_change=1e-4,
                precoder_change_tolerance=1e-3,
                has_previous_state=True,
                residuals=KktResiduals(1e-4, 0.0, 1e-4),
                kkt_tolerances=KktTolerances(1e-3, 1e-6, 1e-3),
            ),
            "kkt_converged",
        )


if __name__ == "__main__":
    unittest.main()
