import unittest

from latency_optimization.optimization.blocklength_search import build_monte_carlo_n_search_config


class MonteCarloBlocklengthSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.simulation = {
            "n_kl_step": 2,
            "n_search_direction": "descending",
            "n_search_strategy": "fixed_step",
            "n_search_coarse_step": 6,
            "n_search_exponential_factor": 3,
            "monte_carlo_test_n_search_direction": "ascending",
            "monte_carlo_test_n_search_strategy": "binary",
            "monte_carlo_test_n_search_coarse_step": 8,
            "monte_carlo_test_n_search_exponential_factor": 4,
        }

    def test_training_uses_training_search_fields(self) -> None:
        search = build_monte_carlo_n_search_config(
            self.simulation,
            n_min=1,
            n_max=20,
            phase="training",
        )
        self.assertEqual(search["direction"], "descending")
        self.assertEqual(search["strategy"], "fixed_step")
        self.assertEqual(search["coarse_step"], 6)

    def test_testing_uses_test_search_overrides(self) -> None:
        search = build_monte_carlo_n_search_config(
            self.simulation,
            n_min=1,
            n_max=20,
            phase="testing",
        )
        self.assertEqual(search["direction"], "ascending")
        self.assertEqual(search["strategy"], "binary")
        self.assertEqual(search["exponential_factor"], 4)

    def test_training_rejects_non_fixed_search(self) -> None:
        self.simulation["n_search_strategy"] = "binary"
        with self.assertRaises(ValueError):
            build_monte_carlo_n_search_config(
                self.simulation,
                n_min=1,
                n_max=20,
                phase="training",
            )


if __name__ == "__main__":
    unittest.main()
