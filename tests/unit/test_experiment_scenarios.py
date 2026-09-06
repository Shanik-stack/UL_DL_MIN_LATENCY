import unittest

from latency_optimization.core.scenarios import (
    PAYLOAD_MODE,
    STREAMING_MODE,
    build_experiment_scenario,
    validate_experiment_scenario_config,
)


class ExperimentScenarioTests(unittest.TestCase):
    def setUp(self) -> None:
        self.system_params = {"K": 3, "B": [10, 20, 30]}

    def test_payload_uses_one_total_payload_per_user(self) -> None:
        scenario = build_experiment_scenario(
            self.system_params,
            {
                "max_total_blocks": 20,
                "experiment_scenario": {
                    "mode": PAYLOAD_MODE,
                    "payload_bits_source": "system_B",
                },
            },
            seed=7,
        )

        self.assertEqual(scenario["mode"], PAYLOAD_MODE)
        self.assertEqual(scenario["payload_bits_per_user"], [10, 20, 30])
        self.assertEqual(scenario["per_user_total_target_bits"], [10, 20, 30])
        self.assertEqual(scenario["termination_rule"], "until_payload_drained")

    def test_streaming_repeats_system_B_for_every_block_without_carryover(self) -> None:
        scenario = build_experiment_scenario(
            self.system_params,
            {
                "max_total_blocks": 20,
                "experiment_scenario": {
                    "mode": STREAMING_MODE,
                    "number_of_blocks": 3,
                },
            },
            seed=7,
        )

        self.assertEqual(scenario["mode"], STREAMING_MODE)
        self.assertEqual(scenario["bits_per_block_per_user"], [10, 20, 30])
        self.assertEqual(
            scenario["streaming_bit_targets_by_block"],
            [[10, 10, 10], [20, 20, 20], [30, 30, 30]],
        )
        self.assertEqual(scenario["per_user_total_target_bits"], [30, 60, 90])
        self.assertEqual(scenario["total_target_bits"], 180)
        self.assertFalse(scenario["carry_unserved_bits_to_next_block"])
        self.assertEqual(scenario["termination_rule"], "after_number_of_blocks")

    def test_streaming_requires_an_explicit_positive_block_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires.*number_of_blocks"):
            validate_experiment_scenario_config(
                {"mode": STREAMING_MODE},
                system_params=self.system_params,
                max_total_blocks=20,
            )
        with self.assertRaisesRegex(ValueError, "must be positive"):
            validate_experiment_scenario_config(
                {"mode": STREAMING_MODE, "number_of_blocks": 0},
                system_params=self.system_params,
                max_total_blocks=20,
            )

    def test_streaming_horizon_cannot_exceed_the_block_safety_cap(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            validate_experiment_scenario_config(
                {"mode": STREAMING_MODE, "number_of_blocks": 21},
                system_params=self.system_params,
                max_total_blocks=20,
            )

    def test_streaming_requires_a_positive_target_for_every_user(self) -> None:
        with self.assertRaisesRegex(ValueError, "every system B"):
            validate_experiment_scenario_config(
                {"mode": STREAMING_MODE, "number_of_blocks": 3},
                system_params={"K": 3, "B": [10, 0, 30]},
                max_total_blocks=20,
            )

    def test_unknown_scenario_name_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be exactly one of"):
            validate_experiment_scenario_config(
                {"mode": "fixed_block_targets"},
                system_params=self.system_params,
                max_total_blocks=20,
            )

    def test_payload_completion_alias_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be exactly one of"):
            validate_experiment_scenario_config(
                {"mode": "payload_completion"},
                system_params=self.system_params,
                max_total_blocks=20,
            )

    def test_unused_scenario_options_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported experiment_scenario option"):
            validate_experiment_scenario_config(
                {"mode": STREAMING_MODE, "number_of_blocks": 3, "skip_infeasible_blocks": True},
                system_params=self.system_params,
                max_total_blocks=20,
            )


if __name__ == "__main__":
    unittest.main()
