import unittest

from latency_optimization.downlink.config import load_config as load_downlink_config
from latency_optimization.downlink.monte_carlo.network_operations import (
    _build_training_user_models,
)
from latency_optimization.downlink.monte_carlo.rollout import (
    _collect_downlink_episode_rollout_queries,
)
from latency_optimization.uplink.config import load_config as load_uplink_config
from latency_optimization.uplink.monte_carlo.rollout import (
    _collect_uplink_payload_rollout_queries_for_episode,
)
from latency_optimization.uplink.precoders.models import build_user_precoder_net


class MonteCarloRolloutLimitTests(unittest.TestCase):
    def test_uplink_incomplete_payload_episode_is_retained(self) -> None:
        system, simulation, _ = load_uplink_config("tests/fixtures/uplink_streaming_smoke.yaml")
        simulation = {**simulation, "max_total_blocks": 1}
        models = [
            build_user_precoder_net(system["NR"][k], system["NT"][k], system["dk"][k])
            for k in range(system["K"])
        ]
        queries_by_user = _collect_uplink_payload_rollout_queries_for_episode(
            system,
            simulation,
            {
                "seed": 1,
                "scenario": {
                    "mode": "payload",
                    "payload_bits_per_user": [1_000_000 for _ in range(system["K"])],
                },
                "snr_db_by_user": [4.0 for _ in range(system["K"])],
            },
            models,
        )

        for queries in queries_by_user:
            self.assertTrue(queries)
            self.assertFalse(queries[-1]["episode_completed"])
            self.assertEqual("max_total_blocks_reached", queries[-1]["episode_stop_reason"])
            self.assertEqual(1, queries[-1]["episode_blocks_visited"])

    def test_downlink_zero_service_episode_is_retained(self) -> None:
        system, simulation, _ = load_downlink_config("tests/fixtures/downlink_streaming_smoke.yaml")
        simulation = {**simulation, "max_total_blocks": 1}
        models = _build_training_user_models(system, simulation)
        queries = _collect_downlink_episode_rollout_queries(
            system,
            simulation,
            {
                "seed": 1,
                "scenario": {
                    "mode": "payload",
                    "payload_bits_per_user": [1_000_000 for _ in range(system["K"])],
                },
                "snr_db_by_user": [4.0 for _ in range(system["K"])],
            },
            models,
        )

        self.assertTrue(queries)
        self.assertFalse(queries[-1]["episode_completed"])
        self.assertIn(
            queries[-1]["episode_stop_reason"],
            {"zero_service_frontier", "max_total_blocks_reached"},
        )
        self.assertEqual(1, queries[-1]["episode_blocks_visited"])


if __name__ == "__main__":
    unittest.main()
