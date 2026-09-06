import unittest

import torch

from latency_optimization.precoders.model_state import (
    clone_model_state,
    clone_model_states,
    relative_model_state_change,
    relative_models_state_change,
    restore_model_states,
)


class ModelStateTests(unittest.TestCase):
    def test_change_is_zero_for_unchanged_model(self) -> None:
        model = torch.nn.Linear(2, 1)
        state = clone_model_state(model)
        self.assertEqual(relative_model_state_change(model, state), 0.0)

    def test_restore_recovers_multiple_models(self) -> None:
        models = [torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)]
        states = clone_model_states(models)
        with torch.no_grad():
            for model in models:
                model.weight.add_(1.0)
        self.assertGreater(relative_models_state_change(models, states), 0.0)
        restore_model_states(models, states)
        self.assertEqual(relative_models_state_change(models, states), 0.0)


if __name__ == "__main__":
    unittest.main()
