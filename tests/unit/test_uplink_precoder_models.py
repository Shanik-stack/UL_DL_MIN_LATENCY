import unittest

import torch

from latency_optimization.uplink.precoders.checkpoints import (
    export_user_model_specs,
    export_user_model_states,
    load_user_precoder_models,
)
from latency_optimization.uplink.precoders.inference import infer_precoder
from latency_optimization.uplink.precoders.models import build_user_precoder_net


class UplinkPrecoderModelTests(unittest.TestCase):
    def test_tensor_inference_shape_power_and_checkpoint_roundtrip(self) -> None:
        model = build_user_precoder_net(2, 3, 1)
        device = next(model.parameters()).device
        channel = torch.ones((2, 3), dtype=torch.complex64, device=device)

        precoder = infer_precoder(model, channel, 20, 0.2, 1.0e-5, 3, 1, 4.0)

        self.assertEqual(tuple(precoder.shape), (3, 1))
        self.assertEqual(precoder.dtype, torch.complex64)
        power = torch.linalg.norm(precoder.detach()).square().real.cpu()
        self.assertLessEqual(float(power), 4.0)

        specs = export_user_model_specs([2], [3], [1])
        restored = load_user_precoder_models(specs, export_user_model_states([model]))
        restored_precoder = infer_precoder(
            restored[0], channel, 20, 0.2, 1.0e-5, 3, 1, 4.0
        )
        self.assertTrue(torch.allclose(precoder, restored_precoder))


if __name__ == "__main__":
    unittest.main()
