import unittest

import torch

from latency_optimization.downlink.precoders.models import (
    build_shared_bs_precoder_net,
    build_shared_bs_precoder_net_with_blocklength,
    build_user_precoder_net,
    build_user_precoder_net_with_blocklength,
)


class DownlinkPrecoderModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.channels = [
            torch.ones((2, 3), dtype=torch.complex64),
            torch.ones((1, 3), dtype=torch.complex64),
        ]

    def channels_for(self, model):
        device = next(model.parameters()).device
        return [channel.to(device) for channel in self.channels]

    def test_per_user_channel_model_output_shape(self) -> None:
        model = build_user_precoder_net(2, 3, 2)
        self.assertEqual(tuple(model(self.channels_for(model)[0]).shape), (1, 12))

    def test_per_user_blocklength_model_output_shape(self) -> None:
        model = build_user_precoder_net_with_blocklength(
            2, 3, 2, k_count=2, max_nr=2, max_nb=3
        )
        output = model(
            self.channels_for(model),
            20,
            [1, 1],
            torch.eye(2, dtype=torch.complex64),
            1.0e-5,
        )
        self.assertEqual(tuple(output.shape), (1, 12))

    def test_shared_channel_model_output_shape(self) -> None:
        model = build_shared_bs_precoder_net(k_count=2, max_nr=2, max_nb=3, max_dk=2)
        self.assertEqual(tuple(model(self.channels_for(model), [1, 1]).shape), (1, 24))

    def test_shared_blocklength_model_output_shape(self) -> None:
        model = build_shared_bs_precoder_net_with_blocklength(
            k_count=2, max_nr=2, max_nb=3, max_dk=2
        )
        output = model(self.channels_for(model), [20, 30], [1, 1], [0.1, 0.2], [1.0e-5, 1.0e-5])
        self.assertEqual(tuple(output.shape), (1, 24))


if __name__ == "__main__":
    unittest.main()
