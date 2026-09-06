# Downlink Monte Carlo Configuration

Use `downlink_dispersion_heavy.yaml` for payload completion or
`downlink_streaming.yaml` for fresh per-block bit targets.

For `payload`, set `monte_carlo_num_training_channels`: each seed is one
independent payload channel episode. The payload rollout creates and optimizes
later blocks online until the payload is drained.

For `streaming`, set `monte_carlo_num_training_blocks`: each seed is one
independent one-block training sample. Streaming has no carried payload, so
each block is trained directly. Use the corresponding
`monte_carlo_num_test_channels` or `monte_carlo_num_test_blocks` for testing.
The streaming count is direct: it is not multiplied by
`experiment_scenario.number_of_blocks`.
Choose `per_user_nets` or `bs_shared_net`, set the user SNR ranges, and set
`monte_carlo_training_max_epochs`. No hand-built blocklength grid or beam-label
dataset is used.

## Precoders

The only supported neural model is an MLP. It predicts the complex precoder
directly and does not use RZF inputs, labels, residuals, or initialization.

With `per_user_nets`, each user has an MLP that emits its own precoder. With
`bs_shared_net`, one MLP sees the joint block context, emits the full BS
precoder, and the output is split into user slices.
