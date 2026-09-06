# Downlink Monte Carlo

This is the offline train-and-test method. In payload mode, each training seed
creates one channel episode and the rollout generates later blocks while the
payload is drained. In streaming mode, each training seed is one independent
one-block sample. The rollout selects `n_kl` states; there are no expert labels
and no MSE beam loss.

Choose `per_user_nets` for one network per user beam, or `bs_shared_net` for
one network that emits the complete BS precoder and is split into user slices.
Both optimize normalized inverse-CNR weighted finite-blocklength sum rate.

For streaming with `bs_shared_net`,
`shared_bs_streaming_blocklength_input_mode` selects either one joint
blocklength-vector query or the older one-user-change-at-a-time comparison.
The joint vector is order independent and is the recommended setting.

The only neural model is an MLP. It predicts the complex precoder directly and
does not use RZF initialization, labels, or residuals.

Run:

```powershell
python -m latency_optimization run --link downlink --method monte_carlo --cfg_name downlink_dispersion_heavy.yaml --num_train_channels 11 --test_seed 3
```

For streaming, use block-count options instead:

```powershell
python -m latency_optimization run --link downlink --method monte_carlo --cfg_name downlink_streaming.yaml --num_train_blocks 11 --num_test_blocks 11 --test_seed 3
```

Training artifacts are under `training/data`; held-out evaluation results are
under `testing/data`. Payload samples are stored in `training/channels`, while
streaming samples are stored in `training/blocks`. Each directory has a
`manifest.json`, and every saved sample records a separate SNR value for every
user. Testing uses the corresponding `testing/channels` or `testing/blocks`
directory.

Payload testing uses `monte_carlo_num_test_channels`; streaming testing uses
`monte_carlo_num_test_blocks`. Both use `monte_carlo_test_snr_db_ranges`. Their
results are saved under `testing/samples`, and
`testing/data/test_dataset_summary.txt` reports the mean and standard deviation.

The streaming scenario's `experiment_scenario.number_of_blocks` is used by
multi-block convergence runs. Monte Carlo intentionally sets it to one for
each sample, so `monte_carlo_num_training_blocks` and
`monte_carlo_num_test_blocks` count blocks directly rather than producing a
product of samples and blocks.
