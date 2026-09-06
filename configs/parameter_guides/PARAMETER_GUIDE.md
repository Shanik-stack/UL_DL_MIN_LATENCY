# Parameter Guide

The canonical YAML files are experiment definitions. They expose only choices
that change the retained methods. Unknown `simulation` keys are rejected.

All choice values are exact and case-sensitive. Use `payload` or `streaming`
for the scenario, `ascending` or `descending` for search direction, and the
documented full names for every method/model option. Removed spellings are not
converted or accepted. Per-user system arrays must contain exactly `K` values;
the loader does not broadcast a scalar across users.

## Canonical regime

The supplied configs use payload completion or streaming at 4 dB reference
SNR, `epsilon = 1e-14`, and short per-user maximum blocklengths. Monte Carlo train
and test episodes span 0-8 dB per user, so the learned model is evaluated both
below and above the reference channel quality without changing the FBL regime.

## System parameters

- `K`: number of users.
- `T`: maximum symbols available to each user in one block.
- `B`: bits assigned to each user. In `payload`, this is the total
  payload. In `streaming`, this is the fresh target requested in every block.
- `P`: transmit-power budget. In downlink it is one shared BS power budget, so
  every entry must be the same value.
- `snr_db`: reference SNR used to set the noise level.
- `fs`: symbol rate used to convert used symbols to seconds.
- `epsilon`: target packet error probability used by the FBL rate formula.
- `Nt`, `Nr`: Uplink transmit and BS receive antennas, respectively.
- `Nb`, `Nr`: Downlink BS transmit and user receive antennas, respectively.
- `initial_bits_per_symbol`: Downlink-only random-baseline setting. It does not
  restrict the optimized method.

## Convergence method

- `convergence_precoder_update_mode`
  - `direct_precoder`: optimize the complex block precoder itself.
  - `precoder_net`: optimize the neural precoder parameterization online.
- `max_epochs`: maximum updates for one precoder solve.
- `lr_net`: Uplink online precoder learning rate.
- `user_update_lr`: Downlink online precoder learning rate.
- `convergence_stopping_rule`
  - `objective_stationarity`: stop once the relative precoder change is below
    `precoder_change_tolerance`. This is the compact choice for payload runs.
  - `kkt_residuals`: stop only after the precoder change is small and the
    primal/complementarity diagnostics meet their configured limits. This is
    the streaming choice, where an unchanged but infeasible beam must not be
    accepted.
- `precoder_change_tolerance`: relative change between successive precoders.
- `kkt_primal_tolerance`: maximum permitted rate or power violation for the
  residual-based stopping rule.
- `kkt_complementarity_tolerance`: complementarity diagnostic threshold. It is
  zero in the retained objective-only solvers because they have no dual update.
- `kkt_stationarity_tolerance`: maximum relative precoder change accepted by
  the residual-based stopping rule.
- `print_every_epoch`: terminal logging interval.

## Downlink network scope

`downlink_precoder_net_scope` matters only with `precoder_net`:

- `per_user_nets`: one precoder network per user.
- `bs_shared_net`: one network produces the full BS precoder in a joint forward
  pass, then the output is split into user precoder slices.

`shared_bs_streaming_blocklength_input_mode` applies only when both streaming
and `bs_shared_net` are selected:

- `joint_blocklength_vector`: query the shared network with every user's
  candidate `n_kl` in one vector. This is the preferred order-independent mode.
- `one_user_change_at_a_time`: change one user's `n_kl` per query while the
  other entries remain fixed. This preserves the older sequential comparison.

The retained Downlink objective is normalized inverse-CNR weighted FBL sum
rate. For active users `A`, it is `sum_{k in A} w_k R_k`, where
`w_k = (1/CNR_k) / mean_{j in A}(1/CNR_j)`. This gives weaker channels more
weight while keeping the average weight equal to one.

## Monte Carlo method

- `monte_carlo_num_training_channels`: payload-only number of independent
  training channel episodes.
- `monte_carlo_num_training_blocks`: streaming-only number of independent
  one-block training samples.
  Seeds are generated in order from `1`, skipping the held-out test seed.
- `monte_carlo_train_seeds`: optional explicit comma-separated seed list; it
  replaces the generated range.
- `monte_carlo_training_snr_db_ranges`: one inclusive `[minimum_dB, maximum_dB]`
  range per user, in user order. Each user is evenly stratified across its own
  range. The schedules are cyclically offset, so users do not always receive
  the same relative SNR in an episode.
- `monte_carlo_num_test_channels`: payload-only number of held-out channel
  episodes. Their seeds are disjoint from the training seed set.
- `monte_carlo_num_test_blocks`: streaming-only number of held-out one-block
  samples. Their seeds are disjoint from the training seed set.
- `monte_carlo_test_snr_db_ranges`: one `[minimum_dB, maximum_dB]` range per
  user for held-out evaluation. Test SNR values use midpoint stratification,
  so they fall between the training grid points.
- `monte_carlo_training_max_epochs`: offline neural-network update budget.

Payload uses channel episodes as its base dataset; streaming uses independent
one-block samples. Payload later blocks are generated by rollout. In both cases
the current model visits blocklength states through its own rollout and is not
supervised with expert beams or MSE targets.

## Scenario and blocklength search

- `experiment_scenario.mode`: `payload` or `streaming`.
- `experiment_scenario.number_of_blocks`: required only for `streaming`; a
  convergence run processes exactly this many blocks. A Monte Carlo streaming
  sample is intentionally reduced to one block, so its block count comes from
  `monte_carlo_num_training_blocks` or `monte_carlo_num_test_blocks` instead.
- `payload_bits_source`: payload-completion-only source for the total payload.
  `system_B` uses `test.B`; `explicit` uses `payload_bits_values`.

Streaming behavior is fixed rather than configurable: a zero-service block is
recorded with `n_kl = T[k]`, contributes full-block time, and does not carry its
missed bits into the next block.

- `max_total_blocks`: safety cap on created blocks. A streaming horizon cannot
  exceed this value.
- `uplink_rate_model`: Uplink-only `snr` or `sinr` rate evaluation.
- `n_kl_range.min`: smallest allowed blocklength.
- `n_kl_range.step`: fine search increment.
- `n_search_direction`: `descending` starts at `T_k`; `ascending` starts at
  the minimum blocklength.
- `n_search_strategy`: `fixed_step`, `coarse_to_fine`, `exponential`, or
  `binary`.
- `n_search_coarse_step`: first-stage increment for `coarse_to_fine`.
- `n_search_exponential_factor`: spacing growth factor for `exponential`.

Monte Carlo training always visits `n_kl` with `fixed_step` so the rollout
states are well defined. Its test search can use the selected strategy.
