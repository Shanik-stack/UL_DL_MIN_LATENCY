# Optimization Flow

This page is the navigation map for the executable algorithms. Every method has
one `experiment.py` entry point. Read that file first, then follow the method's
explicitly named optimization stages.

## Common Entry Path

```text
python -m latency_optimization
  -> latency_optimization/__main__.py
  -> <link>/methods/<method>/experiment.py
```

The CLI layer only selects a method and forwards arguments. An `experiment.py`
module loads configuration and channels, invokes the algorithm, calculates
metrics, and persists results. Mathematical optimization begins in the next
file in each call chain.

## Downlink Convergence

```text
methods/convergence/experiment.py::run_convergence_experiment
  -> optimize_transmission.py::optimize_downlink_transmission
     -> optimize_payload.py::optimize_payload_transmission
        or optimize_streaming.py::optimize_streaming_transmission
        -> optimize_precoder.py::optimize_precoders_for_block
        -> optimize_precoder.py::_reduce_blocklengths_with_reoptimization
           -> downlink/objectives/precoder.py::DownlinkPrecoderObjective
           -> downlink/simulation/system.py
```

`optimize_transmission.py` owns channel-block creation, active-user selection,
full-blocklength optimization, blocklength reduction, and service commitment.
`optimize_precoder.py` owns gradient updates for one coupled BS block and
enforces the joint BS power constraint.

## Uplink Convergence

```text
methods/convergence/experiment.py::run_convergence_experiment
  -> optimize_transmission.py::run_convergence_baseline
     -> optimize_payload.py::optimize_payload_with_precoder_training
        or optimize_streaming.py::optimize_streaming_blocks_with_precoder_training
        -> optimize_user_transmission.py::optimize_user_blocklength_and_precoder
           -> optimize_precoder.py::optimize_precoder_for_nl
              -> uplink/objectives/precoder.py::UplinkPrecoderObjective
              -> uplink/physics/rate.py
```

`optimize_transmission.py` owns each user's block and blocklength decisions.
`optimize_precoder.py` owns the fixed-`n_kl` gradient solve for one user.

## Downlink Monte Carlo

```text
methods/monte_carlo/experiment.py::main
  -> build_training_rollouts.py::build_training_dataset
  -> train_precoder_network.py::train_blocklength_aware_precoder_net
     -> build_training_rollouts.py::_generate_rollout_queries_for_downlink
     -> precoder_network.py::_scenario_forward_pass
     -> downlink/objectives/precoder.py::DownlinkPrecoderObjective
  -> evaluate_precoder_network.py::evaluate_downlink_precoder_net
```

Training updates model weights. Evaluation fixes those weights and performs
inference, blocklength search, and schedule commitment on held-out channels.

## Uplink Monte Carlo

```text
methods/monte_carlo/experiment.py::main
  -> build_training_rollouts.py::build_training_dataset
  -> train_precoder_network.py::train_blocklength_aware_precoder_net
     -> build_training_rollouts.py::_generate_rollout_queries_for_training_episodes
     -> precoder_network.py::_compute_r_fbl_torch
     -> uplink/objectives/precoder.py::UplinkPrecoderObjective
  -> evaluate_precoder_network.py::evaluate_blocklength_precoder_net
```

Each uplink user network is trained independently. Testing uses the same fixed
networks on held-out channels and does not backpropagate.

## Shared Mathematical Components

- `optimization/blocklength_search.py`: candidate `n_kl` generation and search.
- `optimization/convergence_criteria.py`: objective and KKT convergence checks.
- `physics/rate_law.py`: configured finite-blocklength rate-law selection.
- `physics/finite_blocklength.py`: differentiable finite-blocklength equation.
- `<link>/physics/rate.py`: link-specific covariance-to-rate evaluation.
- `<link>/precoders/`: MLP definitions, inference, power handling, checkpoints.
- `<link>/objectives/precoder.py`: differentiable optimization objective.
- `<link>/simulation/system.py`: channel state and physical link evaluation.
- `<link>/results/`: link-specific metrics, reports, persistence, and plots.

Files under `results/` never participate in optimization.
