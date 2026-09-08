# Code Organization

All executable source is under `latency_optimization`. The package is organized
around two questions: which communication link is being simulated, and which
method is being run.

```text
latency_optimization/
  __main__.py                 unified CLI dispatch
  experiments/               configuration, scenarios, channels, seeds, cost
  optimization/              blocklength search and convergence criteria
  physics/                    finite-blocklength rate equations
  precoders/                  shared power, parameters, state, serialization
  results/                    metrics, paths, plots, and persistence
  uplink/
    configuration/            config loading and validation
    simulation/               system state, parameters, and schedule operations
    physics/                  uplink SNR/SINR covariance and rate evaluation
    objectives/               differentiable objective and objective settings
    precoders/                uplink MLP, inference, checkpoints
    results/                  metrics, reports, persistence, and plots
    methods/
      convergence/            online per-channel optimization
      monte_carlo/            offline training and held-out evaluation
    benchmarks/               ZF, RZF, exhaustive search
  downlink/
    configuration/            config loading and validation
    simulation/               system state, block state, and random baselines
    physics/                  downlink interference-coupled rate evaluation
    objectives/               coupled-BS objective, settings, and user weights
    precoders/                downlink MLP, inference, checkpoints, model service
    results/                  metrics, reports, persistence, and plots
    methods/
      convergence/            online per-channel optimization
      monte_carlo/            offline training and held-out evaluation
    benchmarks/               ZF and RZF
```

## Method Layout

Every method follows the same navigation convention:

- `experiment.py`: complete run from configuration to persisted results.
- `optimize_transmission.py`: convergence block and blocklength procedure.
- `optimize_payload.py`: payload-completion block loop.
- `optimize_streaming.py`: fixed-horizon streaming block loop.
- `optimize_precoder.py`: convergence solve for a fixed blocklength state.
- `build_training_rollouts.py`: Monte Carlo dataset and visited-state generation.
- `train_precoder_network.py`: Monte Carlo epoch and optimizer loop.
- `evaluate_precoder_network.py`: inference-only held-out scheduling.
- `precoder_network.py`: differentiable network operations used in training.

Only files applicable to a method are present. There are no old-path wrappers.

## Dependency Direction

```text
experiments / optimization / physics / precoders
                      -> link simulation, objective, and precoders
                      -> method optimization procedure
                      -> method experiment entry point
                      -> results
```

Shared packages never import uplink or downlink implementations. Uplink and
downlink never import each other. Monte Carlo and convergence may share link
objectives, systems, precoders, and reporting, but must not import each other's
method internals.

## Tensor Boundary

Torch tensors are canonical inside inference, objectives, rate evaluation, and
gradient optimization. Conversion to NumPy occurs only when a completed schedule
is committed to simulator state or passed to reporting. Mathematical objectives
accept tensors rather than simulator objects.

## Changing Behavior

- Convergence scenario dispatch: `<link>/methods/convergence/optimize_transmission.py`.
- Payload/streaming behavior: `<link>/methods/convergence/optimize_payload.py`
  or `<link>/methods/convergence/optimize_streaming.py`.
- Fixed-blocklength gradient solve: `<link>/methods/convergence/optimize_precoder.py`.
- Monte Carlo state generation: `<link>/methods/monte_carlo/build_training_rollouts.py`.
- Monte Carlo learning: `<link>/methods/monte_carlo/train_precoder_network.py`.
- Monte Carlo test scheduling: `<link>/methods/monte_carlo/evaluate_precoder_network.py`.
- MLP architecture: `<link>/precoders/models.py`.
- Link configuration: `<link>/configuration/loader.py`.
- Link simulation state: `<link>/simulation/system.py`.
- Differentiable objective: `<link>/objectives/precoder.py`.
- Uplink SNR/SINR construction: `uplink/physics/rate.py`.
- Shared rate equation: `physics/rate_law.py` or `physics/finite_blocklength.py`.
- Reporting and plots: `<link>/results/`.
- Scenario semantics: `experiments/scenarios.py`.
- Blocklength candidate ordering: `optimization/blocklength_search.py`.

The full call graph is in [OPTIMIZATION_FLOW.md](OPTIMIZATION_FLOW.md).
Commands are in [HOW_TO_RUN_EXPERIMENTS.md](../HOW_TO_RUN_EXPERIMENTS.md).
