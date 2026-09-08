# Code Organization

All executable source is in `latency_optimization`. Configuration, generated
channels, results, and documentation sit outside that package. A method never
imports plots, saved results, or command-line code.

```text
latency_optimization/
  core/            scenario rules, validation, and n-search
  physics/         finite-blocklength rate laws
  precoders/       shared complex-parameter and power operations
  optimization/    convergence diagnostics and stopping rules
  experiments/     seed, channel, cost, and test-dataset infrastructure
  results/         shared metrics, plots, names, paths, and persistence
  uplink/
    objective.py   `UplinkPrecoderObjective` for one user and block
    objective_settings.py  uplink objective selection and validation
    precoders/     MLP models, Torch inference, and checkpoints
    convergence/   online precoder optimization and allocation
    monte_carlo/   rollout, trainer, evaluator
    benchmarks/    ZF, RZF, and exhaustive validation
  downlink/
    objective.py   `DownlinkPrecoderObjective` for one coupled BS block
    objective_settings.py  downlink objective selection and display names
    reporting.py   downlink result assembly shared by experiment methods
    precoders/     MLP models, inference, and checkpoints
    convergence/   joint BS optimization and allocation
    monte_carlo/   rollout, trainer, evaluator
    benchmarks/    ZF and RZF evaluation
```

## Ownership

`physics/rate_law.py` is the only finite-blocklength rate-law registry.
`physics/finite_blocklength.py` contains one Torch implementation of the rate
equation. NumPy implementations of optimization or rate equations are not
allowed.
`precoders/` owns complex parameter conversion and power projection.
It also owns framework-neutral model checkpoint and parameter-change handling.
`uplink/objective.py` and `downlink/objective.py` expose matching
`nn.Module` objective interfaces with named result dictionaries. The uplink
objective evaluates one user's independent precoder; the downlink objective
evaluates all active user slices of the joint BS precoder because their SINRs
and the BS power constraint are coupled. Configuration parsing and labels do
not belong in either mathematical objective, and objectives receive Torch
tensors rather than simulator objects. `convergence/solver.py` improves a beam; `allocation.py` decides
payload bits and `n_kl`. Monte Carlo `rollout.py` selects visited states,
`trainer.py` updates network weights, and `evaluator.py` runs held-out
schedules.

`optimization/stopping.py` owns both stopping modes:

- `objective_stationarity` uses only relative precoder change.
- `kkt_residuals` also requires primal, complementarity, and stationarity
  diagnostics. Streaming configurations select this mode.

Every experiment configuration is strict. Labels such as `payload`,
`streaming`, `per_user_nets`, and `bs_shared_net` have no aliases.
Both links expose the same `load_config()` contract, returning system
parameters, simulation parameters, and run metadata.

## Tensor Boundary

Torch tensors are the canonical representation inside model inference,
objectives, rate evaluation, candidate search, and gradient optimization.
Channels loaded from generated datasets are converted when they enter this
compute layer. Precoder tensors are converted to NumPy only when a completed
schedule is committed to the current simulator or passed to persistence and
plotting code. Shared conversion helpers live in `precoders/serialization.py`;
downlink joint-precoder assembly lives in `downlink/model_service.py`. Model
and physics modules must not return NumPy arrays.

`results/metrics.py` owns link-independent latency, asynchronality, and
reference-schedule schemas. `results/plotting.py` owns plot primitives shared
by uplink and downlink. Method evaluators calculate schedules and diagnostics;
they do not redefine result metrics or plotting mechanics.

`core/blocklength.py` owns every blocklength-search strategy and the shared
Monte Carlo training/testing search configuration. Link implementations supply
candidate evaluations but cannot redefine how candidate values are generated.

## Dependency Direction

```text
core / physics / precoders
        -> link system and objective
        -> convergence solver or Monte Carlo rollout
        -> allocation, evaluator, runner
        -> results and plots
```

The uplink and downlink packages do not import each other. Benchmarks may call
the public online allocation entry point only when comparing against the same
online scheduling rule.

## Extending Safely

- Add a rate law in `physics/rate_law.py`, register it there, and select it
  through `finite_blocklength_rate_law`.
- Add a downlink network in `downlink/precoders/models.py` and matching
  inference/checkpoint metadata in that package. Do not duplicate an objective
  or evaluator.
- Add a training strategy as a separate Monte Carlo trainer that reuses the
  existing objective, rollout, evaluator, and result persistence layers.
- Add a scenario in `core/scenarios.py`; both links then receive the same
  semantics automatically.

Run commands and configuration details are in
[`HOW_TO_RUN_EXPERIMENTS.md`](../HOW_TO_RUN_EXPERIMENTS.md) and
[`configs/parameter_guides/PARAMETER_GUIDE.md`](../configs/parameter_guides/PARAMETER_GUIDE.md).
