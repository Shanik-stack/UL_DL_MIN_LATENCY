# Code Organization

All executable Python source belongs to the `latency_optimization` package.
Configuration, generated data, documentation, and tests remain outside the
package so simulation code cannot accidentally depend on artifacts.

Configuration ownership is explicit: `configs/experiments/` holds complete
runs, `configs/benchmarks/` holds small validation-only systems, and
`configs/batch_runs/` holds launch manifests.

## Execution Flow

Every supported command enters through `latency_optimization/__main__.py`:

```text
command line
  -> link and method runner
  -> scenario construction
  -> solver or Monte Carlo trainer/evaluator
  -> uplink/downlink physical system
  -> result metrics, plots, and persistence
```

The method `main.py` modules only parse method-specific arguments and
orchestrate a run. Mathematical optimization stays in `objective.py`,
`solver.py`, and `allocation.py`.

## Package Layout

```text
latency_optimization/
  __main__.py
  project.py
  cli/
    batch.py
  core/
    blocklength.py
    scenarios.py
  experiments/
    channels.py
    cost.py
    determinism.py
    monte_carlo_testing.py
    seeds.py
  results/
    console.py
    naming.py
    paths.py
    persistence.py
  uplink/
    config.py
    system.py
    system_parameters.py
    precoder_models.py
    simulation.py
    reporting.py
    plotting.py
    convergence/
    monte_carlo/
    benchmarks/
  downlink/
    config.py
    system.py
    precoder_models.py
    user_weights.py
    runner.py
    plotting.py
    convergence/
    monte_carlo/
    benchmarks/
```

## Responsibilities

| Package | Responsibility |
| --- | --- |
| `core` | Scenario semantics and blocklength search shared by both links |
| `experiments` | Seeds, channel datasets, deterministic execution, and cost accounting |
| `results` | Output paths, stable result names, serialization, and terminal formatting |
| `uplink` | Uplink channel model, rate model, precoders, methods, and figures |
| `downlink` | Downlink channel model, precoders, weighting, methods, and figures |
| `cli` | Batch execution only; no physical-layer or optimization equations |

## Method Layout

Both links use the same method boundaries:

| Module | Responsibility |
| --- | --- |
| `convergence/objective.py` | Objective and constraint definitions |
| `convergence/solver.py` | Precoder optimization and stopping conditions |
| `convergence/allocation.py` | Payload progression and blocklength decisions |
| `monte_carlo/network_operations.py` | Differentiable rates, model forwards, and snapshots |
| `monte_carlo/rollout.py` | Channel episodes and visited blocklength states |
| `monte_carlo/trainer.py` | Offline network optimization and checkpoint selection |
| `monte_carlo/evaluator.py` | Held-out channel scheduling and result construction |
| `*/main.py` | CLI arguments and orchestration only |

## Dependency Rules

Dependencies flow in one direction:

```text
core -> link system/model -> objective -> solver -> allocation
experiments -> rollout -> trainer/evaluator -> runner
completed result dictionaries -> results/reporting/plotting
```

Physical-system and optimization modules must not import CLI, plotting, or
generated result files. Uplink and downlink modules must not import each other.
There are no `sys.path` mutations or wildcard imports.

## User Commands

```powershell
python -m latency_optimization run --link uplink --method convergence --cfg_name uplink_dispersion_heavy.yaml --seed 3
python -m latency_optimization run --link downlink --method monte_carlo --cfg_name downlink_dispersion_heavy.yaml --test_seed 3
python -m latency_optimization batch --batch_run run_all.yaml --dry_run
python -m latency_optimization benchmark --link uplink --name rzf --cfg_name uplink_dispersion_heavy.yaml --seed 3
```
