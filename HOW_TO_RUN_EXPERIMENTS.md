# Running the Latency Optimization Experiments

This guide shows how to run the uplink/downlink experiments and benchmarks from the project root.

All commands use:

```bash
python -m latency_optimization ...
```

Run them from the directory that contains the `latency_optimization/` package.

---

## Command-Line Flags

### `run`

Used for the main convergence and Monte Carlo experiments.

```bash
python -m latency_optimization run --link <LINK> --method <METHOD> [experiment-specific flags]
```

#### `--link`

Selects which communication link is simulated.

Allowed values:

```text
uplink
downlink
```

Example:

```bash
--link uplink
```

---

#### `--method`

Selects the experiment type.

Allowed values:

```text
convergence
monte_carlo
```

Example:

```bash
--method convergence
```

---

#### `--cfg_name`

Specifies the YAML configuration file used by the selected experiment.

The filename can be supplied directly:

```bash
--cfg_name uplink_dispersion_heavy.yaml
```

A `config/` prefix is **not required**.

For the uplink convergence experiment, the default is:

```text
uplink_dispersion_heavy.yaml
```

If `--cfg_name` is omitted, that default configuration is used.

---

#### `--seed`

Sets the deterministic random seed for the experiment.

Example:

```bash
--seed 0
```

Using the same seed and configuration makes stochastic parts of the experiment reproducible.

For uplink convergence, the default is:

```text
0
```

---

#### `--help`

Displays help for the selected command or experiment.

Examples:

```bash
python -m latency_optimization --help
```

```bash
python -m latency_optimization run --link uplink --method convergence --help
```

---

## Main Experiments

## 1. Uplink Convergence

Runs:

```text
latency_optimization.uplink.convergence.main
```

Basic command:

```bash
python -m latency_optimization run --link uplink --method convergence
```

With an explicit configuration:

```bash
python -m latency_optimization run --link uplink --method convergence --cfg_name uplink_dispersion_heavy.yaml
```

With configuration and seed:

```bash
python -m latency_optimization run --link uplink --method convergence --cfg_name uplink_dispersion_heavy.yaml --seed 0
```

General form:

```bash
python -m latency_optimization run --link uplink --method convergence --cfg_name <CONFIG_FILE>.yaml --seed <SEED>
```

Example:

```bash
python -m latency_optimization run --link uplink --method convergence --cfg_name my_uplink_experiment.yaml --seed 42
```

---

## 2. Uplink Monte Carlo

Runs:

```text
latency_optimization.uplink.monte_carlo.main
```

Basic command:

```bash
python -m latency_optimization run --link uplink --method monte_carlo
```

With a configuration:

```bash
python -m latency_optimization run --link uplink --method monte_carlo --cfg_name <CONFIG_FILE>.yaml
```

With a configuration and any experiment-specific arguments:

```bash
python -m latency_optimization run --link uplink --method monte_carlo --cfg_name <CONFIG_FILE>.yaml <OTHER_ARGUMENTS>
```

To see the exact Monte Carlo-specific options:

```bash
python -m latency_optimization run --link uplink --method monte_carlo --help
```

---

## 3. Downlink Convergence

Runs:

```text
latency_optimization.downlink.convergence.main
```

Basic command:

```bash
python -m latency_optimization run --link downlink --method convergence
```

With a configuration:

```bash
python -m latency_optimization run --link downlink --method convergence --cfg_name <CONFIG_FILE>.yaml
```

Example:

```bash
python -m latency_optimization run --link downlink --method convergence --cfg_name my_downlink_experiment.yaml
```

To see all downlink convergence-specific arguments:

```bash
python -m latency_optimization run --link downlink --method convergence --help
```

---

## 4. Downlink Monte Carlo

Runs:

```text
latency_optimization.downlink.monte_carlo.main
```

Basic command:

```bash
python -m latency_optimization run --link downlink --method monte_carlo
```

With a configuration:

```bash
python -m latency_optimization run --link downlink --method monte_carlo --cfg_name <CONFIG_FILE>.yaml
```

To see all Monte Carlo-specific arguments:

```bash
python -m latency_optimization run --link downlink --method monte_carlo --help
```

---

# Benchmarks

Benchmark commands use:

```bash
python -m latency_optimization benchmark --link <LINK> --name <BENCHMARK>
```

## Benchmark Flags

### `--link`

Selects:

```text
uplink
downlink
```

---

### `--name`

Selects the benchmark method.

Allowed values:

```text
zf
rzf
exhaustive
```

where:

- `zf` = Zero Forcing
- `rzf` = Regularized Zero Forcing
- `exhaustive` = Exhaustive Search

The exhaustive-search benchmark is currently available for **uplink only**.

---

## 5. Uplink Zero-Forcing Benchmark

```bash
python -m latency_optimization benchmark --link uplink --name zf
```

Runs:

```text
latency_optimization.uplink.benchmarks.zero_forcing.main
```

Additional benchmark-specific arguments can be appended to the command.

Help:

```bash
python -m latency_optimization benchmark --link uplink --name zf --help
```

---

## 6. Uplink Regularized Zero-Forcing Benchmark

```bash
python -m latency_optimization benchmark --link uplink --name rzf
```

Runs:

```text
latency_optimization.uplink.benchmarks.regularized_zero_forcing.main
```

Help:

```bash
python -m latency_optimization benchmark --link uplink --name rzf --help
```

---

## 7. Uplink Exhaustive-Search Benchmark

```bash
python -m latency_optimization benchmark --link uplink --name exhaustive
```

Runs:

```text
latency_optimization.uplink.benchmarks.exhaustive_search.exhaustive_payload_compare
```

Help:

```bash
python -m latency_optimization benchmark --link uplink --name exhaustive --help
```

The following is **not valid**:

```bash
python -m latency_optimization benchmark --link downlink --name exhaustive
```

because exhaustive search is currently restricted to the uplink.

---

## 8. Downlink Zero-Forcing Benchmark

```bash
python -m latency_optimization benchmark --link downlink --name zf
```

Downlink ZF is evaluated through:

```text
latency_optimization.downlink.benchmarks.evaluate_test_dataset
```

with:

```text
--method zf
```

---

## 9. Downlink Regularized Zero-Forcing Benchmark

```bash
python -m latency_optimization benchmark --link downlink --name rzf
```

Downlink RZF is evaluated through:

```text
latency_optimization.downlink.benchmarks.evaluate_test_dataset
```

with:

```text
--method rzf
```

---

# Uplink Benchmark Test Dataset

For uplink `zf` and `rzf`, the normal benchmark module is used unless a test manifest is supplied.

If the benchmark arguments contain:

```bash
--test_manifest ...
```

the CLI switches to:

```text
latency_optimization.uplink.benchmarks.evaluate_test_dataset
```

For example:

```bash
python -m latency_optimization benchmark --link uplink --name zf --test_manifest <MANIFEST>
```

or:

```bash
python -m latency_optimization benchmark --link uplink --name rzf --test_manifest <MANIFEST>
```

---

# Batch Runs

Batch experiments use:

```bash
python -m latency_optimization batch ...
```

The remaining arguments are forwarded to:

```text
latency_optimization.cli.batch
```

To see the exact batch syntax:

```bash
python -m latency_optimization batch --help
```

---

# Quick Reference

| Experiment | Command |
|---|---|
| Uplink convergence | `python -m latency_optimization run --link uplink --method convergence --cfg_name <CONFIG>.yaml` |
| Uplink Monte Carlo | `python -m latency_optimization run --link uplink --method monte_carlo --cfg_name <CONFIG>.yaml` |
| Downlink convergence | `python -m latency_optimization run --link downlink --method convergence --cfg_name <CONFIG>.yaml` |
| Downlink Monte Carlo | `python -m latency_optimization run --link downlink --method monte_carlo --cfg_name <CONFIG>.yaml` |
| Uplink ZF | `python -m latency_optimization benchmark --link uplink --name zf` |
| Uplink RZF | `python -m latency_optimization benchmark --link uplink --name rzf` |
| Uplink exhaustive | `python -m latency_optimization benchmark --link uplink --name exhaustive` |
| Downlink ZF | `python -m latency_optimization benchmark --link downlink --name zf` |
| Downlink RZF | `python -m latency_optimization benchmark --link downlink --name rzf` |

---

# Example

To run uplink convergence using `uplink_dispersion_heavy.yaml` with seed `0`:

```bash
python -m latency_optimization run --link uplink --method convergence --cfg_name uplink_dispersion_heavy.yaml --seed 0
```

The top-level CLI consumes:

```text
run
--link uplink
--method convergence
```

and forwards:

```text
--cfg_name uplink_dispersion_heavy.yaml
--seed 0
```

to the uplink convergence experiment.
