# How to Run Experiments

Run commands from the project root:

```powershell
cd "C:\All Codes\Taiwan_Internship\UL_DL_MIN_LATENCY"
```

The unified entry point is:

```powershell
python -m latency_optimization <run|benchmark|batch> ...
```

## Configurations

Current experiment configs in `configs/experiments/` are:

- `uplink_dispersion_heavy.yaml`: uplink payload completion
- `downlink_dispersion_heavy.yaml`: downlink payload completion
- `uplink_streaming.yaml`: uplink streaming
- `downlink_streaming.yaml`: downlink streaming

Pass only the filename; the loader searches the experiment and benchmark config
directories automatically. Names must end in `.yaml`. Scenario values are
strict: use `payload` or `streaming`; old aliases are rejected.

## Convergence

Convergence directly optimizes one seeded channel realization. It has no
separate training dataset.

```powershell
python -m latency_optimization run --link uplink --method convergence --cfg_name uplink_dispersion_heavy.yaml --seed 3
python -m latency_optimization run --link downlink --method convergence --cfg_name downlink_dispersion_heavy.yaml --seed 3
python -m latency_optimization run --link uplink --method convergence --cfg_name uplink_streaming.yaml --seed 3
python -m latency_optimization run --link downlink --method convergence --cfg_name downlink_streaming.yaml --seed 3
```

Add `--quiet` to suppress optimization logging. The config selects
`precoder_net` or `direct_precoder` with
`convergence_precoder_update_mode`. Downlink selects `per_user_nets` or
`bs_shared_net` with `downlink_precoder_net_scope`.

## Monte Carlo

Monte Carlo trains precoder networks on deterministic samples and evaluates
them on held-out samples.

- Payload uses `--num_train_channels` and `--num_test_channels`. One seed is one
  channel episode; later payload blocks are generated during the rollout.
- Streaming uses `--num_train_blocks` and `--num_test_blocks`. One seed is one
  independent streaming block.

Do not mix channel-count and block-count options.

```powershell
python -m latency_optimization run --link uplink --method monte_carlo --cfg_name uplink_dispersion_heavy.yaml --num_train_channels 20 --num_test_channels 10 --test_seed 100
python -m latency_optimization run --link downlink --method monte_carlo --cfg_name downlink_dispersion_heavy.yaml --num_train_channels 20 --num_test_channels 10 --test_seed 100 --quiet
python -m latency_optimization run --link uplink --method monte_carlo --cfg_name uplink_streaming.yaml --num_train_blocks 20 --num_test_blocks 10 --test_seed 100
python -m latency_optimization run --link downlink --method monte_carlo --cfg_name downlink_streaming.yaml --num_train_blocks 20 --num_test_blocks 10 --test_seed 100 --quiet
```

Counts and test seed may instead be set in YAML. CLI values override YAML.
Explicit training seeds are also supported:

```powershell
python -m latency_optimization run --link uplink --method monte_carlo --cfg_name uplink_dispersion_heavy.yaml --train_seeds 1,2,4,5 --test_seed 3
```

When a count generates training seeds, they start at `1` and exclude the test
seed. Held-out seeds start at `--test_seed`, exclude training seeds, and
continue upward until the requested count is reached.

Useful overrides:

```text
--precoder_net_epochs N
--precoder_net_batch_size N
--precoder_net_lr VALUE
--reuse_train_artifact <training/data/train_artifact.pt>
--test_n_search_strategy fixed_step|coarse_to_fine|exponential|binary
--test_n_search_direction ascending|descending
--test_n_search_coarse_step N
--test_n_search_exponential_factor N
```

Uplink Monte Carlo additionally supports `--skip_test`; downlink Monte Carlo
supports `--quiet`. Use method-specific `--help` for the authoritative flags.

## ZF and RZF

Uplink can evaluate one seed directly:

```powershell
python -m latency_optimization benchmark --link uplink --name zf --cfg_name uplink_dispersion_heavy.yaml --seed 3
python -m latency_optimization benchmark --link uplink --name rzf --cfg_name uplink_dispersion_heavy.yaml --seed 3
```

For fair Monte Carlo comparison, use the held-out manifest generated at:

```text
Results/<Link>/<scenario>/monte_carlo/<run>/testing/channels/manifest.json
```

Payload uses `testing/channels/manifest.json`; streaming uses
`testing/blocks/manifest.json`. ZF and RZF support both scenarios and preserve
the scenario semantics from the matching configuration.

Uplink held-out benchmarks:

```powershell
python -m latency_optimization benchmark --link uplink --name zf --cfg_name uplink_dispersion_heavy.yaml --test_manifest "<manifest.json>"
python -m latency_optimization benchmark --link uplink --name rzf --cfg_name uplink_dispersion_heavy.yaml --test_manifest "<manifest.json>"
```

Downlink ZF and RZF always require a manifest:

```powershell
python -m latency_optimization benchmark --link downlink --name zf --cfg_name downlink_dispersion_heavy.yaml --test_manifest "<manifest.json>"
python -m latency_optimization benchmark --link downlink --name rzf --cfg_name downlink_dispersion_heavy.yaml --test_manifest "<manifest.json>"
```

Use the same config and manifest for Monte Carlo, ZF, and RZF. The manifest
fixes test seeds and per-user SNR values.

## Exhaustive Search

Exhaustive payload validation is currently uplink-only:

```powershell
python -m latency_optimization benchmark --link uplink --name exhaustive --cfg_name uplink_exhaustive_dispersion_heavy.yaml --seed 3 --catalog_mode full
```

`--catalog_mode` accepts `none`, `outcomes_only`, or `full`.

## Batch Runs

The default file is `configs/batch_runs/run_all.yaml`. It runs both links,
methods, and scenarios in parallel terminals by default.

```powershell
python -m latency_optimization batch
python -m latency_optimization batch --dry_run
python -m latency_optimization batch --links uplink --methods convergence
python -m latency_optimization batch --configs uplink_dispersion_heavy --launch_mode sequential
python -m latency_optimization batch --continue_on_error
```

## Results

Results use one hierarchy:

```text
Results/<Uplink|Downlink>/<payload_completion|streaming>/<method>/<run_tag>/
```

The run tag contains the config stem, a hash of the complete YAML contents, and
the relevant seed. Changing a config creates a distinct result directory even
when its filename is unchanged.

Convergence stores `testing/` only. Monte Carlo stores:

```text
training/data/                   summaries and train_artifact.pt
training/channels/ or blocks/    deterministic samples and manifest.json
testing/data/                    aggregate held-out results
testing/channels/ or blocks/     deterministic samples and manifest.json
testing/samples/                 one detailed directory per test sample
```

Every experiment root contains `run_manifest.json`.

## Help

```powershell
python -m latency_optimization --help
python -m latency_optimization run --link uplink --method convergence --help
python -m latency_optimization run --link downlink --method monte_carlo --help
python -m latency_optimization benchmark --link uplink --name exhaustive --help
python -m latency_optimization batch --help
```
