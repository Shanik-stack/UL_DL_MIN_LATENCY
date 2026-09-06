# Run All Experiments

This project exposes one package-level batch command:

`python -m latency_optimization batch`

and one default batch-run configuration:

`C:\All Codes\Taiwan_Internship\UL_UPLINK_DOWNLINK_MONTE_CARLO\configs\batch_runs\run_all.yaml`

## Default full run

From `C:\All Codes\Taiwan_Internship\UL_UPLINK_DOWNLINK_MONTE_CARLO`:

```powershell
python -m latency_optimization batch
```

This launches all selected experiments in parallel, each in its own PowerShell window.

The default file runs both active experiment configurations for all four
link/method combinations, producing eight runs:

- Uplink convergence on all `uplink_*.yaml` configs
- Uplink Monte Carlo on all `uplink_*.yaml` configs
- Downlink convergence on all `downlink_*.yaml` configs
- Downlink Monte Carlo on all `downlink_*.yaml` configs

For Monte Carlo runs, the batch-run file sets test seed 3. Each referenced
experiment configuration controls its own number of training channels.

## Dry run

```powershell
python -m latency_optimization batch --dry_run
```

This prints the full command matrix without opening any new terminals.

Use another batch-run file by name or path:

```powershell
python -m latency_optimization batch --batch_run run_all.yaml --dry_run
```

## Useful filters

Only uplink:

```powershell
python -m latency_optimization batch --links uplink
```

Only convergence:

```powershell
python -m latency_optimization batch --methods convergence
```

Only one config stem:

```powershell
python -m latency_optimization batch --configs uplink_dispersion_heavy
```

Keep going after a failed run:

```powershell
python -m latency_optimization batch --continue_on_error
```

Run everything sequentially in the current terminal instead:

```powershell
python -m latency_optimization batch --launch_mode sequential
```

## Batch-run format

The batch-run file is a YAML file with:

- `defaults`
  - shared fallback settings such as `seed`, `test_seed`, `train_seeds`, `quiet`
- `runs`
  - one run group per `(link, method)` combination

Example:

```yaml
defaults:
  seed: 3
  launch_mode: parallel_terminals

runs:
  - link: uplink
    method: convergence
    cfg_names: [uplink_dispersion_heavy.yaml, uplink_streaming.yaml]
    seed: 3

  - link: downlink
    method: monte_carlo
    cfg_names:
      - downlink_dispersion_heavy.yaml
    test_seed: 3
    num_train_channels: 10
    quiet: true
```

Supported method names:

- `convergence`
- `monte_carlo`

Supported link names:

- `uplink`
- `downlink`

Supported launch modes:

- `parallel_terminals`
- `sequential`

For Monte Carlo entries in the batch-run file:

- `train_seeds` passes an explicit comma-separated seed list to the method.
- `num_train_channels` passes `--num_train_channels N`, which creates exactly `N` training channel episodes while skipping `test_seed`.
- `num_train_blocks` passes `--num_train_blocks N` for streaming, which creates exactly `N` independent one-block training samples while skipping `test_seed`.
- `num_test_channels` and `num_test_blocks` provide the matching payload or streaming test-set count.
- Use channel-count options only with payload configs and block-count options only with streaming configs.
- If neither is provided, the method uses the config defaults.

The batch command launches the same package-level `run` command used for
individual experiments. It does not execute implementation files by path.
