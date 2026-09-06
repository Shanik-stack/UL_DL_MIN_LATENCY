# Configurations

Configuration files are separated by purpose:

- `experiments/` contains complete physical-system and simulation definitions
  for one experiment.
- `batch_runs/` selects which experiment configurations and methods to launch
  together. It does not define channel or optimization parameters.
- `benchmarks/` contains intentionally small configurations used only by
  validation benchmarks such as exhaustive search.
- `parameter_guides/` documents every supported configuration decision and
  numerical parameter.

Experiment entry points accept a short name such as
`--cfg_name uplink_dispersion_heavy.yaml`; the code automatically searches
`configs/experiments/`.

The default batch can be inspected without running simulations:

```powershell
python -m latency_optimization batch --dry_run
```

See `parameter_guides/README.md` for the active configuration list and links
to the detailed guides.
