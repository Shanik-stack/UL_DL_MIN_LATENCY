# Exhaustive Payload Validation

This small benchmark enumerates feasible `(B_kl, n_kl)` allocations for two
uplink users and compares the best allocation with the online convergence
strategy. Its `configs/benchmarks/uplink_exhaustive_dispersion_heavy.yaml`
configuration uses the same short-blocklength, strict-error regime as the main
experiments, but keeps the state space deliberately small. It is a validation
tool, not a scalable method.

Run:

```powershell
python -m latency_optimization benchmark --link uplink --name exhaustive --seed 3
```

Pass `--cfg_name` only when intentionally using another small `K=2` config.
