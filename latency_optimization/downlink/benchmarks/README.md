# Downlink Benchmark Methods

`evaluate_test_dataset.py` evaluates standard stacked-channel ZF or RZF on a
saved held-out channel manifest. Both baselines use the same channel episodes,
finite-blocklength rate equation, payload-completion rule, and blocklength
search as the retained methods; only the beamformer changes.

```powershell
python -m latency_optimization benchmark --link downlink --name zf --cfg_name downlink_dispersion_heavy.yaml --test_manifest <test_channels.json>
python -m latency_optimization benchmark --link downlink --name rzf --cfg_name downlink_dispersion_heavy.yaml --test_manifest <test_channels.json>
```

The `Exhaustive Search` subfolder is reserved for the small downlink allocation
validator. Its implementation is not currently present; the runnable uplink
validator is documented in the corresponding uplink benchmark folder.
