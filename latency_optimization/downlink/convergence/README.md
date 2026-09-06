# Downlink Online Convergence

This is the training-only downlink method. It optimizes a block precoder for
the active users, applies the full-BS power projection, then searches for the
smallest feasible `n_kl` values.

The configuration keeps two deliberate network-scope choices:

- `convergence_precoder_update_mode`: `direct_precoder` or `precoder_net`.
- `downlink_precoder_net_scope`: `per_user_nets` or `bs_shared_net`.

The only neural model is an MLP. The retained downlink objective is normalized inverse-CNR weighted sum rate,
so weaker instantaneous channels receive larger weight.

With `downlink_streaming.yaml`, every user requests fresh `B[k]` bits in each
of the configured blocks. A partial block records delivered and missed bits;
the missed bits are not carried into the next block.

Run:

```powershell
python -m latency_optimization run --link downlink --method convergence --cfg_name downlink_dispersion_heavy.yaml --seed 3
```

The method has no separate training and testing phases. Results are saved under
`Results/Downlink/Method-Convergence per epoch/` and mirrored by scenario.
