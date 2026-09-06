# Uplink Online Convergence

This is the training-only uplink method. For one deterministic channel seed it
optimizes each user's precoder block by block, first at `n_kl = T_k` and then
checks smaller feasible blocklengths using the configured search method. It
supports both payload completion and fixed-horizon streaming.

Choose the parameterization in the configuration:

- `convergence_precoder_update_mode: direct_precoder` optimizes the complex
  precoder directly for the current channel.
- `convergence_precoder_update_mode: precoder_net` optimizes the user's
  channel-to-precoder network online.

Run:

```powershell
python -m latency_optimization run --link uplink --method convergence --cfg_name uplink_dispersion_heavy.yaml --seed 3
```

Use `uplink_streaming.yaml` to request fresh `B[k]` bits in every configured
streaming block. Unserved bits are counted as missed and are not carried over.

The method has no separate training and testing phases. Results are saved under
`Results/Uplink/Method-Convergence per epoch/` and mirrored by scenario.
