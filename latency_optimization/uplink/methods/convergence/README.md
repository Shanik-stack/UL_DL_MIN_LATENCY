# Uplink Online Convergence

This is the training-only uplink method. For one deterministic channel seed it
optimizes each user's precoder block by block, first at `n_kl = T_k` and then
checks smaller feasible blocklengths using the configured search method. It
supports both payload completion and fixed-horizon streaming.

Start at `experiment.py::run_convergence_experiment`, then follow
`optimize_transmission.py::run_convergence_baseline`. It selects
`optimize_payload.py` or `optimize_streaming.py`.
`optimize_user_transmission.py::optimize_user_blocklength_and_precoder` shows
the full beam/blocklength sequence for one user and block, and calls
`optimize_precoder.py::optimize_precoder_for_nl` for each fixed-n gradient solve. `experiment.py`
handles experiment setup, reporting, and CLI arguments rather than optimization.

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
