# Uplink Convergence Configuration

Use `uplink_dispersion_heavy.yaml`.

Choose `direct_precoder` to optimize the current user's complex precoder, or
`precoder_net` to optimize that user's channel-to-precoder network online.
Use `max_epochs`, `lr_net`, and `convergence_stopping_rule` to control the
inner solve. `objective_stationarity` checks only relative beam change;
`kkt_residuals` also checks the configured primal, complementarity, and
stationarity diagnostics. Streaming uses `kkt_residuals`.
Blocklength search is controlled by the shared `n_search_*` fields described in
`PARAMETER_GUIDE.md`.
