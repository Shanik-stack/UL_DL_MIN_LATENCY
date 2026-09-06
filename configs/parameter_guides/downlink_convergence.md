# Downlink Convergence Configuration

Use `downlink_dispersion_heavy.yaml`.

Choose `direct_precoder` or `precoder_net`. With a precoder network, choose
`per_user_nets` or `bs_shared_net`. The retained objective is normalized
inverse-CNR weighted FBL sum rate. `max_epochs`, `user_update_lr`, and
`convergence_stopping_rule` control the online solve. `objective_stationarity`
checks relative beam change; `kkt_residuals` also checks the configured
primal, complementarity, and stationarity diagnostics. Streaming uses
`kkt_residuals`.
