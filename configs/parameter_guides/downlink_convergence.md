# Downlink Convergence Configuration

Use `downlink_dispersion_heavy.yaml`.

Choose `direct_precoder` or `precoder_net`. With a precoder network, choose
`per_user_nets` or `bs_shared_net`. The retained objective is normalized
inverse-CNR weighted FBL sum rate. `max_epochs`, `user_update_lr`, and the
three `kkt_*_tol` values control the online solve.
