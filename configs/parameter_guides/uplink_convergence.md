# Uplink Convergence Configuration

Use `uplink_dispersion_heavy.yaml`.

Choose `direct_precoder` to optimize the current user's complex precoder, or
`precoder_net` to optimize that user's channel-to-precoder network online.
Use `max_epochs`, `lr_net`, and the three `kkt_*_tol` values to control the
inner solve.
Blocklength search is controlled by the shared `n_search_*` fields described in
`PARAMETER_GUIDE.md`.
