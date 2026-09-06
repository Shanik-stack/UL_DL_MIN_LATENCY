# Uplink

`convergence/` is the online training-only baseline. It
optimizes each user's precoder for the current deterministic channel and
searches `n_kl` under payload completion or streaming.

`monte_carlo/` trains one user-specific precoder network from channel
episodes, then evaluates the trained networks on a held-out seed.

`benchmarks/` contains closed-form ZF/RZF baselines and the
small exhaustive validator. Shared system, FBL-rate, result, and plotting code
stays at this folder level.
