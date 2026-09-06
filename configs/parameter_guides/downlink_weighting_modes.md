# Downlink Weighting

The cleaned Downlink methods use one objective: normalized inverse-CNR weighted
finite-blocklength sum rate.

For active users `A`:

`J(F) = sum_{k in A} w_k R_k(F, n_k)`

`w_k = (1 / CNR_k) / mean_{j in A}(1 / CNR_j)`

`CNR_k` is computed from user `k`'s channel and noise level. A weaker channel
therefore receives a larger weight, while the mean active-user weight stays at
one. There are no clipping, exponent, backlog, blended, or augmented-loss
options in the cleaned suite.
