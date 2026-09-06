# Downlink

`convergence/` is the online training-only baseline. It
optimizes the block precoder under one full-BS power budget and searches
blocklengths for payload completion or fixed-horizon streaming.

`monte_carlo/` trains either per-user precoder networks or a shared-BS
precoder network from deterministic joint channel episodes, then evaluates the
MLP on held-out channels.

The retained objective is normalized inverse-CNR weighted FBL sum rate. Shared
runtime code lives at this folder level; results are stored under `Results` by
method and scenario.
