# Downlink Methods

Each method is self-contained under this directory.

## Convergence

Start at `convergence/experiment.py::run_convergence_experiment` for the complete
experiment. Follow `optimize_transmission.py::optimize_downlink_transmission`,
which selects `optimize_payload.py` or `optimize_streaming.py`. Both call
`optimize_precoder.py::optimize_precoders_for_block` for the continuous solve.

## Monte Carlo

Start at `monte_carlo/experiment.py::main` for the complete train/test run. The
three algorithm stages are `build_training_rollouts.py`,
`train_precoder_network.py`, and `evaluate_precoder_network.py`.
