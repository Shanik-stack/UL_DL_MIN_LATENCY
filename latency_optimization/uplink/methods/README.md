# Uplink Methods

Each method is self-contained under this directory.

## Convergence

Start at `convergence/experiment.py::run_convergence_experiment` for the complete
experiment. Follow `optimize_transmission.py::run_convergence_baseline`, which
selects `optimize_payload.py` or `optimize_streaming.py`. Both call
`optimize_user_transmission.py` and then `optimize_precoder.py`.

## Monte Carlo

Start at `monte_carlo/experiment.py::main` for the complete train/test run. The
three algorithm stages are `build_training_rollouts.py`,
`train_precoder_network.py`, and `evaluate_precoder_network.py`.
