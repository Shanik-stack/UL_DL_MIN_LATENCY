# Configuration Guide

Individual experiment configurations are stored in `../experiments/`:

- `uplink_dispersion_heavy.yaml`
- `downlink_dispersion_heavy.yaml`
- `uplink_streaming.yaml`
- `downlink_streaming.yaml`

Batch-run files are stored separately in `../batch_runs/`. The default
`run_all.yaml` selects which links, methods, and experiment configurations are
launched together; it does not define the communication system.

All four system configurations use the canonical dispersion-heavy regime:
short blocklengths, `epsilon = 1e-14`, and 4 dB reference SNR. The
`dispersion_heavy` files run payload completion. The `streaming` files request
fresh `test.B[k]` bits from every user in every block for the configured block
horizon; missed bits are measured but not carried forward. Augmented
Lagrangians, alternate reward functions, and removed model variants are not
part of the cleaned suite.

The exhaustive-search validator uses the intentionally small
`../benchmarks/uplink_exhaustive_dispersion_heavy.yaml` configuration. Keeping
benchmark configurations separate prevents their state-space limits from being
mistaken for main experiment settings.

See `PARAMETER_GUIDE.md` for the complete public configuration surface.
