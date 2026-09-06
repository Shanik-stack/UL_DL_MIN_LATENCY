# Experiment Results

Each new run is stored in one canonical location:

```text
Results/
  Uplink or Downlink/
    payload_completion or streaming/
      convergence_per_epoch/
      monte_carlo/
      benchmark_zf/
      benchmark_rzf/
      benchmark_exhaustive/
        <run_id>/
          run_manifest.json
          run_manifest.txt
          training/              # Monte Carlo only
            data/                # model, dataset summary, training summary
            channels/ or blocks/ # saved base samples and their manifest
            optimization_history/# network-training curves
            evaluation/          # uplink post-training sample evaluation
          testing/
            data/                # aggregate held-out test-set summary
            channels/ or blocks/ # saved held-out samples and their manifest
            samples/
              seed_<seed>/
                data/
                user_config/
                latency_asynchronality/
                link_quality/
                optimization_history/
                schedule_details/
                interference/
```

The run ID contains the method, configuration stem, configuration-content hash,
and seed or test-set identifier. The hash changes when the configuration values
change, so results from different configurations cannot be mistaken for one
another.

`run_manifest.json` is the entry point for a run. It records the setup first,
then the status, artifact-health result, plot count, files by section, zero-byte
files, and empty folders removed during finalization. The human-readable
`run_manifest.txt` groups the same file index by section for quick navigation.

Run `python -m latency_optimization.results.index` after a batch of experiments
to create `Results/index.json` and `Results/index.txt`, which list every
completed run by link, scenario, method, configuration hash, seed, and file
count.

Convergence runs use `testing/` because they do not have a learned training
phase. Monte Carlo runs use `training/` for the model and base samples and
`testing/samples/seed_<seed>/` for each held-out payload channel episode or
streaming block. Empty plot categories are removed when a run is finalized.

Older `Method-*`, `Scenario-*`, and other historical folders may remain for
previous experiments. New runs do not write to or mirror into those folders.
