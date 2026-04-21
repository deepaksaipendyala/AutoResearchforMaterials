# Experiment Plan

Project family: process_property_prediction
Data modalities detected: tabular, composition, processing

Current baseline:
- Trainer: `tabular_mlp`
- Config file: `specs/model_config.json`
- Primary metric: `mae`
- Split method: `group`

Agent workflow:
1. Validate the project.
2. Train or inspect the current baseline.
3. Pick one small change in `specs/model_config.json`.
4. Train again and compare against the best value in `agent/experiment_log.tsv`.
5. Keep notes here about why the change should help this materials problem.

Candidate experiment ideas:
- Switch `trainer` from `tabular_mlp` to `tabm` for the main TabM baseline.
- Tune `hidden_dim`, `dropout`, `lr`, or `batch_size`.
- Tune TabM settings: `k`, `n_blocks`, `d_block`, `dropout`, and weight decay.
- Adjust feature roles in `specs/data_dictionary.csv` only after asking the researcher.
- For composition-heavy projects, add or request composition featurization.
- For structure-heavy projects, add or request structure descriptors before using structure files directly.
