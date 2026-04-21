# Research Brief

Project: Tiny Materials Band Gap

Scientific goal:
Predict band gap from simple composition and measurement descriptors.

Project family:
process_property_prediction

Data modalities:
tabular, composition, processing

Data to use:
- Primary table: `data/raw/measurements.csv`
- Target column: `band_gap_eV`
- ID column: `material_id`
- Structure column: `not set`
- Group column: `chemical_system`

Evaluation rule:
- Task type: `regression`
- Primary metric: `mae`
- Split method: `group`

Agent instructions:
1. Validate the project before training.
2. Do not edit files in `data/raw/`.
3. Use `agent/edit_scope.json` to decide which files can be changed without asking.
4. Treat the primary metric and split method as scientific decisions. Ask before changing them.
5. Log every run in `agent/experiment_log.tsv`.
6. Explain results in terms of material-property prediction, not code internals.
