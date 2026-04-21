# Materials Autoresearch Project Format

This repository now has a standard project folder for materials-science machine learning work. The format is designed for PhD students, postdocs, and research teams who know their domain data but do not want to manage a complex ML codebase.

## Folder Layout

Every project should look like this:

```text
my_project/
  README.md
  data/
    raw/
      primary_data.csv
    processed/
  specs/
    project.json
    data_dictionary.csv
    model_config.json
  agent/
    edit_scope.json
    research_brief.md
    decision_log.md
    experiment_plan.md
    experiment_log.tsv
  runs/
  reports/
```

The setup command creates these files for you:

```bash
materials-project init my_project --data path/to/measurements.csv --target band_gap
materials-project validate my_project
materials-train my_project
```

## Primary Data Table

The first version expects one primary tabular file in CSV, TSV, or Parquet format.

Rules:

- One row means one material, sample, synthesis condition, calculation, or measurement.
- The first row must contain column names.
- The target property must be in one column.
- Units must be consistent within each column.
- Missing values should be blank, not text like `unknown` or `n/a`.
- Raw data goes in `data/raw/` and should not be edited during experiments.

Recommended columns:

| Role | Example names | Meaning |
| --- | --- | --- |
| Identifier | `material_id`, `sample_id`, `formula` | Keeps predictions traceable. |
| Target | `band_gap`, `formation_energy`, `conductivity` | Property to predict. |
| Numeric features | `temperature_K`, `pressure_GPa`, `density` | Inputs for the first baseline. |
| Processing features | `anneal_time_h`, `sinter_temperature_K`, `pH` | Synthesis, processing, or measurement conditions. |
| Categorical features | `space_group`, `synthesis_route`, `source` | Text or labels used as inputs. |
| Composition | `formula`, `composition` | Formula/composition information. |
| Structure reference | `structure_file`, `cif_file` | Path to CIF, POSCAR, XYZ, or similar file. |
| Group | `chemical_system`, `family`, `batch`, `study_id` | Used for harder train/test splits. |

## Supported Project Families

The setup command detects a project family from the target and columns. This is not a hard limit; it helps the agent choose reasonable next experiments.

| Family | Typical data | Examples |
| --- | --- | --- |
| `composition_property_prediction` | Formula/composition plus property | band gap, formation energy, modulus |
| `structure_property_prediction` | CIF/POSCAR/XYZ references plus property | elastic tensor, energy above hull |
| `process_property_prediction` | Composition plus processing conditions | conductivity after annealing, strength after heat treatment |
| `phase_or_label_classification` | Label target | stable/unstable, phase class, synthesis success |
| `candidate_screening` | Ranking target or desirability score | top candidates for experiments or DFT |

Current trainers:

- `tabular_mlp`: lightweight sanity-check baseline.
- `tabm`: Yandex Research TabM baseline for deep tabular materials learning.

Structure-aware, composition-featurized, and ranking-specific trainers are future experiment files so agents can add them without changing the project contract.

## Project Specification

`specs/project.json` records the scientific decisions that should not change accidentally:

- project name
- scientific goal
- primary table path
- target column
- task type
- primary metric
- split method
- raw-data protection rules

The research agent reads this file before every run.

## Data Dictionary

`specs/data_dictionary.csv` explains every column.

Required fields:

| Field | Meaning |
| --- | --- |
| `column` | Column name in the primary table. |
| `role` | Column role such as `id`, `target`, `feature`, `processing_feature`, `categorical_feature`, `composition`, `structure_ref`, `group`, or `metadata`. |
| `data_type` | Original data type detected from the file. |
| `units` | Physical units. Use `unitless` if there are no units. |
| `required` | `yes` if the column must be present for training. |
| `description` | Short scientific meaning of the column. |
| `missing_values` | Count of missing values at setup time. |
| `example` | Example value. |

The first baseline trainer uses roles listed in `specs/model_config.json`. By default those are `feature`, `processing_feature`, `categorical_feature`, and `composition`.

## Agent-Tunable Files

The agent should not have to edit the whole repository for every project. Each project gets a local edit boundary:

- `specs/model_config.json`: safe experiment settings such as hidden size, dropout, learning rate, batch size, seed, and feature roles.
- `agent/experiment_plan.md`: the agent's short plan for the next experiment.
- `agent/experiment_log.tsv`: append-only run log.

The agent must ask before changing:

- `specs/project.json`, because it defines the target, metric, split, and project family.
- `specs/data_dictionary.csv`, because column roles can add leakage or remove important science.
- `data/raw/*`, because raw measurements and calculations are the source of truth.

## TabM Benchmark Config

For the DataScribe-style benchmark goal, set `specs/model_config.json` to use TabM:

```json
{
  "trainer": "tabm",
  "tabm": {
    "arch_type": "tabm",
    "k": 32,
    "n_blocks": 3,
    "d_block": 256,
    "dropout": 0.1
  },
  "training": {
    "epochs": 300,
    "patience": 40,
    "batch_size": 256,
    "lr": 0.001,
    "weight_decay": 0.0001,
    "seed": 42
  }
}
```

Keep target, split, and metric aligned with the original benchmark paper when those details are available. The agent may tune TabM architecture and optimization settings, but it should ask before changing scientific evaluation decisions.

## Evaluation Patterns

Evaluation is a scientific decision, not just a coding detail. Choose it before training.

Regression tasks:

- Use when the target is a number, such as band gap, formation energy, modulus, or conductivity.
- Recommended primary metric: `mae`.
- Secondary metrics: `rmse`, `r2`.

Classification tasks:

- Use when the target is a label, such as stable/unstable, phase class, or pass/fail.
- Recommended primary metric: `balanced_accuracy` when classes are imbalanced.
- Secondary metrics: `accuracy`, `roc_auc` for binary labels.

Screening tasks:

- Use when the practical goal is ranking candidates for experiments or calculations.
- Recommended metrics: `top_k_recall`, `enrichment_factor`, `roc_auc`.
- The current baseline trains like regression; ranking metrics are the next planned backend feature.

Split choices:

- `random`: easiest first split. Good for early debugging.
- `group`: keeps families, batches, sources, or chemical systems together. Better for measuring generalization.
- `split_column`: use when the researcher already provides train/validation/test labels.
- Future: time-based split for projects with clear publication, synthesis, or measurement dates.

## Researcher Decisions

The setup command asks for decisions in plain language:

1. Project name: used in reports and run folders.
2. Scientific goal: tells the agent what improvements matter.
3. Target column: the property to predict.
4. Task type: number prediction, label prediction, or screening.
5. Primary metric: the score used to judge experiments.
6. Identifier column: keeps predictions traceable.
7. Structure column: enables future structure-aware models.
8. Group column: enables stricter evaluation.

If the researcher is unsure, start with random split and `mae` for regression or `balanced_accuracy` for classification. Then tighten the split after the first baseline works.
