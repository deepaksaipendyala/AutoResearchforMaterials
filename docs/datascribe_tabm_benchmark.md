# DataScribe TabM Benchmark Plan

Goal: establish TabM as a materials-science tabular baseline on the DataScribe datasets benchmarked by the Vahid Attari paper, then let autoresearch tune TabM overnight.

Primary public sources to use when assembling the benchmark tables:

- DataScribe Deep Tabular Learning GitHub: `https://github.com/vahid2364/DataScribe_DeepTabularLearning`
- Zenodo record: `https://doi.org/10.5281/zenodo.16396374`
- Paper: `https://doi.org/10.1039/D5DD00166H`

## What To Prepare

Create one row per dataset in a manifest:

```csv
dataset_name,data_path,target_column,task_type,primary_metric,id_column,group_column,notes
band_gap,path/to/band_gap.csv,band_gap_eV,regression,mae,material_id,chemical_system,
stability,path/to/stability.csv,stable,classification,balanced_accuracy,material_id,chemical_system,
```

Use `examples/datascribe_manifest_template.csv` as the starting file.

## One Dataset Workflow

Create a project:

```bash
python materials_project.py init projects/datascribe_band_gap \
  --data path/to/band_gap.csv \
  --target band_gap_eV \
  --task regression \
  --metric mae \
  --id-column material_id \
  --group-column chemical_system \
  --name "DataScribe Band Gap" \
  --goal "Benchmark TabM for band-gap prediction on the DataScribe dataset." \
  --yes
```

Validate:

```bash
python materials_project.py validate projects/datascribe_band_gap
```

Run the lightweight sanity baseline:

```bash
python train_materials.py projects/datascribe_band_gap --run-id mlp-sanity
```

Switch to TabM by editing `projects/datascribe_band_gap/specs/model_config.json`:

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

Then train:

```bash
python train_materials.py projects/datascribe_band_gap --run-id tabm-baseline
```

## Overnight Autoresearch Scope

The agent may tune:

- `specs/model_config.json`
- `agent/experiment_plan.md`
- `agent/experiment_log.tsv`

Good TabM parameters to tune first:

- `k`: 8, 16, 32, 64
- `n_blocks`: 2, 3, 4, 5
- `d_block`: 128, 256, 384, 512
- `dropout`: 0.0, 0.05, 0.1, 0.2
- `lr`: 0.0003, 0.001, 0.003
- `weight_decay`: 0.0, 0.0001, 0.001
- `batch_size`: 64, 128, 256, 512

The agent must ask before changing:

- target column
- train/validation/test split
- primary metric
- row filtering
- column roles in `specs/data_dictionary.csv`

## Reporting

For each dataset, report:

- dataset name
- target property
- task type
- split rule
- primary metric
- MLP sanity score
- first TabM score
- best overnight TabM score
- changed TabM settings
- caveats about split quality, leakage risk, and small-data uncertainty
