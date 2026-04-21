# Materials Autoresearch

**AI-assisted machine learning for materials science.**
Designed for PhD students, postdocs, and research groups who know the science and want results — not an ML engineering project.

The core workflow is:

1. Put your dataset (CSV, TSV, or Parquet) in a standard project folder.
2. Answer a few plain-language questions about your data (target property, units, split strategy).
3. Train a first baseline model to see where you stand.
4. Let an AI agent propose and test model improvements overnight — while you sleep.

The main research target is **TabM** — Yandex Research's 2024 state-of-the-art tabular deep learning model — applied to materials science datasets for the first time, using the **DataScribe** benchmark suite from Vahid Attari et al. (2025).

---

## Who this is for

- You have a materials dataset as a spreadsheet (CSV/Excel/TSV) or Parquet file.
- You want to train a serious machine learning model on it without writing PyTorch code.
- You want to know whether the model is actually better overnight, not after a week of manual tuning.
- You work in materials science, not software engineering.

---

## What you need before starting

- Python 3.10 or newer
- [`uv`](https://docs.astral.sh/uv/) (a fast Python package manager — install once with `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- An Anthropic API key for the overnight agent ([get one here](https://console.anthropic.com/))
- Your dataset as a CSV or similar file

No GPU is required for small datasets (< 10,000 rows), but a GPU makes training 10–50× faster.

---

## Install

```bash
git clone https://github.com/your-org/autoresearch.git
cd autoresearch
uv sync
```

That installs everything, including PyTorch, TabM, and the Anthropic SDK.

---

## The five-step workflow

### Step 1 — Create a project folder from your data

```bash
uv run materials-project init projects/my_project --data path/to/my_data.csv --target band_gap
```

This will ask you a few questions:

- **What is the target property?** The column you want to predict (e.g., `band_gap`, `formation_energy`, `yield_strength`).
- **Is it a number or a category?** Numbers → regression. Labels like "stable/unstable" → classification.
- **Which metric decides "better"?** MAE is recommended for physical properties (it has units you can interpret). Balanced accuracy for classification.
- **Random split or grouped by chemistry?** Grouping by chemical system gives a harder, more realistic test.
- **Which columns are inputs?** The tool auto-detects; you confirm or correct.

**Why does the tool ask these questions?**
These decisions define the entire experiment. The target and metric cannot change once the project is set up (the agent will ask your permission first if it thinks they should change). This protects you from accidentally comparing apples and oranges across runs.

### Step 2 — Validate the project

```bash
uv run materials-project validate projects/my_project
```

This checks that the data file is readable, the target column exists, the split will produce non-empty train/validation/test sets, and all required files are present.

**Why validate before training?**
Training on a bad split or missing column will silently produce meaningless numbers. Validate first.

### Step 3 — Train a first baseline

```bash
uv run materials-train projects/my_project
```

By default this runs a lightweight MLP to give you a number quickly. To use TabM (the main model), edit `projects/my_project/specs/model_config.json` and set `"trainer": "tabm"`, then run the same command.

Results appear in `projects/my_project/runs/<run_id>/`:
- `summary.md` — plain-language summary (MAE, split info, best epoch)
- `metrics.json` — all metrics as a machine-readable file
- `validation_predictions.csv` — row-by-row predictions
- `model.pt` — the saved model checkpoint

The experiment is also logged to `projects/my_project/agent/experiment_log.tsv` — a tab-separated file you can open in Excel.

### Step 4 — Let the agent tune overnight

```bash
export ANTHROPIC_API_KEY=sk-ant-...
uv run research-agent projects/my_project --max-hours 8
```

The agent will:
1. Read your project state, best result, and full experiment history.
2. Ask Claude to propose one experiment (a specific hyperparameter change) and explain why.
3. Apply the change to `model_config.json`.
4. Run training.
5. Check if the result improved.
6. Log the result and repeat.

It runs for 8 hours (or the number you set) and respects your project's boundaries: it will **never** change the target column, split, or primary metric without asking you first.

To watch it live:
```bash
tail -f projects/my_project/agent/experiment_plan.md
```

### Step 5 — Read the results

```bash
cat projects/my_project/agent/experiment_log.tsv
cat projects/my_project/runs/<best_run_id>/summary.md
```

Or open `experiment_log.tsv` in Excel/Sheets. Each row is one training run with the metric value, model settings, and notes.

---

## Data format

Your data must be a table where:
- **One row = one material, sample, measurement, or calculation**
- **One column = the property you want to predict** (your target)

Accepted formats: `.csv`, `.tsv`, `.parquet`

### Column types the tool understands

| Role | What it means | Examples |
|------|---------------|---------|
| `target` | The property to predict | `band_gap`, `formation_energy`, `stable` |
| `feature` | Numeric inputs the model uses | `density`, `electronegativity`, `lattice_a` |
| `processing_feature` | Synthesis or measurement conditions | `temperature`, `pressure`, `annealing_time` |
| `categorical_feature` | Non-numeric labels | `space_group`, `crystal_system`, `synthesis_route` |
| `composition` | Chemical formula column | `formula`, `composition` |
| `id` | Unique identifier (not a feature) | `material_id`, `sample_name`, `mp_id` |
| `group` | Chemistry family (for group split) | `chemical_system`, `batch_id`, `study_id` |
| `metadata` | Notes, DOI, source — never a feature | `reference`, `notes`, `doi` |

The tool auto-detects column roles from names and data types. You confirm or correct them during setup.

### What if my data has missing values?

The tool handles missing values automatically:
- Missing numeric values: filled with the column median.
- Missing categorical values: treated as a separate "missing" category.
- Rows where the **target is missing**: dropped (with a warning).

### What if my dataset is small (< 500 rows)?

This is common in materials science. The tool will:
- Warn you if any split (train/val/test) has fewer than 10 rows.
- Use random split by default (group split needs at least 10 groups).
- Work anyway, but results will have high uncertainty — the agent will note this in its reasoning.

---

## Project types

The setup command detects which kind of materials project you have:

| Type | Description | Example target |
|------|-------------|---------------|
| Composition-property | Predict from formula-level descriptors | band gap from chemical formula |
| Structure-property | Use CIF/POSCAR/XYZ structure files | bulk modulus from crystal structure |
| Process-property | Synthesis/processing conditions → outcome | yield strength from alloy composition + heat treatment |
| Phase classification | Stable/unstable, phase label | thermodynamic stability |
| Candidate screening | Rank materials for experiments | desirability score |

---

## The DataScribe benchmark (TabM on materials science)

This project is establishing **TabM as a new baseline for materials science tabular datasets** using the DataScribe benchmark from Vahid Attari et al. ([paper](https://doi.org/10.1039/D5DD00166H), [GitHub](https://github.com/vahid2364/DataScribe_DeepTabularLearning), [Zenodo](https://doi.org/10.5281/zenodo.16396374)).

**TabM** is the current state-of-the-art tabular deep learning model (Yandex Research, 2024). It has never been applied to these materials science datasets. This project changes that.

The DataScribe datasets are all **High Entropy Alloy (HEA) / Multi-Principal Element Alloy (MPEA)** datasets:

| Dataset | Alloy system | Target | Task |
|---------|-------------|--------|------|
| `atlas_hea_liquidus` | AlCuCrNbNiFeMo | Liquidus temperature (K) | Regression |
| `atlas_rhea_creep` | NbCrVWZr refractory | Creep merit index | Regression |
| `birdshot_hea_hardness` | AlCoCrCuFeMnNiV | Vickers hardness (HV) | Regression |
| `birdshot_hea_v5_hardness` | AlCoCrCuFeMnNiV (expanded) | Vickers hardness (HV) | Regression |
| `borgHEA_hardness` | MPEA literature compilation | Vickers hardness (HV) | Regression |

### Quick DataScribe setup

**Option A — Download datasets automatically:**
```bash
uv run setup-datascribe --download --output-dir projects/datascribe
uv run setup-datascribe --manifest projects/datascribe/datascribe_manifest.csv \
    --output-dir projects/datascribe/projects --train-baseline
```

**Option B — Use your own data with the manifest template:**
```bash
# Edit examples/datascribe_manifest_template.csv with your file paths
uv run setup-datascribe --manifest examples/datascribe_manifest_template.csv \
    --output-dir projects/datascribe
```

**Option C — Set up one dataset manually:**
```bash
uv run materials-project init projects/datascribe/band_gap \
  --data path/to/band_gap.csv \
  --target band_gap \
  --task regression \
  --metric mae \
  --id-column material_id \
  --group-column chemical_system \
  --name "DataScribe Band Gap" \
  --goal "Benchmark TabM for band-gap prediction on the DataScribe dataset." \
  --yes

uv run materials-project validate projects/datascribe/band_gap
uv run materials-train projects/datascribe/band_gap --run-id mlp-sanity
# Edit specs/model_config.json → set "trainer": "tabm"
uv run materials-train projects/datascribe/band_gap --run-id tabm-baseline
```

### TabM config for benchmarking

Open `projects/<dataset>/specs/model_config.json` and set:

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

---

## The agent — what it can and cannot change

### What the agent changes (without asking):
- `specs/model_config.json`: learning rate, batch size, model width (d_block), depth (n_blocks), dropout, the k parameter in TabM, number of epochs

### What the agent must ask you before changing:
- The target column
- The evaluation metric
- The train/validation/test split
- Column roles (what counts as an input feature)
- Anything in `data/raw/` (your original data is never touched)

This contract is stored in `agent/edit_scope.json` inside your project folder.

### Why this boundary matters

The agent is optimizing hyperparameters, not doing science. The scientific decisions (what to predict, how to measure it, what counts as an input) are yours. The tool makes this boundary explicit and enforces it.

---

## Project folder structure

Every project follows this layout:

```
my_project/
  README.md                   ← quick-start for this specific project
  data/
    raw/
      my_data.csv             ← original data, never modified
    processed/                ← for future preprocessed versions
  specs/
    project.json              ← scientific contract (target, metric, split)
    data_dictionary.csv       ← column roles, units, descriptions
    model_config.json         ← model + training settings (agent-tunable)
  agent/
    edit_scope.json           ← what the agent can change without asking
    research_brief.md         ← plain-language project summary for the agent
    decision_log.md           ← record of scientific decisions made
    experiment_plan.md        ← agent's reasoning log (updated each iteration)
    experiment_log.tsv        ← all run results (open in Excel)
  runs/
    <run_id>/
      summary.md              ← plain-language result
      metrics.json            ← all metrics
      model.pt                ← saved model checkpoint
      validation_predictions.csv
      test_predictions.csv
  reports/                    ← for your write-ups and figures
```

---

## Commands

| Command | What it does |
|---------|-------------|
| `uv run materials-project init <folder> --data <file> --target <column>` | Create a project folder and answer setup questions |
| `uv run materials-project validate <folder>` | Check the folder is ready to train |
| `uv run materials-project summary <folder>` | Print a project overview |
| `uv run materials-train <folder>` | Train the baseline |
| `uv run materials-train <folder> --run-id my-run` | Train with a specific run name |
| `uv run materials-train <folder> --cpu` | Force CPU training |
| `uv run research-agent <folder>` | Start the overnight agent |
| `uv run research-agent <folder> --max-hours 8` | Agent runs for 8 hours |
| `uv run research-agent <folder> --dry-run` | See what the agent would propose without training |
| `uv run setup-datascribe --download` | Download DataScribe datasets from GitHub |
| `uv run setup-datascribe --manifest manifest.csv` | Set up projects from a manifest |

---

## Evaluation

**Regression** (predicting a number like band gap or formation energy):
- Primary metric: **MAE** — mean absolute error in the target units. Easier to interpret than RMSE.
- Secondary: RMSE, R²

**Classification** (predicting a label like stable/unstable):
- Primary metric: **balanced accuracy** — correct even when classes are imbalanced.
- Secondary: accuracy, ROC-AUC

**Split options:**
- `random` — rows are randomly shuffled into train/val/test. Easy first test.
- `group` — groups of materials (e.g. by chemical system) are kept together. Tests whether the model generalises to new chemistry. Harder and more realistic.

---

## Contributing

This project is open source (MIT license). We welcome:
- New DataScribe project examples
- Additional tabular model trainers (XGBoost, LightGBM, CatBoost, etc.)
- Composition and structure featurizers (pymatgen, DScribe integration)
- Improvements to the question-asking flow for domain scientists
- Results and comparisons across materials datasets

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add a new dataset example or trainer.

---

## Current status

**Implemented and tested:**
- Project folder format (v0.2)
- Guided interactive project setup with domain-scientist-friendly questions
- Project validation
- Data dictionary generation (column roles, types, units)
- Project family and modality detection
- Tabular MLP baseline (PyTorch)
- TabM baseline (Yandex Research's 2024 model)
- Regression and classification metrics
- Run folders with model checkpoints, metrics, predictions, and summaries
- Agentic overnight optimizer (`research-agent`) using Claude

**Planned next:**
- Composition featurization (Magpie descriptors via pymatgen)
- Structure featurization (SOAP, ACSF via DScribe or matminer)
- Ranking/screening metrics (top-k recall, enrichment factor)
- Additional tabular models (XGBoost, CatBoost, LightGBM)
- Multi-dataset comparison dashboard

---

## Paper references

- **TabM**: Gorishniy, Y. et al. *TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling.* arXiv 2410.24210 (2024)
- **DataScribe**: Attari, V. et al. *DataScribe: A Benchmark for Deep Tabular Learning in Materials Science.* Digital Discovery (2025). DOI: [10.1039/D5DD00166H](https://doi.org/10.1039/D5DD00166H)
- **DataScribe data**: [GitHub](https://github.com/vahid2364/DataScribe_DeepTabularLearning) | [Zenodo](https://doi.org/10.5281/zenodo.16396374)

---

## License

MIT
