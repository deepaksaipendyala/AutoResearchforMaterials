# Materials Autoresearch — Claude Code Context

## What this project is

AI-assisted machine learning framework for materials scientists (PhD students, postdocs, research groups). The goal is to make serious ML accessible to domain scientists without an ML engineering background.

**Core research idea:** Apply Yandex Research's TabM (2024 state-of-the-art tabular deep learning model) to the DataScribe benchmark datasets — the first time TabM has been applied to materials science. Establish novel baselines, then use an overnight agentic optimizer to improve them.

**Target users:** Materials scientists who know the science but not ML engineering. All questions and explanations should be in domain terms (properties, alloys, compositions) not code terms.

## Key papers

- **TabM:** Gorishniy et al., arXiv:2410.24210 (2024) — the model being benchmarked
- **DataScribe:** Attari et al., Digital Discovery (2025), DOI: 10.1039/D5DD00166H — the benchmark datasets
- **DataScribe data:** github.com/vahid2364/DataScribe_DeepTabularLearning | zenodo.org/doi/10.5281/zenodo.16396374

## Repository structure

```
autoresearch/
├── materials_project.py   # CLI: materials-project init/validate/summary
├── train_materials.py     # CLI: materials-train  (tabular_mlp + TabM trainers)
├── research_agent.py      # CLI: research-agent   (overnight Claude-driven optimizer)
├── setup_datascribe.py    # CLI: setup-datascribe (download + setup DataScribe projects)
├── program.md             # Agent instructions and scientific boundaries
├── docs/
│   ├── materials_project_format.md      # Full project folder spec
│   └── datascribe_tabm_benchmark.md     # TabM benchmark protocol
└── examples/
    ├── datascribe_manifest_template.csv  # Template for bulk project setup
    └── tiny_materials/                   # Fully configured smoke-test example
```

## CLI entry points (all via `uv run <command>`)

| Command | What it does |
|---------|-------------|
| `materials-project init <folder> --data <csv> --target <col>` | Create project folder, ask guided questions |
| `materials-project validate <folder>` | Check folder is ready to train |
| `materials-project summary <folder>` | Print project overview |
| `materials-train <folder>` | Train baseline (MLP or TabM) |
| `materials-train <folder> --run-id my-run` | Train with a specific run name |
| `research-agent <folder> --max-hours 8` | Start overnight agentic optimizer |
| `research-agent <folder> --dry-run` | See what Claude would propose without training |
| `setup-datascribe --download --output-dir projects/datascribe` | Download DataScribe datasets |
| `setup-datascribe --manifest manifest.csv` | Set up projects from a manifest |

## Project folder format (one folder per dataset/experiment)

```
my_project/
  specs/
    project.json          # Scientific contract — target, metric, split (DO NOT EDIT)
    data_dictionary.csv   # Column roles, units, descriptions
    model_config.json     # Model + training settings — AGENT EDITABLE
  agent/
    edit_scope.json       # What the agent can change without asking
    research_brief.md     # Plain-language project summary for the agent
    experiment_log.tsv    # All run results — open in Excel/Sheets
    experiment_plan.md    # Agent's reasoning log per iteration
    decision_log.md       # Record of scientific decisions
  data/raw/               # Original data — NEVER modified
  runs/<run_id>/          # Training outputs: metrics.json, summary.md, model.pt, predictions
```

## Agent boundaries (enforced by edit_scope.json)

**Agent can freely change:** `specs/model_config.json` — TabM hyperparameters (k, n_blocks, d_block, dropout, arch_type), training settings (lr, weight_decay, batch_size, epochs, patience, seed), `agent/experiment_plan.md`, `agent/experiment_log.tsv`

**Agent must ask before changing:** target column, evaluation metric, train/val/test split, column roles, anything in `data/raw/`

## The DataScribe datasets (actual files in the GitHub repo)

All five are High Entropy Alloy (HEA) / MPEA datasets:

| Project name | Alloy system | Target column | File |
|---|---|---|---|
| `atlas_hea_liquidus` | AlCuCrNbNiFeMo | `PROP LT (K)` | `datasets/ATLAS-HEADATA_Jonathan Frutschy/data/data_LIQUID_variable_temprange9_processed.csv` |
| `atlas_rhea_creep` | NbCrVWZr refractory | `Creep Merit` | `datasets/ATLAS-RHEADATA/input_data/v3/NbCrVWZr_data_stoic_creep_equil_v3.csv` |
| `birdshot_hea_hardness` | AlCoCrCuFeMnNiV | `Hardness, HV` | `datasets/BIRDSHOT-HEADATA/data/HTMDEC_MasterTable_Iterations_v3_processed.csv` |
| `birdshot_hea_v5_hardness` | AlCoCrCuFeMnNiV expanded | `Hardness, HV` | `datasets/BIRDSHOT-HEADATA-DeepGP-Alvi/data/HTMDEC_MasterTable_Iterations_v5_processed.csv` |
| `borgHEA_hardness` | MPEA literature | `PROPERTY: HV` | `datasets/BorgHEA-DATA/data/Borg_df_updated.csv` |

Note: `"Hardness, HV"` has a comma — pass it quoted in shell, it's handled correctly as a list element in subprocess calls.

## Training models

Two trainers are available in `specs/model_config.json`:
- `"trainer": "tabular_mlp"` — lightweight MLP, fast sanity check
- `"trainer": "tabm"` — TabM (main benchmark model)

Default TabM config for benchmarking:
```json
{
  "trainer": "tabm",
  "tabm": { "arch_type": "tabm", "k": 32, "n_blocks": 3, "d_block": 256, "dropout": 0.1 },
  "training": { "epochs": 300, "patience": 40, "batch_size": 256, "lr": 0.001, "weight_decay": 0.0001, "seed": 42 }
}
```

## Overnight agent (`research_agent.py`)

- Requires `ANTHROPIC_API_KEY` environment variable
- Default model: `claude-sonnet-4-6` (override with `--model claude-opus-4-7`)
- Proposes one hyperparameter change per iteration, runs training, logs result
- Rolls back config if result is not better than current best
- Logs reasoning to `agent/experiment_plan.md`, results to `agent/experiment_log.tsv`
- Safe to interrupt with Ctrl+C at any time

## Workstation setup (GPU training)

```bash
git pull
uv sync                          # needs CUDA 12.8+ (pytorch-cu128 index)
export ANTHROPIC_API_KEY=sk-ant-...
# Data files are gitignored — download or rsync from Mac:
uv run setup-datascribe --download --output-dir projects/datascribe
```

CUDA requirement: `pyproject.toml` uses the `pytorch-cu128` index (CUDA 12.8+). If the workstation has a different CUDA version, update the `[[tool.uv.index]]` URL in `pyproject.toml` to match (e.g. `cu121` for CUDA 12.1).

## Demo / smoke test

```bash
uv run materials-project validate examples/tiny_materials
uv run materials-train examples/tiny_materials --run-id smoke-test
cat examples/tiny_materials/runs/smoke-test/summary.md
```

## Important conventions

- One project folder per dataset — never mix datasets
- Always validate before training (`materials-project validate`)
- The agent is tuning hyperparameters, not making scientific decisions — the researcher owns the target, metric, and split
- All runs are logged to `agent/experiment_log.tsv` — this is the source of truth for results
- Raw data in `data/raw/` is never modified
- `projects/` and `**/runs/` are gitignored — results stay local, code travels via git
