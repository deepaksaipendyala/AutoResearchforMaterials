# Tiny Materials Band Gap

This folder follows the Materials Autoresearch project format.

Start here:

```bash
materials-project validate .
materials-train .
```

Important files:

- `data/raw/` contains original input data. Do not edit these files during experiments.
- `specs/project.json` records the target property, task type, split, and metric.
- `specs/data_dictionary.csv` explains each column and its role.
- `specs/model_config.json` contains agent-tunable training settings.
- `agent/edit_scope.json` says which files the agent may change.
- `agent/research_brief.md` is the short project brief for the research agent.
- `agent/experiment_plan.md` tracks the next few experiment ideas.
- `agent/experiment_log.tsv` records model runs and results.
- `runs/` stores trained baselines and later experiments.
- `reports/` stores human-readable summaries.
