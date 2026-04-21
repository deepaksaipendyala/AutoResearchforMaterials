#!/usr/bin/env python3
"""
Create and validate Materials Autoresearch project folders.

The goal of this file is to keep the researcher-facing contract simple:
one project folder, one primary data table, one machine-readable project
specification, and clear notes for the research agent.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "materials-autoresearch/v0.2"
SUPPORTED_TABLE_EXTENSIONS = {".csv", ".tsv", ".parquet"}
STRUCTURE_EXTENSIONS = {".cif", ".xyz", ".vasp", ".poscar", ".contcar", ".pdb"}
DATA_DICTIONARY_FIELDS = [
    "column",
    "role",
    "data_type",
    "units",
    "required",
    "description",
    "missing_values",
    "example",
]

TRAINING_CONFIG_DEFAULTS = {
    "trainer": "tabular_mlp",
    "model": {
        "hidden_dim": 128,
        "dropout": 0.05,
    },
    "tabm": {
        "arch_type": "tabm",
        "k": 32,
        "n_blocks": 3,
        "d_block": 256,
        "dropout": 0.1,
    },
    "training": {
        "epochs": 200,
        "patience": 30,
        "batch_size": 64,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "seed": 42,
    },
    "feature_selection": {
        "numeric_roles": ["feature", "processing_feature"],
        "categorical_roles": ["categorical_feature", "composition"],
        "excluded_roles": ["id", "target", "metadata", "structure_ref", "group"],
    },
}


def load_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "pandas is required for materials project setup. Run `uv sync` first."
        ) from exc
    return pd


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_table(path: Path):
    pd = load_pandas()
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported data file extension: {path.suffix}")


def project_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def relpath(path: Path, base: Path) -> str:
    return path.resolve().relative_to(base.resolve()).as_posix()


def ensure_directories(root: Path) -> None:
    for relative in [
        "data/raw",
        "data/processed",
        "specs",
        "agent",
        "runs",
        "reports",
    ]:
        (root / relative).mkdir(parents=True, exist_ok=True)


def find_primary_tables(root: Path) -> list[Path]:
    search_roots = [root / "data" / "raw", root]
    seen: set[Path] = set()
    tables: list[Path] = []
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for path in sorted(search_root.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_TABLE_EXTENSIONS:
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    tables.append(path)
    return tables


def infer_column(columns: list[str], candidates: list[str]) -> str | None:
    lower_to_original = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    for column in columns:
        lowered = column.lower()
        if any(candidate.lower() in lowered for candidate in candidates):
            return column
    return None


def profile_dataframe(df) -> list[dict[str, Any]]:
    profile: list[dict[str, Any]] = []
    for column in df.columns:
        series = df[column]
        non_null = series.dropna()
        example = "" if non_null.empty else str(non_null.iloc[0])
        profile.append(
            {
                "column": str(column),
                "dtype": str(series.dtype),
                "missing": int(series.isna().sum()),
                "unique": int(series.nunique(dropna=True)),
                "example": example[:120],
            }
        )
    return profile


def print_data_profile(profile: list[dict[str, Any]], row_count: int) -> None:
    print(f"\nI found {row_count:,} rows and {len(profile)} columns in the primary data table.")
    print("Columns:")
    for item in profile:
        print(
            f"  - {item['column']}: {item['dtype']}, "
            f"{item['unique']} unique, {item['missing']} missing, example={item['example']!r}"
        )


def is_interactive(args: argparse.Namespace) -> bool:
    return not args.yes and sys.stdin.isatty()


def ask_text(prompt: str, default: str | None = None, required: bool = True) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        answer = input(f"{prompt}{suffix}: ").strip()
        if answer:
            return answer
        if default is not None:
            return default
        if not required:
            return ""
        print("Please enter a value.")


def ask_choice(
    label: str,
    why: str,
    choices: list[str],
    default: str,
) -> str:
    print(f"\nDecision needed: {label}")
    print(f"Why this matters: {why}")
    for index, choice in enumerate(choices, start=1):
        marker = " (recommended)" if choice == default else ""
        print(f"  {index}. {choice}{marker}")
    while True:
        answer = input(f"Choose 1-{len(choices)} [{default}]: ").strip()
        if not answer:
            return default
        if answer in choices:
            return answer
        if answer.isdigit() and 1 <= int(answer) <= len(choices):
            return choices[int(answer) - 1]
        print("Please choose one of the listed options.")


def infer_task_type(df, target_column: str) -> str:
    series = df[target_column]
    numeric = load_pandas().api.types.is_numeric_dtype(series)
    unique = series.nunique(dropna=True)
    if numeric and unique > 12:
        return "regression"
    return "classification"


def default_metric(task_type: str) -> str:
    if task_type == "regression":
        return "mae"
    if task_type == "classification":
        return "balanced_accuracy"
    if task_type == "screening":
        return "top_k_recall"
    return "mae"


def secondary_metrics(task_type: str, primary_metric: str) -> list[str]:
    options = {
        "regression": ["mae", "rmse", "r2"],
        "classification": ["accuracy", "balanced_accuracy", "roc_auc"],
        "screening": ["top_k_recall", "enrichment_factor", "roc_auc"],
    }.get(task_type, [primary_metric])
    return [metric for metric in options if metric != primary_metric]


def role_for_column(
    column: str,
    dtype: str,
    decisions: dict[str, str | None],
) -> str:
    lowered = column.lower()
    if column == decisions.get("target_column"):
        return "target"
    if column == decisions.get("id_column"):
        return "id"
    if column == decisions.get("structure_column"):
        return "structure_ref"
    if column == decisions.get("group_column"):
        return "group"
    if any(token in lowered for token in ["formula", "composition", "chemical_formula"]):
        return "composition"
    if any(token in lowered for token in ["temperature", "pressure", "time", "ph", "rate", "anneal", "sinter"]):
        return "processing_feature"
    if any(token in lowered for token in ["notes", "comment", "doi", "paper", "source", "reference"]):
        return "metadata"
    if dtype.startswith(("int", "float", "uint")):
        return "feature"
    return "categorical_feature"


def write_data_dictionary(
    path: Path,
    profile: list[dict[str, Any]],
    decisions: dict[str, str | None],
    force: bool,
) -> None:
    if path.exists() and not force:
        print(f"Kept existing {path}")
        return
    rows = []
    for item in profile:
        role = role_for_column(item["column"], item["dtype"], decisions)
        rows.append(
            {
                "column": item["column"],
                "role": role,
                "data_type": item["dtype"],
                "units": "TODO" if role in {"target", "feature"} else "",
                "required": "yes" if role in {"target", "id"} else "no",
                "description": describe_column_role(role),
                "missing_values": str(item["missing"]),
                "example": item["example"],
            }
        )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DATA_DICTIONARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def describe_column_role(role: str) -> str:
    return {
        "id": "Unique material, sample, or calculation identifier.",
        "target": "Property the model should predict.",
        "feature": "Numeric input used by the first baseline model.",
        "processing_feature": "Processing, synthesis, or measurement condition used as an input.",
        "categorical_feature": "Text/category input. The baseline one-hot encodes it.",
        "composition": "Composition or formula input. The baseline one-hot encodes it until composition featurizers are added.",
        "structure_ref": "Path or identifier for a structure file such as CIF or POSCAR.",
        "group": "Family/batch/source column used to make a harder evaluation split.",
    }.get(role, "Metadata kept for traceability.")


def unique_destination(raw_dir: Path, suffix: str, force: bool) -> Path:
    destination = raw_dir / f"primary_data{suffix}"
    if force or not destination.exists():
        return destination
    counter = 2
    while True:
        candidate = raw_dir / f"primary_data_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def copy_primary_data(source: Path, root: Path, force: bool) -> Path:
    source = source.expanduser().resolve()
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if root in source.parents and source.suffix.lower() in SUPPORTED_TABLE_EXTENSIONS:
        return source
    destination = unique_destination(raw_dir, source.suffix.lower(), force=force)
    shutil.copy2(source, destination)
    print(f"Copied data table to {destination}")
    return destination


def detect_structure_files(root: Path) -> list[str]:
    files: list[str] = []
    for path in sorted((root / "data").rglob("*")) if (root / "data").exists() else []:
        if path.is_file() and path.suffix.lower() in STRUCTURE_EXTENSIONS:
            files.append(relpath(path, root))
    return files


def detect_data_modalities(root: Path, df, decisions: dict[str, str | None]) -> list[str]:
    columns = [str(column).lower() for column in df.columns]
    column_tokens = [set(re.split(r"[^a-z0-9]+", column)) for column in columns]
    modalities = ["tabular"]
    if decisions.get("structure_column") or detect_structure_files(root):
        modalities.append("structure")
    if any("formula" in column or "composition" in column for column in columns):
        modalities.append("composition")
    if any(token in column for column in columns for token in ["temperature", "pressure", "anneal", "sinter", "time"]):
        modalities.append("processing")
    if any("spectrum" in column or {"xrd", "raman", "ftir"} & tokens for column, tokens in zip(columns, column_tokens)):
        modalities.append("spectra")
    if any("image" in column or "micrograph" in column or {"sem", "tem"} & tokens for column, tokens in zip(columns, column_tokens)):
        modalities.append("images")
    return modalities


def infer_project_family(task_type: str, modalities: list[str]) -> str:
    if task_type == "screening":
        return "candidate_screening"
    if task_type == "classification":
        return "phase_or_label_classification"
    if "processing" in modalities:
        return "process_property_prediction"
    if "structure" in modalities:
        return "structure_property_prediction"
    if "composition" in modalities:
        return "composition_property_prediction"
    return "tabular_property_prediction"


def write_json(path: Path, payload: dict[str, Any], force: bool) -> None:
    if path.exists() and not force:
        print(f"Kept existing {path}")
        return
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {path}")


def write_text(path: Path, text: str, force: bool) -> None:
    if path.exists() and not force:
        print(f"Kept existing {path}")
        return
    path.write_text(text)
    print(f"Wrote {path}")


def build_project_spec(
    root: Path,
    args: argparse.Namespace,
    data_path: Path,
    df,
    decisions: dict[str, str | None],
) -> dict[str, Any]:
    task_type = str(decisions["task_type"])
    primary_metric = str(decisions["primary_metric"])
    group_column = decisions.get("group_column") or None
    split_method = "group" if group_column else "random"
    modalities = detect_data_modalities(root, df, decisions)
    return {
        "schema_version": SCHEMA_VERSION,
        "project_name": decisions["project_name"],
        "created_at": utc_now(),
        "scientific_goal": decisions["scientific_goal"],
        "project_family": infer_project_family(task_type, modalities),
        "data": {
            "primary_table": relpath(data_path, root),
            "row_count_at_setup": int(len(df)),
            "target_column": decisions["target_column"],
            "id_column": decisions.get("id_column") or None,
            "structure_column": decisions.get("structure_column") or None,
            "group_column": group_column,
            "modalities": modalities,
            "feature_policy": "Use roles listed in specs/model_config.json. Defaults include feature, processing_feature, categorical_feature, and composition.",
            "structure_files_detected": detect_structure_files(root),
        },
        "task": {
            "type": task_type,
            "prediction_target": decisions["target_column"],
        },
        "evaluation": {
            "primary_metric": primary_metric,
            "secondary_metrics": secondary_metrics(task_type, primary_metric),
            "split": {
                "method": split_method,
                "train_fraction": 0.8,
                "validation_fraction": 0.1,
                "test_fraction": 0.1,
                "group_column": group_column,
                "random_seed": 42,
            },
        },
        "research_rules": {
            "do_not_modify_raw_data": True,
            "record_every_experiment": True,
            "prefer_simple_models_until_baseline_is_strong": True,
            "ask_before_changing_target_or_evaluation_split": True,
            "agent_tunable_files": [
                "specs/model_config.json",
                "agent/experiment_plan.md",
                "agent/experiment_log.tsv",
            ],
            "researcher_decision_files": [
                "specs/project.json",
                "specs/data_dictionary.csv",
            ],
        },
    }


def research_brief(spec: dict[str, Any]) -> str:
    data = spec["data"]
    evaluation = spec["evaluation"]
    modalities = ", ".join(data.get("modalities", ["tabular"]))
    return f"""# Research Brief

Project: {spec["project_name"]}

Scientific goal:
{spec["scientific_goal"]}

Project family:
{spec.get("project_family", "materials_project")}

Data modalities:
{modalities}

Data to use:
- Primary table: `{data["primary_table"]}`
- Target column: `{data["target_column"]}`
- ID column: `{data.get("id_column") or "not set"}`
- Structure column: `{data.get("structure_column") or "not set"}`
- Group column: `{data.get("group_column") or "not set"}`

Evaluation rule:
- Task type: `{spec["task"]["type"]}`
- Primary metric: `{evaluation["primary_metric"]}`
- Split method: `{evaluation["split"]["method"]}`

Agent instructions:
1. Validate the project before training.
2. Do not edit files in `data/raw/`.
3. Use `agent/edit_scope.json` to decide which files can be changed without asking.
4. Treat the primary metric and split method as scientific decisions. Ask before changing them.
5. Log every run in `agent/experiment_log.tsv`.
6. Explain results in terms of material-property prediction, not code internals.
"""


def project_readme(spec: dict[str, Any]) -> str:
    return f"""# {spec["project_name"]}

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
"""


def decision_log(spec: dict[str, Any]) -> str:
    data = spec["data"]
    evaluation = spec["evaluation"]
    return f"""# Decision Log

Created: {spec["created_at"]}

Initial decisions:

- Project name: {spec["project_name"]}
- Scientific goal: {spec["scientific_goal"]}
- Target column: {data["target_column"]}
- Task type: {spec["task"]["type"]}
- Primary metric: {evaluation["primary_metric"]}
- Split method: {evaluation["split"]["method"]}
- Group column: {data.get("group_column") or "not set"}

Use this file to record later scientific decisions, especially changes to target, split, or metric.
"""


def experiment_log_header() -> str:
    return "run_id\tstarted_at\tstatus\tprimary_metric\tvalue\tsplit\tmodel\tnotes\n"


def model_config() -> dict[str, Any]:
    return json.loads(json.dumps(TRAINING_CONFIG_DEFAULTS))


def edit_scope(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": spec["schema_version"],
        "purpose": "Defines which project files an autonomous agent may change during experiments.",
        "agent_may_edit_without_asking": [
            {
                "path": "specs/model_config.json",
                "why": "Choose tabular_mlp or tabm and tune model size, training time, learning rate, feature roles, and similar experiment settings.",
            },
            {
                "path": "agent/experiment_plan.md",
                "why": "Keep a short plan for the next experiment.",
            },
            {
                "path": "agent/experiment_log.tsv",
                "why": "Record every completed, failed, or skipped run.",
            },
        ],
        "agent_may_edit_with_researcher_approval": [
            {
                "path": "specs/data_dictionary.csv",
                "why": "Changing column roles can add or remove scientific information from the model.",
            },
            {
                "path": "specs/project.json",
                "why": "This file defines the target, metric, split, and scientific contract.",
            },
            {
                "path": "train_materials.py",
                "why": "Shared trainer changes affect all projects, not just this one.",
            },
        ],
        "agent_must_not_edit": [
            "data/raw/*",
            "runs/*/metrics.json",
            "runs/*/validation_predictions.csv",
            "runs/*/test_predictions.csv",
        ],
        "ask_before": [
            "changing the target column",
            "changing the task type",
            "changing the primary metric",
            "changing the train/validation/test split",
            "dropping rows for scientific rather than technical reasons",
            "using a target-derived or leakage-prone column as a feature",
        ],
    }


def experiment_plan(spec: dict[str, Any]) -> str:
    modalities = ", ".join(spec["data"].get("modalities", ["tabular"]))
    return f"""# Experiment Plan

Project family: {spec.get("project_family", "materials_project")}
Data modalities detected: {modalities}

Current baseline:
- Trainer: `tabular_mlp`
- Config file: `specs/model_config.json`
- Primary metric: `{spec["evaluation"]["primary_metric"]}`
- Split method: `{spec["evaluation"]["split"]["method"]}`

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
"""


def choose_decisions(args: argparse.Namespace, df) -> dict[str, str | None]:
    columns = [str(column) for column in df.columns]
    profile = profile_dataframe(df)
    print_data_profile(profile, len(df))

    target_default = args.target or infer_column(
        columns,
        [
            "target",
            "property",
            "band_gap",
            "formation_energy",
            "conductivity",
            "yield_strength",
            "capacity",
            "label",
        ],
    )
    id_default = args.id_column or infer_column(
        columns, ["material_id", "sample_id", "calculation_id", "id", "formula", "composition"]
    )
    structure_default = args.structure_column or infer_column(
        columns, ["structure_file", "cif_file", "cif", "poscar", "structure"]
    )
    group_default = args.group_column or infer_column(
        columns, ["family", "composition_family", "chemical_system", "prototype", "batch", "source", "study_id"]
    )

    if is_interactive(args):
        print("\nDecision needed: project name")
        print("Why this matters: this name appears in reports and run folders.")
        project_name = ask_text("Project name", args.name or Path(args.project_dir).name)

        print("\nDecision needed: scientific goal")
        print("Why this matters: the agent uses this to decide which experiments are relevant.")
        scientific_goal = ask_text(
            "In one sentence, what should the model help you predict or discover",
            args.goal or "Predict the target material property from the provided data.",
        )

        print("\nDecision needed: target column")
        print("Why this matters: this is the property the model will learn to predict.")
        target_column = ask_text("Target column", target_default)
        if target_column not in columns:
            raise SystemExit(f"Target column {target_column!r} was not found in the data table.")

        task_default = args.task or infer_task_type(df, target_column)
        task_type = ask_choice(
            "task type",
            "Regression predicts a number. Classification predicts a label or class.",
            ["regression", "classification", "screening"],
            task_default,
        )

        metric_choices = {
            "regression": ["mae", "rmse", "r2"],
            "classification": ["balanced_accuracy", "accuracy", "roc_auc"],
            "screening": ["top_k_recall", "enrichment_factor", "roc_auc"],
        }[task_type]
        primary_metric = ask_choice(
            "primary metric",
            "This single score decides whether a later experiment is better than the baseline.",
            metric_choices,
            args.metric or default_metric(task_type),
        )

        print("\nDecision needed: identifier column")
        print("Why this matters: this keeps predictions traceable to a material, sample, or calculation.")
        id_column = ask_text("Identifier column, or leave blank if none", id_default, required=False)

        print("\nDecision needed: structure column")
        print("Why this matters: structure files such as CIF or POSCAR may be used by later models.")
        structure_column = ask_text("Structure-file column, or leave blank if none", structure_default, required=False)

        print("\nDecision needed: group column")
        print("Why this matters: grouping by chemistry family, batch, or source tests generalization more honestly.")
        group_column = ask_text("Group column for harder splitting, or leave blank for random split", group_default, required=False)
    else:
        project_name = args.name or Path(args.project_dir).name
        scientific_goal = args.goal or "Predict the target material property from the provided data."
        target_column = target_default
        if not target_column:
            raise SystemExit(
                "Could not infer the target column. Re-run with `--target COLUMN` or use interactive setup."
            )
        if target_column not in columns:
            raise SystemExit(f"Target column {target_column!r} was not found in the data table.")
        task_type = args.task or infer_task_type(df, target_column)
        primary_metric = args.metric or default_metric(task_type)
        id_column = id_default
        structure_column = structure_default
        group_column = group_default

    for optional_name, optional_value in [
        ("id column", id_column),
        ("structure column", structure_column),
        ("group column", group_column),
    ]:
        if optional_value and optional_value not in columns:
            raise SystemExit(f"{optional_name.title()} {optional_value!r} was not found in the data table.")

    return {
        "project_name": project_name,
        "scientific_goal": scientific_goal,
        "target_column": target_column,
        "task_type": task_type,
        "primary_metric": primary_metric,
        "id_column": id_column or None,
        "structure_column": structure_column or None,
        "group_column": group_column or None,
    }


def init_project(args: argparse.Namespace) -> int:
    root = project_path(args.project_dir)
    root.mkdir(parents=True, exist_ok=True)
    ensure_directories(root)

    if args.data:
        source_data = Path(args.data)
    else:
        tables = find_primary_tables(root)
        if tables:
            source_data = tables[0]
            print(f"Using detected primary data table: {source_data}")
        elif is_interactive(args):
            print("\nDecision needed: primary data table")
            print("Why this matters: the model needs one table with one row per material, sample, or calculation.")
            source_data = Path(ask_text("Path to CSV, TSV, or Parquet data file"))
        else:
            raise SystemExit("No data table found. Re-run with `--data path/to/table.csv`.")

    if not source_data.exists():
        raise SystemExit(f"Data file does not exist: {source_data}")
    if source_data.suffix.lower() not in SUPPORTED_TABLE_EXTENSIONS:
        raise SystemExit(f"Unsupported data file: {source_data}")

    data_path = copy_primary_data(source_data, root, force=args.force)
    df = read_table(data_path)
    decisions = choose_decisions(args, df)
    spec = build_project_spec(root, args, data_path, df, decisions)

    write_json(root / "specs" / "project.json", spec, force=args.force)
    write_json(root / "specs" / "model_config.json", model_config(), force=args.force)
    write_data_dictionary(
        root / "specs" / "data_dictionary.csv",
        profile_dataframe(df),
        decisions,
        force=args.force,
    )
    write_json(root / "agent" / "edit_scope.json", edit_scope(spec), force=args.force)
    write_text(root / "agent" / "research_brief.md", research_brief(spec), force=args.force)
    write_text(root / "agent" / "decision_log.md", decision_log(spec), force=args.force)
    write_text(root / "agent" / "experiment_plan.md", experiment_plan(spec), force=args.force)
    write_text(root / "agent" / "experiment_log.tsv", experiment_log_header(), force=args.force)
    write_text(root / "README.md", project_readme(spec), force=args.force)

    print("\nProject folder is ready.")
    print(f"Next: `materials-project validate {root}`")
    return 0


def load_project_spec(root: Path) -> dict[str, Any]:
    spec_path = root / "specs" / "project.json"
    if not spec_path.exists():
        raise FileNotFoundError("Missing specs/project.json")
    return json.loads(spec_path.read_text())


def load_data_dictionary(root: Path) -> list[dict[str, str]]:
    path = root / "specs" / "data_dictionary.csv"
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def feature_rows(data_dictionary: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in data_dictionary
        if row.get("role") in {"feature", "processing_feature", "categorical_feature", "composition"}
    ]


def validate_project(args: argparse.Namespace) -> int:
    root = project_path(args.project_dir)
    errors: list[str] = []
    warnings: list[str] = []

    try:
        spec = load_project_spec(root)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    if spec.get("schema_version") != SCHEMA_VERSION:
        warnings.append(f"Schema version is {spec.get('schema_version')!r}; expected {SCHEMA_VERSION!r}.")

    data = spec.get("data", {})
    table_path = root / str(data.get("primary_table", ""))
    if not table_path.exists():
        errors.append(f"Primary table is missing: {table_path}")
        df = None
    else:
        try:
            df = read_table(table_path)
        except Exception as exc:
            errors.append(f"Could not read primary table: {exc}")
            df = None

    target_column = data.get("target_column")
    if df is not None:
        columns = set(map(str, df.columns))
        if target_column not in columns:
            errors.append(f"Target column {target_column!r} is not in the primary table.")
        elif df[target_column].isna().any():
            warnings.append(f"Target column {target_column!r} has missing values.")
        if data.get("id_column") and data["id_column"] not in columns:
            errors.append(f"ID column {data['id_column']!r} is not in the primary table.")
        if data.get("structure_column") and data["structure_column"] not in columns:
            errors.append(f"Structure column {data['structure_column']!r} is not in the primary table.")
        if data.get("group_column") and data["group_column"] not in columns:
            errors.append(f"Group column {data['group_column']!r} is not in the primary table.")
        if len(df) < 30:
            warnings.append("The table has fewer than 30 rows. Training can run, but evaluation will be noisy.")

    data_dictionary = load_data_dictionary(root)
    if not data_dictionary:
        warnings.append("Missing specs/data_dictionary.csv. Training can infer features, but the project is less clear.")
    else:
        dictionary_columns = {row.get("column") for row in data_dictionary}
        if df is not None:
            missing_from_dictionary = [column for column in map(str, df.columns) if column not in dictionary_columns]
            if missing_from_dictionary:
                warnings.append(
                    "These columns are not documented in specs/data_dictionary.csv: "
                    + ", ".join(missing_from_dictionary)
                )
        if not feature_rows(data_dictionary):
            errors.append(
                "No columns are marked as feature, processing_feature, categorical_feature, "
                "or composition in specs/data_dictionary.csv."
            )

    model_config_path = root / "specs" / "model_config.json"
    if not model_config_path.exists():
        warnings.append("Missing specs/model_config.json. Training will use built-in defaults.")
    else:
        try:
            config = json.loads(model_config_path.read_text())
            if config.get("trainer") not in {"tabular_mlp", "tabm"}:
                warnings.append(
                    f"Unknown trainer {config.get('trainer')!r}; current baseline supports `tabular_mlp` and `tabm`."
                )
        except Exception as exc:
            errors.append(f"Could not parse specs/model_config.json: {exc}")

    if not (root / "agent" / "edit_scope.json").exists():
        warnings.append("Missing agent/edit_scope.json. The agent has no project-specific edit boundary.")

    evaluation = spec.get("evaluation", {})
    if not evaluation.get("primary_metric"):
        errors.append("Missing evaluation.primary_metric in specs/project.json.")
    split = evaluation.get("split", {})
    if split.get("method") == "group" and not split.get("group_column"):
        errors.append("Group split was selected, but no group_column is set.")
    if split.get("method") == "random":
        total = sum(float(split.get(key, 0)) for key in ["train_fraction", "validation_fraction", "test_fraction"])
        if abs(total - 1.0) > 1e-6:
            errors.append("Random split fractions must add up to 1.0.")

    if errors:
        print("Project validation failed.")
        for error in errors:
            print(f"ERROR: {error}")
    else:
        print("Project validation passed.")

    for warning in warnings:
        print(f"WARNING: {warning}")

    if not errors:
        print(f"Next: `materials-train {root}`")
    return 1 if errors else 0


def summarize_project(args: argparse.Namespace) -> int:
    root = project_path(args.project_dir)
    spec = load_project_spec(root)
    data = spec["data"]
    evaluation = spec["evaluation"]
    data_dictionary = load_data_dictionary(root)
    features = feature_rows(data_dictionary)

    print(f"Project: {spec['project_name']}")
    print(f"Goal: {spec['scientific_goal']}")
    print(f"Primary table: {data['primary_table']}")
    print(f"Target: {data['target_column']}")
    print(f"Project family: {spec.get('project_family', 'materials_project')}")
    print(f"Modalities: {', '.join(data.get('modalities', ['tabular']))}")
    print(f"Task: {spec['task']['type']}")
    print(f"Metric: {evaluation['primary_metric']}")
    print(f"Split: {evaluation['split']['method']}")
    print(f"Features marked for baseline training: {len(features)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materials Autoresearch project setup")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="Create or complete a materials project folder")
    init.add_argument("project_dir", help="Project folder to create or update")
    init.add_argument("--data", help="CSV, TSV, or Parquet table with one row per material/sample/calculation")
    init.add_argument("--name", help="Human-readable project name")
    init.add_argument("--goal", help="One-sentence scientific goal")
    init.add_argument("--target", help="Column the model should predict")
    init.add_argument("--task", choices=["regression", "classification", "screening"], help="Prediction task type")
    init.add_argument("--metric", help="Primary evaluation metric")
    init.add_argument("--id-column", help="Identifier column")
    init.add_argument("--structure-column", help="Column containing structure-file paths or structure IDs")
    init.add_argument("--group-column", help="Column used for grouped train/validation/test split")
    init.add_argument("--yes", action="store_true", help="Accept inferred choices and do not ask questions")
    init.add_argument("--force", action="store_true", help="Overwrite generated project files")
    init.set_defaults(func=init_project)

    validate = subparsers.add_parser("validate", help="Check that a project folder is ready to train")
    validate.add_argument("project_dir", help="Project folder")
    validate.set_defaults(func=validate_project)

    summary = subparsers.add_parser("summary", help="Print a short project summary")
    summary.add_argument("project_dir", help="Project folder")
    summary.set_defaults(func=summarize_project)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
