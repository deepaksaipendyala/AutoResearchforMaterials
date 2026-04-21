#!/usr/bin/env python3
"""
Set up Materials Autoresearch project folders for DataScribe datasets.

Two modes:

1. From a manifest CSV (recommended):
   uv run setup-datascribe --manifest examples/datascribe_manifest_template.csv

2. Attempt to download DataScribe datasets from GitHub (requires internet):
   uv run setup-datascribe --download --output-dir projects/datascribe

The manifest CSV must have these columns:
  dataset_name, data_path, target_column, task_type, primary_metric
Optional columns:
  id_column, group_column, split_method, train_fraction, notes

After running, each dataset gets its own project folder. You can then train
baselines in one shot:
  uv run materials-train projects/datascribe/<dataset_name>

Or run all of them in sequence:
  uv run setup-datascribe --manifest my_manifest.csv --train-baseline

Full DataScribe benchmark workflow:
  1. uv run setup-datascribe --manifest examples/datascribe_manifest_template.csv
  2. uv run setup-datascribe --manifest examples/datascribe_manifest_template.csv --train-baseline
  3. uv run research-agent projects/datascribe/<dataset> --max-hours 8
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Known DataScribe datasets from the Vahid Attari paper
# (vahid2364/DataScribe_DeepTabularLearning on GitHub)
# These are the actual datasets benchmarked in the paper DOI: 10.1039/D5DD00166H
#
# All five datasets are High Entropy Alloy (HEA) / Multi-Principal Element Alloy
# (MPEA) datasets, covering composition-to-property prediction for mechanical
# and thermophysical properties.
# ---------------------------------------------------------------------------

DATASCRIBE_KNOWN_DATASETS = [
    {
        "dataset_name": "atlas_hea_liquidus",
        # AlCuCrNbNiFeMo HEA: predict liquidus temperature from composition
        "github_path": "datasets/ATLAS-HEADATA_Jonathan Frutschy/data/data_LIQUID_variable_temprange9_processed.csv",
        "target_column": "PROP LT (K)",
        "task_type": "regression",
        "primary_metric": "mae",
        "id_column": "",
        "group_column": "",
        "split_method": "random",
        "notes": "AlCuCrNbNiFeMo HEA liquidus temperature (CALPHAD) — ATLAS campaign",
    },
    {
        "dataset_name": "atlas_rhea_creep",
        # NbCrVWZr refractory HEA: predict creep merit index from composition
        "github_path": "datasets/ATLAS-RHEADATA/input_data/v3/NbCrVWZr_data_stoic_creep_equil_v3.csv",
        "target_column": "Creep Merit",
        "task_type": "regression",
        "primary_metric": "mae",
        "id_column": "",
        "group_column": "",
        "split_method": "random",
        "notes": "NbCrVWZr refractory HEA creep merit index — ATLAS campaign",
    },
    {
        "dataset_name": "birdshot_hea_hardness",
        # AlCoCrCuFeMnNiV HEA: predict Vickers hardness from composition
        "github_path": "datasets/BIRDSHOT-HEADATA/data/HTMDEC_MasterTable_Iterations_v3_processed.csv",
        "target_column": "Hardness, HV",
        "task_type": "regression",
        "primary_metric": "mae",
        "id_column": "",
        "group_column": "",
        "split_method": "random",
        "notes": "AlCoCrCuFeMnNiV HEA Vickers hardness — BIRDSHOT campaign (iterations v3)",
    },
    {
        "dataset_name": "birdshot_hea_v5_hardness",
        # Expanded v5 dataset (Alvi/DeepGP extension) — more iterations, same alloy space
        "github_path": "datasets/BIRDSHOT-HEADATA-DeepGP-Alvi/data/HTMDEC_MasterTable_Iterations_v5_processed.csv",
        "target_column": "Hardness, HV",
        "task_type": "regression",
        "primary_metric": "mae",
        "id_column": "",
        "group_column": "",
        "split_method": "random",
        "notes": "AlCoCrCuFeMnNiV HEA Vickers hardness — BIRDSHOT v5 (DeepGP/Alvi extension)",
    },
    {
        "dataset_name": "borgHEA_hardness",
        # Literature compilation of ~2000 MPEA compositions with measured properties
        "github_path": "datasets/BorgHEA-DATA/data/Borg_df_updated.csv",
        "target_column": "PROPERTY: HV",
        "task_type": "regression",
        "primary_metric": "mae",
        "id_column": "IDENTIFIER: Reference ID",
        "group_column": "",
        "split_method": "random",
        "notes": "MPEA/HEA Vickers hardness — BorgHEA literature compilation (~2000 alloys)",
    },
]

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/vahid2364/DataScribe_DeepTabularLearning/main"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_pandas():
    try:
        import pandas as pd
        return pd
    except ImportError:
        raise SystemExit("pandas is required. Run: uv sync")


def read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    rows = []
    with manifest_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cleaned = {k.strip(): v.strip() for k, v in row.items()}
            if cleaned.get("dataset_name") and not cleaned["dataset_name"].startswith("#"):
                rows.append(cleaned)
    return rows


def download_file(url: str, dest: Path) -> bool:
    try:
        import requests
    except ImportError:
        print("  ERROR: requests not installed. Run: uv sync")
        return False

    # URL-encode spaces and other special chars in the path portion
    from urllib.parse import quote
    scheme, rest = url.split("://", 1)
    host, path = rest.split("/", 1)
    encoded_url = f"{scheme}://{host}/{quote(path)}"

    try:
        response = requests.get(encoded_url, timeout=120)
        if response.status_code == 200:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(response.content)
            print(f"  Downloaded → {dest}")
            return True
        else:
            print(f"  HTTP {response.status_code}: {encoded_url}")
            return False
    except Exception as exc:
        print(f"  Download failed: {encoded_url} — {exc}")
        return False


def run_materials_project_init(
    project_folder: Path,
    data_path: Path,
    row: dict[str, str],
) -> bool:
    cmd = [
        "uv", "run", "materials-project", "init", str(project_folder),
        "--data", str(data_path),
        "--target", row["target_column"],
    ]
    if row.get("task_type"):
        cmd += ["--task", row["task_type"]]
    if row.get("primary_metric"):
        cmd += ["--metric", row["primary_metric"]]
    if row.get("id_column"):
        cmd += ["--id-column", row["id_column"]]
    if row.get("group_column"):
        cmd += ["--group-column", row["group_column"]]
    if row.get("dataset_name"):
        cmd += ["--name", f"DataScribe {row['dataset_name'].replace('_', ' ').title()}"]
    goal = row.get("notes") or f"TabM benchmark baseline for {row.get('dataset_name','dataset')}."
    cmd += ["--goal", goal, "--yes"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  materials-project init failed:")
            print("  " + result.stderr[-1000:])
            return False
        return True
    except FileNotFoundError:
        # Fallback: run materials_project.py directly
        cmd2 = [sys.executable, "materials_project.py"] + cmd[3:]
        result2 = subprocess.run(cmd2, capture_output=True, text=True)
        if result2.returncode != 0:
            print(f"  init failed (fallback):")
            print("  " + result2.stderr[-1000:])
            return False
        return True


def set_tabm_config(project_folder: Path) -> None:
    config_path = project_folder / "specs" / "model_config.json"
    if not config_path.exists():
        return
    current = json.loads(config_path.read_text())
    current["trainer"] = "tabm"
    current.setdefault("tabm", {}).update(
        {
            "arch_type": "tabm",
            "k": 32,
            "n_blocks": 3,
            "d_block": 256,
            "dropout": 0.1,
        }
    )
    current.setdefault("training", {}).update(
        {
            "epochs": 300,
            "patience": 40,
            "batch_size": 256,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "seed": 42,
        }
    )
    config_path.write_text(json.dumps(current, indent=2) + "\n")
    print(f"  Trainer set to TabM: {config_path}")


def run_baseline(project_folder: Path) -> bool:
    cmd = ["uv", "run", "materials-train", str(project_folder), "--run-id", "mlp-sanity"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            print("  Baseline training failed:")
            print("  " + result.stderr[-800:])
            return False
        for line in result.stdout.splitlines()[-6:]:
            print(f"  {line}")
        return True
    except subprocess.TimeoutExpired:
        print("  Baseline training timed out.")
        return False
    except Exception as exc:
        print(f"  Baseline training error: {exc}")
        return False


# ---------------------------------------------------------------------------
# Main commands
# ---------------------------------------------------------------------------


def cmd_from_manifest(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).expanduser().resolve()
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}")
        return 1

    output_dir = Path(args.output_dir).expanduser().resolve()
    rows = read_manifest(manifest_path)
    if not rows:
        print("ERROR: manifest is empty or has no valid rows.")
        return 1

    print(f"\nSetting up {len(rows)} dataset(s) from {manifest_path}")
    print(f"Output directory: {output_dir}")
    print()

    results: list[dict[str, Any]] = []
    for row in rows:
        name = row.get("dataset_name", "unknown")
        print(f"{'─' * 60}")
        print(f"Dataset: {name}")

        data_path_str = row.get("data_path", "")
        if not data_path_str or data_path_str.startswith("path/to"):
            print(f"  SKIP: data_path not set for '{name}'. Edit the manifest.")
            results.append({"name": name, "status": "skipped", "reason": "data_path not set"})
            continue

        data_path = Path(data_path_str).expanduser()
        if not data_path.is_absolute():
            data_path = manifest_path.parent / data_path

        if not data_path.exists():
            print(f"  ERROR: data file not found: {data_path}")
            results.append({"name": name, "status": "error", "reason": f"file not found: {data_path}"})
            continue

        project_folder = output_dir / name
        if project_folder.exists() and (project_folder / "specs" / "project.json").exists():
            print(f"  Already set up at {project_folder}")
            if args.force:
                print("  --force: removing and recreating.")
                shutil.rmtree(project_folder)
            else:
                print("  Use --force to recreate.")
                results.append({"name": name, "status": "already_exists"})
                if args.train_baseline:
                    print("  Running baseline...")
                    run_baseline(project_folder)
                continue

        print(f"  Initialising project folder: {project_folder}")
        ok = run_materials_project_init(project_folder, data_path, row)
        if not ok:
            results.append({"name": name, "status": "error", "reason": "init failed"})
            continue

        print("  Setting TabM as default trainer.")
        set_tabm_config(project_folder)

        if args.train_baseline:
            print("  Running MLP sanity baseline...")
            train_ok = run_baseline(project_folder)
            if train_ok:
                print("  Running TabM baseline...")
                cmd_tabm = [
                    "uv", "run", "materials-train",
                    str(project_folder),
                    "--run-id", "tabm-baseline-v1",
                ]
                subprocess.run(cmd_tabm, capture_output=False, timeout=7200)

        results.append({"name": name, "status": "ok", "folder": str(project_folder)})
        print(f"  Done: {project_folder}")

    print(f"\n{'═' * 60}")
    print("SETUP SUMMARY")
    print(f"{'═' * 60}")
    for r in results:
        print(f"  {r['name']}: {r['status']}")
    print()
    print("Next steps:")
    for r in results:
        if r.get("status") == "ok":
            folder = r.get("folder", "")
            print(f"  uv run materials-train {folder}")
    print()
    print("To run the overnight research agent on a project:")
    print("  export ANTHROPIC_API_KEY=sk-ant-...")
    for r in results:
        if r.get("status") == "ok":
            print(f"  uv run research-agent {r['folder']} --max-hours 8")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).expanduser().resolve()
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nAttempting to download DataScribe datasets from GitHub...")
    print(f"GitHub base: {GITHUB_RAW_BASE}")
    print(f"Local data dir: {data_dir}")
    print()

    downloaded_manifest_rows = []
    for ds in DATASCRIBE_KNOWN_DATASETS:
        name = ds["dataset_name"]
        github_path = ds["github_path"]
        url = f"{GITHUB_RAW_BASE}/{github_path}"
        dest = data_dir / f"{name}.csv"

        print(f"  {name}: ", end="", flush=True)
        if dest.exists() and not args.force:
            print(f"already present at {dest}")
        else:
            ok = download_file(url, dest)
            if not ok:
                print(f"  WARNING: Could not download {name}. Check the URL or download manually.")
                print(f"  Manual URL: {url}")
                continue

        row = dict(ds)
        row["data_path"] = str(dest)
        downloaded_manifest_rows.append(row)

    if not downloaded_manifest_rows:
        print("\nNo datasets were downloaded. Please download manually.")
        print(f"GitHub repository: https://github.com/vahid2364/DataScribe_DeepTabularLearning")
        return 1

    # Write a ready-to-use manifest
    manifest_path = output_dir / "datascribe_manifest.csv"
    fieldnames = [
        "dataset_name", "data_path", "target_column", "task_type",
        "primary_metric", "id_column", "group_column", "split_method", "notes",
    ]
    with manifest_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(downloaded_manifest_rows)

    print(f"\nManifest written: {manifest_path}")
    print(f"\nNext: set up project folders from the manifest:")
    print(f"  uv run setup-datascribe --manifest {manifest_path} --output-dir {output_dir}/projects")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Set up Materials Autoresearch project folders for DataScribe benchmark datasets."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    # -- from manifest --
    p_manifest = sub.add_parser(
        "from-manifest",
        help="Create project folders from a manifest CSV (default command)",
    )
    p_manifest.add_argument(
        "--manifest",
        required=True,
        help="Path to manifest CSV (see examples/datascribe_manifest_template.csv)",
    )
    p_manifest.add_argument(
        "--output-dir",
        default="projects/datascribe",
        help="Parent directory for project folders (default: projects/datascribe)",
    )
    p_manifest.add_argument(
        "--train-baseline",
        action="store_true",
        help="Run MLP sanity + TabM baseline after setup",
    )
    p_manifest.add_argument(
        "--force",
        action="store_true",
        help="Recreate existing project folders",
    )

    # -- download --
    p_download = sub.add_parser(
        "download",
        help="Download DataScribe datasets from GitHub and write a manifest",
    )
    p_download.add_argument(
        "--output-dir",
        default="projects/datascribe",
        help="Directory to store downloaded data and manifest",
    )
    p_download.add_argument(
        "--force",
        action="store_true",
        help="Re-download files that already exist",
    )

    # Top-level shortcut: if no subcommand, default to from-manifest with --manifest
    parser.add_argument(
        "--manifest",
        help="Shortcut: path to manifest CSV (runs from-manifest command)",
    )
    parser.add_argument(
        "--output-dir",
        default="projects/datascribe",
        help="Parent directory for project folders",
    )
    parser.add_argument(
        "--train-baseline",
        action="store_true",
        help="Run MLP sanity + TabM baseline after setup",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate existing project folders",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Shortcut: download DataScribe datasets from GitHub",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Handle top-level shortcuts
    if args.download or (hasattr(args, "command") and args.command == "download"):
        return cmd_download(args)

    if args.manifest or (hasattr(args, "command") and args.command == "from-manifest"):
        return cmd_from_manifest(args)

    parser.print_help()
    print("\nQUICK START:")
    print("  # Download DataScribe datasets:")
    print("  uv run setup-datascribe --download --output-dir projects/datascribe")
    print()
    print("  # Or use a manifest with local files:")
    print("  uv run setup-datascribe --manifest examples/datascribe_manifest_template.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
