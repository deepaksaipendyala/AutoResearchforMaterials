#!/usr/bin/env python3
"""
Overnight research agent for Materials Autoresearch.

Reads the current project state, asks Claude to propose one hyperparameter
experiment at a time, runs training, and logs results. Iterates until it
reaches the time or iteration limit, or the researcher presses Ctrl+C.

Usage:
    uv run research-agent projects/my_project
    uv run research-agent projects/my_project --max-iterations 10
    uv run research-agent projects/my_project --max-hours 8
    uv run research-agent projects/my_project --model claude-opus-4-7
    uv run research-agent projects/my_project --dry-run

Requirements:
    ANTHROPIC_API_KEY environment variable must be set.
    Run `uv run materials-train <project>` at least once before starting the agent.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Reading project state
# ---------------------------------------------------------------------------


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    if not lines:
        return []
    headers = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        if line.strip():
            values = line.split("\t")
            row = dict(zip(headers, values))
            # Fill missing columns with empty string
            for h in headers:
                row.setdefault(h, "")
            rows.append(row)
    return rows


def read_project_state(project_folder: Path) -> dict[str, Any]:
    spec = load_json(project_folder / "specs" / "project.json")
    model_config = load_json(project_folder / "specs" / "model_config.json")
    edit_scope = load_json(project_folder / "agent" / "edit_scope.json")

    brief_path = project_folder / "agent" / "research_brief.md"
    research_brief = brief_path.read_text() if brief_path.exists() else ""

    experiment_log = read_tsv(project_folder / "agent" / "experiment_log.tsv")

    primary_metric = spec.get("evaluation", {}).get("primary_metric", "mae")
    higher_is_better = primary_metric in {
        "accuracy", "balanced_accuracy", "roc_auc", "r2",
        "top_k_recall", "enrichment_factor",
    }

    best_run: dict[str, str] | None = None
    best_value: float | None = None

    for row in experiment_log:
        if row.get("status") != "complete":
            continue
        try:
            value = float(row["value"])
        except (ValueError, KeyError):
            continue
        if best_value is None:
            best_value = value
            best_run = row
        elif higher_is_better and value > best_value:
            best_value = value
            best_run = row
        elif not higher_is_better and value < best_value:
            best_value = value
            best_run = row

    # Read metrics from the best run, not just the latest
    best_metrics: dict[str, Any] = {}
    if best_run:
        run_id = best_run.get("run_id", "")
        metrics_path = project_folder / "runs" / run_id / "metrics.json"
        best_metrics = load_json(metrics_path)

    return {
        "spec": spec,
        "model_config": model_config,
        "edit_scope": edit_scope,
        "research_brief": research_brief,
        "experiment_log": experiment_log,
        "best_run": best_run,
        "best_value": best_value,
        "best_metrics": best_metrics,
        "primary_metric": primary_metric,
        "higher_is_better": higher_is_better,
    }


# ---------------------------------------------------------------------------
# Building the prompt
# ---------------------------------------------------------------------------


def format_log_summary(experiment_log: list[dict[str, str]], max_rows: int = 12) -> str:
    if not experiment_log:
        return "  (none)"
    rows = experiment_log[-max_rows:]
    lines = ["  run_id | metric | value | model | notes"]
    lines.append("  " + "-" * 70)
    for row in rows:
        lines.append(
            f"  {row.get('run_id','')[:28]} | "
            f"{row.get('primary_metric','')} | "
            f"{row.get('value','')} | "
            f"{row.get('model','')} | "
            f"{row.get('notes','')[:40]}"
        )
    return "\n".join(lines)


def build_agent_prompt(state: dict[str, Any], iteration: int) -> str:
    spec = state["spec"]
    model_config = state["model_config"]
    experiment_log = state["experiment_log"]
    best_run = state["best_run"]
    primary_metric = state["primary_metric"]
    higher_is_better = state["higher_is_better"]

    project_name = spec.get("project_name", "Unknown")
    task_type = spec.get("task", {}).get("type", "regression")
    target = spec.get("data", {}).get("target_column", "unknown")
    direction = "higher is better" if higher_is_better else "lower is better"

    if best_run:
        best_info = (
            f"Best run so far:\n"
            f"  run_id: {best_run['run_id']}\n"
            f"  {primary_metric}: {best_run['value']} ({direction})\n"
            f"  model: {best_run.get('model','unknown')}\n"
            f"  notes: {best_run.get('notes','')}"
        )
    else:
        best_info = "No successful runs yet."

    # Summarise tried configs to help the agent avoid repeats
    tried_notes = [row.get("notes", "") for row in experiment_log if row.get("status") == "complete"]
    tried_summary = "\n".join(f"  - {n}" for n in tried_notes[-8:]) if tried_notes else "  (none)"

    return f"""You are an autonomous research agent helping optimize a machine learning model for materials science.

PROJECT
  Name: {project_name}
  Task: predict '{target}' ({task_type})
  Primary metric: {primary_metric} ({direction})
  Iteration: {iteration}

CURRENT MODEL CONFIG (specs/model_config.json)
```json
{json.dumps(model_config, indent=2)}
```

BEST RESULT SO FAR
{best_info}

EXPERIMENT HISTORY
{format_log_summary(experiment_log)}

CONFIGS TRIED (notes from log)
{tried_summary}

EDIT RULES — you may ONLY change these keys inside model_config.json:
  TabM: arch_type, k, n_blocks, d_block, dropout
  Training: lr, weight_decay, batch_size, epochs, patience, seed
  MLP (if trainer=tabular_mlp): hidden_dim, dropout

You may NOT change: trainer (unless switching from tabular_mlp to tabm as a deliberate upgrade),
  target_column, split, primary_metric, data files, or feature roles.

TASK
Propose ONE experiment — a small, specific change to model_config.json.
Prefer changes that directly target the observed weakness in the current best result.
Avoid repeating configurations already tried (check the CONFIGS TRIED section above).
If TabM has not been tried yet and trainer is tabular_mlp, switching to tabm is a high-priority upgrade.

Respond with EXACTLY this format — nothing before or after:

REASONING:
<2-4 sentences. What you are changing, why it might help for this specific property prediction task,
and what you expect to happen. Write for a materials scientist, not a software engineer.>

EXPERIMENT_NAME:
<a short, filesystem-safe name, e.g. "tabm-k64" or "lower-lr-0003">

CONFIG_CHANGE:
```json
<ONLY the keys you want to change, as a nested JSON patch — not the full config>
```
"""


# ---------------------------------------------------------------------------
# Parsing the response
# ---------------------------------------------------------------------------


def parse_agent_response(response_text: str) -> tuple[str, str, dict[str, Any]]:
    reasoning = ""
    experiment_name = ""
    config_change: dict[str, Any] = {}

    reasoning_match = re.search(
        r"REASONING:\s*\n(.*?)(?=EXPERIMENT_NAME:|CONFIG_CHANGE:|$)",
        response_text,
        re.DOTALL,
    )
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    name_match = re.search(r"EXPERIMENT_NAME:\s*\n?(.+?)(?:\n|$)", response_text)
    if name_match:
        experiment_name = name_match.group(1).strip()

    json_match = re.search(
        r"CONFIG_CHANGE:\s*\n```(?:json)?\s*\n(.*?)```",
        response_text,
        re.DOTALL,
    )
    if json_match:
        try:
            config_change = json.loads(json_match.group(1).strip())
        except json.JSONDecodeError as exc:
            print(f"  WARNING: Could not parse config change JSON: {exc}")

    return reasoning, experiment_name, config_change


# ---------------------------------------------------------------------------
# Config management
# ---------------------------------------------------------------------------


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in patch.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def save_config(project_folder: Path, config: dict[str, Any]) -> None:
    config_path = project_folder / "specs" / "model_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def make_run_id(experiment_name: str, iteration: int) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "-", experiment_name)[:40].strip("-")
    return f"{timestamp}-{iteration:02d}-{safe_name}"


def find_repo_root(project_folder: Path) -> Path | None:
    """Walk up from project_folder to find the directory containing pyproject.toml."""
    candidate = project_folder
    for _ in range(6):
        if (candidate / "pyproject.toml").exists():
            return candidate
        candidate = candidate.parent
    return None


def run_training(project_folder: Path, run_id: str) -> tuple[bool, str]:
    cmd = ["uv", "run", "materials-train", str(project_folder), "--run-id", run_id]
    repo_root = find_repo_root(project_folder)
    cwd = repo_root or Path.cwd()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, cwd=cwd)
        output = result.stdout + ("\n" + result.stderr if result.stderr.strip() else "")
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, "Training timed out after 2 hours."
    except FileNotFoundError:
        # Fallback: try running train_materials.py directly
        train_script = cwd / "train_materials.py"
        cmd2 = [sys.executable, str(train_script), str(project_folder), "--run-id", run_id]
        try:
            result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=7200, cwd=cwd)
            output2 = result2.stdout + ("\n" + result2.stderr if result2.stderr.strip() else "")
            return result2.returncode == 0, output2
        except Exception as exc2:
            return False, str(exc2)
    except Exception as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def update_experiment_plan(
    project_folder: Path,
    reasoning: str,
    experiment_name: str,
    result_line: str,
) -> None:
    plan_path = project_folder / "agent" / "experiment_plan.md"
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    block = (
        f"\n---\n"
        f"## {timestamp} — {experiment_name}\n\n"
        f"**Reasoning:** {reasoning}\n\n"
        f"**Result:** {result_line}\n"
    )
    current = plan_path.read_text() if plan_path.exists() else "# Experiment Plan\n"
    plan_path.write_text(current + block)


# ---------------------------------------------------------------------------
# Claude API call
# ---------------------------------------------------------------------------


def call_claude(client: Any, model: str, prompt: str) -> str:
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        system=(
            "You are a careful, methodical research agent for materials-science machine learning. "
            "You propose one targeted experiment at a time and always explain your reasoning in terms "
            "a materials scientist would understand. You never break format."
        ),
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Overnight research agent: uses Claude to propose and test TabM experiments "
            "on a Materials Autoresearch project folder."
        )
    )
    parser.add_argument("project_dir", help="Materials Autoresearch project folder")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="Maximum number of experiments to run (default: 20)",
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=8.0,
        help="Stop after this many hours (default: 8)",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Claude model to use (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be proposed without running training",
    )
    args = parser.parse_args(argv)

    try:
        import anthropic
    except ImportError:
        print("ERROR: anthropic package not installed.")
        print("Fix: uv add anthropic  (or: pip install anthropic)")
        return 1

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.")
        print("Set it with:  export ANTHROPIC_API_KEY=sk-ant-...")
        print("Get your key: https://console.anthropic.com/")
        return 1

    project_folder = Path(args.project_dir).expanduser().resolve()
    if not (project_folder / "specs" / "project.json").exists():
        print(f"ERROR: Not a valid project folder: {project_folder}")
        print("Run first:  uv run materials-project validate <project_folder>")
        return 1

    client = anthropic.Anthropic(api_key=api_key)
    deadline = time.time() + args.max_hours * 3600
    start_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

    print("=" * 70)
    print("MATERIALS AUTORESEARCH — OVERNIGHT AGENT")
    print("=" * 70)
    print(f"Project:        {project_folder}")
    print(f"Started:        {start_iso}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Max hours:      {args.max_hours}")
    print(f"Claude model:   {args.model}")
    if args.dry_run:
        print("Mode:           DRY RUN (no training)")
    print("=" * 70)
    print()

    for iteration in range(1, args.max_iterations + 1):
        if time.time() > deadline:
            print(f"\nTime limit reached ({args.max_hours}h). Stopping.")
            break

        elapsed_h = (time.time() - (deadline - args.max_hours * 3600)) / 3600
        remaining_h = (deadline - time.time()) / 3600
        print(f"\n{'─' * 70}")
        print(
            f"ITERATION {iteration}/{args.max_iterations}  "
            f"(elapsed {elapsed_h:.1f}h, remaining {remaining_h:.1f}h)"
        )
        print(f"{'─' * 70}")

        state = read_project_state(project_folder)

        if not state["experiment_log"]:
            print(
                "\nERROR: No baseline run found in experiment_log.tsv.\n"
                "Run the baseline first:\n"
                f"  uv run materials-train {project_folder}\n"
            )
            return 1

        saved_config = dict(state["model_config"])  # copy for rollback

        print(f"  Best {state['primary_metric']} so far: {state['best_value']}")
        print(f"  Asking {args.model} for experiment proposal...")

        prompt = build_agent_prompt(state, iteration)
        try:
            response = call_claude(client, args.model, prompt)
        except Exception as exc:
            print(f"  ERROR calling Claude API: {exc}")
            print("  Stopping agent.")
            break

        reasoning, experiment_name, config_change = parse_agent_response(response)

        if not config_change:
            print("  Agent did not return a valid config change. Full response:")
            print("  " + response.replace("\n", "\n  "))
            print("  Stopping.")
            break

        print(f"\n  Proposed: {experiment_name}")
        print(f"  Reasoning: {reasoning}")
        print(f"  Config change: {json.dumps(config_change)}")

        if args.dry_run:
            print("  [Dry run] Would apply this change and run training. Skipping.")
            continue

        # Apply config change
        new_config = deep_merge(state["model_config"], config_change)
        save_config(project_folder, new_config)

        run_id = make_run_id(experiment_name or f"exp-{iteration}", iteration)
        print(f"\n  Training (run_id: {run_id}) ...")
        t0 = time.time()
        success, output = run_training(project_folder, run_id)
        elapsed_train = time.time() - t0

        print(f"  Training took {elapsed_train/60:.1f} min — {'SUCCESS' if success else 'FAILED'}")

        if not success:
            print("  Training output (last 20 lines):")
            for line in output.splitlines()[-20:]:
                print(f"    {line}")
            print("  Restoring config and continuing.")
            save_config(project_folder, saved_config)
            update_experiment_plan(
                project_folder,
                reasoning,
                experiment_name or run_id,
                f"FAILED — training error",
            )
            continue

        # Read results
        metrics_path = project_folder / "runs" / run_id / "metrics.json"
        metrics = load_json(metrics_path)
        primary_value = metrics.get("primary_value", float("nan"))
        best_value = state["best_value"]
        higher_is_better = state["higher_is_better"]
        primary_metric = state["primary_metric"]

        is_better = False
        if best_value is not None and not (primary_value != primary_value):  # NaN check
            is_better = (
                (higher_is_better and primary_value > best_value)
                or (not higher_is_better and primary_value < best_value)
            )

        tag = "IMPROVED" if is_better else "no improvement"
        result_line = (
            f"{primary_metric}={primary_value:.6f}  "
            f"(best: {best_value:.6f if best_value is not None else 'N/A'})  "
            f"— {tag}"
        )
        print(f"\n  Result: {result_line}")

        # Print last few lines of training output
        print("  Training output (last 8 lines):")
        for line in output.splitlines()[-8:]:
            print(f"    {line}")

        update_experiment_plan(
            project_folder,
            reasoning,
            experiment_name or run_id,
            result_line,
        )

        if not is_better and best_value is not None:
            print("  Not better — restoring config to the best-known state.")
            save_config(project_folder, saved_config)

        time.sleep(2)  # brief pause to avoid hammering the API

    print("\n" + "=" * 70)
    print("AGENT FINISHED")
    print(f"Experiment log: {project_folder}/agent/experiment_log.tsv")
    print(f"Experiment plan: {project_folder}/agent/experiment_plan.md")
    print(f"Runs: {project_folder}/runs/")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
