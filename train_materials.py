#!/usr/bin/env python3
"""
Train a simple baseline model for a Materials Autoresearch project.

This is intentionally conservative. It gives every project a first model,
first metric, and first experiment folder before an agent starts changing
feature processing or model design.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from materials_project import read_table


@dataclass
class TrainConfig:
    epochs: int = 200
    patience: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 128
    dropout: float = 0.05
    seed: int = 42


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-baseline")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_model_config(root: Path) -> dict[str, Any]:
    path = root / "specs" / "model_config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def build_train_config(root: Path, args: argparse.Namespace) -> TrainConfig:
    project_config = load_model_config(root)
    training = project_config.get("training", {})
    model = project_config.get("model", {})
    defaults = TrainConfig()

    def choose(name: str, section: dict[str, Any], cli_value: Any):
        if cli_value is not None:
            return cli_value
        if name in section:
            return section[name]
        return getattr(defaults, name)

    return TrainConfig(
        epochs=int(choose("epochs", training, args.epochs)),
        patience=int(choose("patience", training, args.patience)),
        batch_size=int(choose("batch_size", training, args.batch_size)),
        lr=float(choose("lr", training, args.lr)),
        weight_decay=float(choose("weight_decay", training, args.weight_decay)),
        hidden_dim=int(choose("hidden_dim", model, args.hidden_dim)),
        dropout=float(choose("dropout", model, args.dropout)),
        seed=int(choose("seed", training, args.seed)),
    )


def load_data_dictionary(root: Path) -> list[dict[str, str]]:
    path = root / "specs" / "data_dictionary.csv"
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def choose_feature_columns(
    df,
    spec: dict[str, Any],
    data_dictionary: list[dict[str, str]],
    model_config: dict[str, Any],
):
    feature_selection = model_config.get("feature_selection", {})
    numeric_roles = set(feature_selection.get("numeric_roles", ["feature", "processing_feature"]))
    categorical_roles = set(feature_selection.get("categorical_roles", ["categorical_feature", "composition"]))
    if data_dictionary:
        numeric = [row["column"] for row in data_dictionary if row.get("role") in numeric_roles]
        categorical = [row["column"] for row in data_dictionary if row.get("role") in categorical_roles]
    else:
        data = spec["data"]
        excluded = {
            data.get("target_column"),
            data.get("id_column"),
            data.get("structure_column"),
            data.get("group_column"),
        }
        numeric = [column for column in df.select_dtypes(include=[np.number]).columns if column not in excluded]
        categorical = [
            column
            for column in df.columns
            if column not in excluded and column not in numeric and df[column].nunique(dropna=True) <= 50
        ]

    numeric = [column for column in numeric if column in df.columns]
    categorical = [column for column in categorical if column in df.columns]
    return numeric, categorical


def split_dataframe(df, spec: dict[str, Any], seed: int):
    data = spec["data"]
    split = spec["evaluation"]["split"]
    method = split.get("method", "random")

    if "split_column" in split and split["split_column"] in df.columns:
        column = split["split_column"]
        lower = df[column].astype(str).str.lower()
        return (
            df[lower == "train"].copy(),
            df[lower.isin(["val", "valid", "validation"])].copy(),
            df[lower == "test"].copy(),
            f"column:{column}",
        )

    rng = np.random.default_rng(seed)
    if method == "group" and data.get("group_column"):
        group_column = data["group_column"]
        groups = np.array(sorted(df[group_column].dropna().astype(str).unique()))
        rng.shuffle(groups)
        n_groups = len(groups)
        n_train = max(1, int(round(n_groups * float(split.get("train_fraction", 0.8)))))
        n_val = max(1, int(round(n_groups * float(split.get("validation_fraction", 0.1)))))
        train_groups = set(groups[:n_train])
        val_groups = set(groups[n_train : n_train + n_val])
        test_groups = set(groups[n_train + n_val :])
        if not test_groups and val_groups:
            test_groups.add(val_groups.pop())
        group_values = df[group_column].astype(str)
        return (
            df[group_values.isin(train_groups)].copy(),
            df[group_values.isin(val_groups)].copy(),
            df[group_values.isin(test_groups)].copy(),
            f"group:{group_column}",
        )

    indices = np.arange(len(df))
    rng.shuffle(indices)
    n = len(indices)
    n_train = max(1, int(round(n * float(split.get("train_fraction", 0.8)))))
    n_val = max(1, int(round(n * float(split.get("validation_fraction", 0.1)))))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy(), df.iloc[test_idx].copy(), "random"


def fit_feature_transform(train_df, numeric_columns: list[str], categorical_columns: list[str]):
    import pandas as pd

    numeric_train = train_df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    medians = numeric_train.median(numeric_only=True)
    means = numeric_train.fillna(medians).mean()
    stds = numeric_train.fillna(medians).std().replace(0, 1).fillna(1)

    categories: dict[str, list[str]] = {}
    for column in categorical_columns:
        values = train_df[column].fillna("<missing>").astype(str)
        categories[column] = sorted(values.unique().tolist())
    return {"medians": medians, "means": means, "stds": stds, "categories": categories}


def transform_features(df, numeric_columns: list[str], categorical_columns: list[str], fit: dict[str, Any]):
    import pandas as pd

    frames = []
    if numeric_columns:
        numeric = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.fillna(fit["medians"])
        numeric = (numeric - fit["means"]) / fit["stds"]
        frames.append(numeric)

    for column in categorical_columns:
        values = df[column].fillna("<missing>").astype(str)
        for category in fit["categories"][column]:
            frames.append((values == category).astype(float).rename(f"{column}={category}").to_frame())

    if not frames:
        raise ValueError("No usable feature columns were found.")

    features = pd.concat(frames, axis=1)
    return features.to_numpy(dtype=np.float32), list(features.columns)


def fit_tabular_transform(train_df, numeric_columns: list[str], categorical_columns: list[str]):
    import pandas as pd

    numeric_train = train_df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    medians = numeric_train.median(numeric_only=True)
    means = numeric_train.fillna(medians).mean()
    stds = numeric_train.fillna(medians).std().replace(0, 1).fillna(1)

    categories: dict[str, dict[str, int]] = {}
    for column in categorical_columns:
        values = train_df[column].fillna("<missing>").astype(str)
        categories[column] = {value: index for index, value in enumerate(sorted(values.unique().tolist()))}

    return {
        "medians": medians,
        "means": means,
        "stds": stds,
        "categories": categories,
    }


def transform_tabular_features(
    df,
    numeric_columns: list[str],
    categorical_columns: list[str],
    fit: dict[str, Any],
):
    import pandas as pd

    if numeric_columns:
        numeric = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.fillna(fit["medians"])
        numeric = (numeric - fit["means"]) / fit["stds"]
        x_num = numeric.to_numpy(dtype=np.float32)
    else:
        x_num = np.zeros((len(df), 0), dtype=np.float32)

    cat_arrays = []
    cat_cardinalities = []
    for column in categorical_columns:
        mapping = fit["categories"][column]
        unknown_id = len(mapping)
        values = df[column].fillna("<missing>").astype(str)
        encoded = values.map(lambda value: mapping.get(value, unknown_id)).to_numpy(dtype=np.int64)
        cat_arrays.append(encoded)
        cat_cardinalities.append(unknown_id + 1)

    if cat_arrays:
        x_cat = np.stack(cat_arrays, axis=1).astype(np.int64)
    else:
        x_cat = np.zeros((len(df), 0), dtype=np.int64)

    return x_num, x_cat, numeric_columns + categorical_columns, cat_cardinalities


def prepare_targets(train_df, val_df, test_df, target_column: str, task_type: str):
    if task_type in {"regression", "screening"}:
        return (
            train_df[target_column].astype(float).to_numpy(dtype=np.float32).reshape(-1, 1),
            val_df[target_column].astype(float).to_numpy(dtype=np.float32).reshape(-1, 1),
            test_df[target_column].astype(float).to_numpy(dtype=np.float32).reshape(-1, 1),
            None,
        )

    labels = sorted(train_df[target_column].dropna().astype(str).unique().tolist())
    label_to_id = {label: index for index, label in enumerate(labels)}

    def encode(frame):
        values = frame[target_column].astype(str)
        unknown = sorted(set(values.unique()) - set(label_to_id))
        if unknown:
            raise ValueError(f"Validation/test split has labels not present in train split: {unknown}")
        return values.map(label_to_id).to_numpy(dtype=np.int64)

    return encode(train_df), encode(val_df), encode(test_df), label_to_id


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum((y_pred - y_true) ** 2) / denom) if denom > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = y_true.reshape(-1)
    scores = scores.reshape(-1)
    positive = y_true == 1
    negative = y_true == 0
    n_pos = int(positive.sum())
    n_neg = int(negative.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def classification_metrics(y_true: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    logits = np.asarray(logits)
    pred = logits.argmax(axis=1)
    accuracy = float(np.mean(pred == y_true))
    per_class = []
    for label in sorted(np.unique(y_true)):
        mask = y_true == label
        per_class.append(float(np.mean(pred[mask] == y_true[mask])))
    metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(per_class)) if per_class else float("nan"),
    }
    if logits.shape[1] == 2:
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        metrics["roc_auc"] = binary_auc(y_true, probs[:, 1])
    return metrics


def evaluate_model(model, x: np.ndarray, y: np.ndarray, task_type: str, device: torch.device):
    model.eval()
    with torch.no_grad():
        tensor_x = torch.tensor(x, dtype=torch.float32, device=device)
        output = model(tensor_x).detach().cpu().numpy()
    if task_type in {"regression", "screening"}:
        return regression_metrics(y, output), output
    return classification_metrics(y, output), output


def call_tabm(model, x_num: torch.Tensor, x_cat: torch.Tensor | None):
    if x_cat is None or x_cat.shape[1] == 0:
        try:
            return model(x_num)
        except TypeError:
            return model(x_num, None)
    return model(x_num, x_cat)


def tabm_loss(output: torch.Tensor, target: torch.Tensor, task_type: str) -> torch.Tensor:
    if task_type == "regression":
        pred = output.squeeze(-1)
        target = target.float().view(-1)
        return nn.functional.mse_loss(pred, target[:, None].expand_as(pred))

    logits = output
    k = logits.shape[1]
    target = target.long().view(-1)
    expanded_target = target[:, None].expand(-1, k).reshape(-1)
    return nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), expanded_target)


def evaluate_tabm(
    model,
    x_num: np.ndarray,
    x_cat: np.ndarray,
    y: np.ndarray,
    task_type: str,
    device: torch.device,
):
    model.eval()
    with torch.no_grad():
        tensor_num = torch.tensor(x_num, dtype=torch.float32, device=device)
        tensor_cat = torch.tensor(x_cat, dtype=torch.long, device=device) if x_cat.shape[1] else None
        output = call_tabm(model, tensor_num, tensor_cat)

    if task_type == "regression":
        predictions = output.squeeze(-1).mean(dim=1).detach().cpu().numpy().reshape(-1, 1)
        return regression_metrics(y, predictions), predictions

    probs = output.softmax(dim=-1).mean(dim=1)
    log_probs = probs.clamp_min(1e-12).log().detach().cpu().numpy()
    return classification_metrics(y, log_probs), log_probs


def metric_is_better(metric_name: str) -> bool:
    return metric_name in {"accuracy", "balanced_accuracy", "roc_auc", "r2", "top_k_recall", "enrichment_factor"}


def choose_primary(metrics: dict[str, float], primary_metric: str) -> float:
    if primary_metric in metrics:
        return float(metrics[primary_metric])
    return float(next(iter(metrics.values())))


def train_loop(
    model,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    task_type: str,
    config: TrainConfig,
    device: torch.device,
):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    if task_type in {"regression", "screening"}:
        criterion = nn.MSELoss()
        target_dtype = torch.float32
    else:
        criterion = nn.CrossEntropyLoss()
        target_dtype = torch.long

    train_dataset = TensorDataset(
        torch.tensor(train_x, dtype=torch.float32),
        torch.tensor(train_y.squeeze() if task_type == "classification" else train_y, dtype=target_dtype),
    )
    loader = DataLoader(train_dataset, batch_size=min(config.batch_size, len(train_dataset)), shuffle=True)

    best_state = None
    best_val_loss = math.inf
    best_epoch = 0
    stale_epochs = 0

    val_tensor_x = torch.tensor(val_x, dtype=torch.float32, device=device)
    val_tensor_y = torch.tensor(val_y.squeeze() if task_type == "classification" else val_y, dtype=target_dtype, device=device)

    for epoch in range(1, config.epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_output = model(val_tensor_x)
            val_loss = float(criterion(val_output, val_tensor_y).item())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "best_val_loss": best_val_loss}


def build_tabm_model(
    n_num_features: int,
    cat_cardinalities: list[int],
    output_dim: int,
    model_config: dict[str, Any],
):
    try:
        from tabm import TabM
    except ImportError as exc:
        raise SystemExit(
            "TabM is selected, but the `tabm` package is not installed. "
            "Run `uv sync` after this repo update, or install `tabm>=0.0.3`."
        ) from exc

    tabm_config = model_config.get("tabm", {})
    kwargs: dict[str, Any] = {
        "n_num_features": n_num_features,
        "cat_cardinalities": cat_cardinalities,
        "d_out": output_dim,
    }
    for key in ["arch_type", "k", "n_blocks", "d_block", "dropout"]:
        if key in tabm_config:
            kwargs[key] = tabm_config[key]
    return TabM.make(**kwargs)


def train_tabm_loop(
    model,
    train_num: np.ndarray,
    train_cat: np.ndarray,
    train_y: np.ndarray,
    val_num: np.ndarray,
    val_cat: np.ndarray,
    val_y: np.ndarray,
    task_type: str,
    config: TrainConfig,
    device: torch.device,
):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    target_dtype = torch.float32 if task_type == "regression" else torch.long
    train_dataset = TensorDataset(
        torch.tensor(train_num, dtype=torch.float32),
        torch.tensor(train_cat, dtype=torch.long),
        torch.tensor(train_y.squeeze() if task_type == "classification" else train_y, dtype=target_dtype),
    )
    loader = DataLoader(train_dataset, batch_size=min(config.batch_size, len(train_dataset)), shuffle=True)

    val_tensor_num = torch.tensor(val_num, dtype=torch.float32, device=device)
    val_tensor_cat = torch.tensor(val_cat, dtype=torch.long, device=device) if val_cat.shape[1] else None
    val_tensor_y = torch.tensor(
        val_y.squeeze() if task_type == "classification" else val_y,
        dtype=target_dtype,
        device=device,
    )

    best_state = None
    best_val_loss = math.inf
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        for batch_num, batch_cat, batch_y in loader:
            batch_num = batch_num.to(device)
            batch_cat = batch_cat.to(device) if batch_cat.shape[1] else None
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = call_tabm(model, batch_num, batch_cat)
            loss = tabm_loss(output, batch_y, task_type)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_output = call_tabm(model, val_tensor_num, val_tensor_cat)
            val_loss = float(tabm_loss(val_output, val_tensor_y, task_type).item())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "best_val_loss": best_val_loss}


def write_predictions(
    path: Path,
    frame,
    id_column: str | None,
    target_column: str,
    task_type: str,
    outputs: np.ndarray,
    label_to_id: dict[str, int] | None,
) -> None:
    rows = []
    id_values = frame[id_column].astype(str).tolist() if id_column and id_column in frame.columns else list(map(str, frame.index))
    if task_type in {"regression", "screening"}:
        predictions = outputs.reshape(-1)
        for item_id, true_value, predicted_value in zip(id_values, frame[target_column].tolist(), predictions):
            rows.append({"id": item_id, "target": true_value, "prediction": float(predicted_value)})
    else:
        id_to_label = {index: label for label, index in (label_to_id or {}).items()}
        pred = outputs.argmax(axis=1)
        for row_index, (item_id, true_value, predicted_id) in enumerate(zip(id_values, frame[target_column].tolist(), pred)):
            row = {
                "id": item_id,
                "target": true_value,
                "prediction": id_to_label.get(int(predicted_id), str(predicted_id)),
            }
            if outputs.shape[1] == 2:
                exp = np.exp(outputs[row_index] - outputs[row_index].max())
                probs = exp / exp.sum()
                row["probability_positive"] = float(probs[1])
            rows.append(row)

    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_experiment_log(
    root: Path,
    run_id: str,
    metric: str,
    value: float,
    split_name: str,
    model_name: str,
    notes: str,
) -> None:
    log_path = root / "agent" / "experiment_log.tsv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        log_path.write_text("run_id\tstarted_at\tstatus\tprimary_metric\tvalue\tsplit\tmodel\tnotes\n")
    with log_path.open("a") as handle:
        handle.write(
            f"{run_id}\t{datetime.now(timezone.utc).isoformat(timespec='seconds')}\tcomplete\t"
            f"{metric}\t{value:.6f}\t{split_name}\t{model_name}\t{notes}\n"
        )


def write_summary(
    path: Path,
    spec: dict[str, Any],
    train_info: dict[str, Any],
    metrics: dict[str, Any],
    primary_metric: str,
    primary_value: float,
    split_name: str,
    model_name: str,
) -> None:
    lines = [
        "# Baseline Training Summary",
        "",
        f"Project: {spec['project_name']}",
        f"Task: {spec['task']['type']}",
        f"Target: {spec['data']['target_column']}",
        f"Model: {model_name}",
        f"Split: {split_name}",
        f"Primary metric: {primary_metric} = {primary_value:.6f}",
        f"Best epoch: {train_info['best_epoch']}",
        "",
        "Validation metrics:",
    ]
    for key, value in metrics["validation"].items():
        lines.append(f"- {key}: {value:.6f}")
    lines.append("")
    lines.append("Test metrics:")
    for key, value in metrics["test"].items():
        lines.append(f"- {key}: {value:.6f}")
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def train_project(args: argparse.Namespace) -> int:
    root = Path(args.project_dir).expanduser().resolve()
    spec = load_json(root / "specs" / "project.json")
    model_config = load_model_config(root)
    trainer = model_config.get("trainer", "tabular_mlp")
    if trainer not in {"tabular_mlp", "tabm"}:
        raise SystemExit("The current baseline runner supports `tabular_mlp` and `tabm`.")
    config = build_train_config(root, args)
    set_seed(config.seed)

    data_path = root / spec["data"]["primary_table"]
    df = read_table(data_path)
    target_column = spec["data"]["target_column"]
    task_type = spec["task"]["type"]
    if task_type == "screening":
        raise SystemExit(
            "Screening projects are defined in the folder format, but the baseline trainer "
            "does not yet implement ranking metrics. Use task type `regression` or "
            "`classification` for the first baseline."
        )
    if df[target_column].isna().any():
        before = len(df)
        df = df.dropna(subset=[target_column]).copy()
        print(f"Dropped {before - len(df)} rows with missing target values.")

    train_df, val_df, test_df, split_name = split_dataframe(df, spec, config.seed)
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise SystemExit("Train/validation/test split produced an empty split. Adjust specs/project.json.")

    data_dictionary = load_data_dictionary(root)
    numeric_columns, categorical_columns = choose_feature_columns(df, spec, data_dictionary, model_config)
    train_y, val_y, test_y, label_to_id = prepare_targets(train_df, val_df, test_df, target_column, task_type)

    if not numeric_columns and not categorical_columns:
        raise SystemExit("No usable features were found. Mark feature columns in specs/data_dictionary.csv.")

    output_dim = 1 if task_type in {"regression", "screening"} else len(label_to_id or {})
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    print(f"Training baseline on {device}.")
    print(f"Rows: train={len(train_df)}, validation={len(val_df)}, test={len(test_df)}")

    if trainer == "tabm":
        transform = fit_tabular_transform(train_df, numeric_columns, categorical_columns)
        train_num, train_cat, feature_names, cat_cardinalities = transform_tabular_features(
            train_df, numeric_columns, categorical_columns, transform
        )
        val_num, val_cat, _, _ = transform_tabular_features(val_df, numeric_columns, categorical_columns, transform)
        test_num, test_cat, _, _ = transform_tabular_features(test_df, numeric_columns, categorical_columns, transform)
        model = build_tabm_model(train_num.shape[1], cat_cardinalities, output_dim, model_config)
        print(
            f"Features: numeric={len(numeric_columns)}, categorical={len(categorical_columns)}, "
            f"tabm_num={train_num.shape[1]}, tabm_cat={train_cat.shape[1]}"
        )
        train_info = train_tabm_loop(
            model, train_num, train_cat, train_y, val_num, val_cat, val_y, task_type, config, device
        )
        validation_metrics, validation_outputs = evaluate_tabm(model, val_num, val_cat, val_y, task_type, device)
        test_metrics, test_outputs = evaluate_tabm(model, test_num, test_cat, test_y, task_type, device)
        model_name = "tabm"
        checkpoint_extra = {
            "n_num_features": train_num.shape[1],
            "cat_cardinalities": cat_cardinalities,
            "tabm_config": model_config.get("tabm", {}),
        }
    else:
        transform = fit_feature_transform(train_df, numeric_columns, categorical_columns)
        train_x, feature_names = transform_features(train_df, numeric_columns, categorical_columns, transform)
        val_x, _ = transform_features(val_df, numeric_columns, categorical_columns, transform)
        test_x, _ = transform_features(test_df, numeric_columns, categorical_columns, transform)
        model = MLP(train_x.shape[1], output_dim, config.hidden_dim, config.dropout)
        print(
            f"Features: numeric={len(numeric_columns)}, categorical={len(categorical_columns)}, "
            f"encoded={train_x.shape[1]}"
        )
        train_info = train_loop(model, train_x, train_y, val_x, val_y, task_type, config, device)
        validation_metrics, validation_outputs = evaluate_model(model, val_x, val_y, task_type, device)
        test_metrics, test_outputs = evaluate_model(model, test_x, test_y, task_type, device)
        model_name = "tabular_mlp"
        checkpoint_extra = {
            "input_dim": train_x.shape[1],
            "hidden_dim": config.hidden_dim,
        }

    primary_metric = spec["evaluation"]["primary_metric"]
    primary_value = choose_primary(validation_metrics, primary_metric)

    run_id = args.run_id or utc_run_id()
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "primary_metric": primary_metric,
        "primary_value": primary_value,
        "metric_direction": "higher_is_better" if metric_is_better(primary_metric) else "lower_is_better",
        "validation": validation_metrics,
        "test": test_metrics,
        "train_info": train_info,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (run_dir / "features.json").write_text(
        json.dumps(
            {
                "numeric_columns": numeric_columns,
                "categorical_columns": categorical_columns,
                "encoded_feature_names": feature_names,
                "trainer": trainer,
                "model_config": model_config,
                "config": asdict(config),
            },
            indent=2,
        )
        + "\n"
    )
    torch.save(
        {
            "trainer": trainer,
            "model_state_dict": model.state_dict(),
            "output_dim": output_dim,
            "task_type": task_type,
            "label_to_id": label_to_id,
            **checkpoint_extra,
        },
        run_dir / "model.pt",
    )
    write_predictions(
        run_dir / "validation_predictions.csv",
        val_df,
        spec["data"].get("id_column"),
        target_column,
        task_type,
        validation_outputs,
        label_to_id,
    )
    write_predictions(
        run_dir / "test_predictions.csv",
        test_df,
        spec["data"].get("id_column"),
        target_column,
        task_type,
        test_outputs,
        label_to_id,
    )
    write_summary(run_dir / "summary.md", spec, train_info, metrics, primary_metric, primary_value, split_name, model_name)
    append_experiment_log(
        root,
        run_id,
        primary_metric,
        primary_value,
        split_name,
        model_name,
        f"{model_name} baseline with {len(feature_names)} input columns",
    )

    print("---")
    print(f"run_id:          {run_id}")
    print(f"primary_metric:  {primary_metric}")
    print(f"primary_value:   {primary_value:.6f}")
    for key, value in validation_metrics.items():
        print(f"val_{key}:       {value:.6f}")
    for key, value in test_metrics.items():
        print(f"test_{key}:      {value:.6f}")
    print(f"run_dir:         {run_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Materials Autoresearch baseline")
    parser.add_argument("project_dir", help="Materials Autoresearch project folder")
    parser.add_argument("--run-id", help="Optional run folder name")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even when CUDA is available")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return train_project(args)


if __name__ == "__main__":
    raise SystemExit(main())
