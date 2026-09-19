"""Fit a small uncertainty-aware action-conditioned model over an atlas.

This is an offline prediction and ranking probe.  It never turns imagined
responses into evidence and keeps no-op as the deployment fallback when a
held-out state is outside the calibrated support region.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .history_jump_selector import Predictor


HORIZONS = ("immediate", "recovery", "final")
STATE_FIELDS = (
    "loss",
    "gradient_norm",
    "update_norm",
    "momentum_alignment",
    "parameter_norm",
    "adam_momentum_norm",
    "adam_variance_norm",
    "global_lr_multiplier",
)
HISTORY_FIELDS = STATE_FIELDS
ARCHITECTURE_FIELDS = ("width", "layers", "heads", "context")


@dataclass(frozen=True)
class Example:
    root: str
    group_id: str
    action: str
    horizon: str
    values: tuple[float, ...]
    state: tuple[float, ...]
    target: float


@dataclass(frozen=True)
class Group:
    root: str
    group_id: str
    rows: dict[str, dict[str, Any]]


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"atlas row must be an object: {path}")
            rows.append(value)
    return rows


def _state_vector(row: dict[str, Any], history: int) -> tuple[float, ...]:
    parent = row.get("parent", {})
    features = parent.get("features", {}) if isinstance(parent, dict) else {}
    case = row.get("case", {})
    if not isinstance(features, dict) or not isinstance(case, dict):
        raise ValueError("atlas row has invalid parent or case fields")
    values = [_finite(features.get(name)) for name in STATE_FIELDS]
    values.extend(_finite(case.get(name)) for name in ARCHITECTURE_FIELDS)
    raw_history = parent.get("history", [])
    if not isinstance(raw_history, list):
        raw_history = []
    selected = raw_history[-history:]
    padded = [{} for _ in range(max(0, history - len(selected)))] + selected
    for item in padded:
        if not isinstance(item, dict):
            item = {}
        values.extend(_finite(item.get(name)) for name in HISTORY_FIELDS)
    return tuple(values)


def _feature_vector(
    row: dict[str, Any],
    action: str,
    horizon: str,
    actions: tuple[str, ...],
    history: int,
) -> tuple[float, ...]:
    state = _state_vector(row, history)
    action_record = row.get("action", {})
    if not isinstance(action_record, dict):
        raise ValueError("atlas row has invalid action")
    values = list(state)
    values.extend(float(action == candidate) for candidate in actions)
    values.append(_finite(action_record.get("strength"), 0.0))
    values.extend(float(horizon == candidate) for candidate in HORIZONS)
    return tuple(values)


def _load_groups(atlas: Path, history: int) -> tuple[list[Group], tuple[str, ...]]:
    rows = _read_jsonl(atlas)
    grouped: dict[str, dict[str, Any]] = {}
    actions: set[str] = set()
    for row in rows:
        group_id = str(row.get("group_id", "")).strip()
        case = row.get("case", {})
        root = str(
            row.get("root_id")
            or (case.get("root_id", "") if isinstance(case, dict) else "")
            or row.get("case_id", "")
        ).strip()
        action_record = row.get("action", {})
        action = str(action_record.get("kind", "")).strip() if isinstance(action_record, dict) else ""
        if not group_id or not root or not action:
            raise ValueError("atlas rows need group_id, case_id, and action.kind")
        actions.add(action)
        entry = grouped.setdefault(group_id, {"root": root, "rows": {}})
        if entry["root"] != root:
            raise ValueError(f"group {group_id} changes case root")
        if action in entry["rows"]:
            raise ValueError(f"duplicate action {action!r} in group {group_id}")
        entry["rows"][action] = row
    if not grouped or "noop" not in actions:
        raise ValueError("atlas must contain at least one matched noop group")
    groups = [
        Group(root=value["root"], group_id=group_id, rows=value["rows"])
        for group_id, value in sorted(grouped.items())
    ]
    return groups, tuple(sorted(actions))


def _examples(
    groups: list[Group], actions: tuple[str, ...], history: int
) -> list[Example]:
    examples: list[Example] = []
    for group in groups:
        for action, row in group.rows.items():
            response = row.get("outcomes", {})
            if not isinstance(response, dict):
                continue
            for horizon in HORIZONS:
                item = response.get(horizon)
                if not isinstance(item, dict) or "loss_delta_vs_noop" not in item:
                    continue
                target = _finite(item.get("loss_delta_vs_noop"), math.nan)
                risk = row.get("risk", {})
                if not math.isfinite(target) or (
                    isinstance(risk, dict) and risk.get("failed", False)
                ):
                    continue
                examples.append(
                    Example(
                        root=group.root,
                        group_id=group.group_id,
                        action=action,
                        horizon=horizon,
                        values=_feature_vector(row, action, horizon, actions, history),
                        state=_state_vector(row, history),
                        target=target,
                    )
                )
    if not examples:
        raise ValueError("atlas contains no finite matched response examples")
    return examples


def _matrix(examples: list[Example], field: str) -> torch.Tensor:
    return torch.tensor([getattr(example, field) for example in examples], dtype=torch.float32)


def _fit_ensemble(
    train_values: torch.Tensor,
    train_targets: torch.Tensor,
    test_values: torch.Tensor,
    *,
    hidden: int,
    epochs: int,
    ensemble: int,
    device: torch.device,
    seed: int,
) -> tuple[
    list[Predictor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    if hidden < 1 or epochs < 1 or ensemble < 1:
        raise ValueError("hidden, epochs, and ensemble must be positive")
    mean = train_values.mean(dim=0)
    scale = train_values.std(dim=0, unbiased=False).clamp_min(1e-6)
    target_mean = train_targets.mean()
    target_scale = train_targets.std(unbiased=False).clamp_min(1e-6)
    normalized_train = ((train_values - mean) / scale).to(device)
    normalized_test = ((test_values - mean) / scale).to(device)
    normalized_targets = ((train_targets - target_mean) / target_scale).to(device)
    models: list[Predictor] = []
    predictions: list[torch.Tensor] = []
    for member in range(ensemble):
        torch.manual_seed(seed + member)
        model = Predictor(normalized_train.shape[1], hidden).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(normalized_train)
            loss = torch.nn.functional.smooth_l1_loss(prediction, normalized_targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("atlas model training produced non-finite loss")
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            predictions.append(model(normalized_test).cpu() * target_scale + target_mean)
        models.append(model.cpu())
    return models, torch.stack(predictions), mean, scale, target_mean, target_scale


def _quantile(values: torch.Tensor, probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError("quantile must be between zero and one")
    return float(torch.quantile(values, probability).item())


def _support_radius(
    train_examples: list[Example], test_groups: list[Group], history: int, quantile: float
) -> tuple[float, dict[str, float]]:
    roots = sorted({example.root for example in train_examples})
    if len(roots) < 2:
        raise ValueError("support calibration needs at least two training roots")
    train_states = _matrix(train_examples, "state")
    mean = train_states.mean(dim=0)
    scale = train_states.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized = (train_states - mean) / scale
    calibration: list[torch.Tensor] = []
    for root in roots:
        held = torch.tensor([item.root == root for item in train_examples], dtype=torch.bool)
        calibration.append(torch.cdist(normalized[held], normalized[~held]).min(dim=1).values)
    radius = _quantile(torch.cat(calibration), quantile)
    distances: dict[str, float] = {}
    for group in test_groups:
        state = torch.tensor([_state_vector(next(iter(group.rows.values())), history)])
        distances[group.group_id] = float(
            torch.cdist((state - mean) / scale, normalized).min().item()
        )
    return radius, distances


def _predict_by_group(
    model_predictions: torch.Tensor,
    test_examples: list[Example],
) -> dict[tuple[str, str, str], tuple[float, float]]:
    positions: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, example in enumerate(test_examples):
        positions[(example.group_id, example.action, example.horizon)].append(index)
    result: dict[tuple[str, str, str], tuple[float, float]] = {}
    for key, indices in positions.items():
        values = model_predictions[:, indices]
        result[key] = (float(values.mean().item()), float(values.std(unbiased=False).item()))
    return result


def _ranking(
    groups: list[Group],
    horizons: tuple[str, ...],
    predictions: dict[tuple[str, str, str], tuple[float, float]],
    priors: dict[tuple[str, str], float],
    risk_radius: dict[str, float],
    support_distances: dict[str, float],
    support_radius: float,
    supported_groups: set[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for horizon in horizons:
        selected: list[float] = []
        prior_values: list[float] = []
        oracle: list[float] = []
        gaps: list[float] = []
        decisions: list[dict[str, Any]] = []
        raw_top1 = 0
        for group in groups:
            actual = {
                action: _finite(
                    group.rows[action].get("outcomes", {}).get(horizon, {}).get(
                        "loss_delta_vs_noop"
                    ),
                    math.nan,
                )
                for action in group.rows
                if isinstance(group.rows[action].get("outcomes", {}).get(horizon), dict)
            }
            actual = {action: value for action, value in actual.items() if math.isfinite(value)}
            if "noop" not in actual:
                continue
            candidates = [action for action in actual if (group.group_id, action, horizon) in predictions]
            if not candidates:
                continue
            predicted_best = min(candidates, key=lambda action: predictions[(group.group_id, action, horizon)][0])
            raw_top1 += predicted_best == min(candidates, key=lambda action: actual[action])
            predicted = min(
                candidates,
                key=lambda action: predictions[(group.group_id, action, horizon)][0]
                + risk_radius[horizon]
                + predictions[(group.group_id, action, horizon)][1],
            )
            deployable = (
                group.group_id in supported_groups
                and predictions[(group.group_id, predicted, horizon)][0]
                + risk_radius[horizon]
                + predictions[(group.group_id, predicted, horizon)][1]
                < 0.0
            )
            chosen = predicted if deployable else "noop"
            prior_action = min(candidates, key=lambda action: priors.get((action, horizon), 0.0))
            prior_action = prior_action if priors.get((prior_action, horizon), 0.0) < 0.0 else "noop"
            selected_value = actual[chosen]
            prior_value = actual.get(prior_action, actual["noop"])
            oracle_value = min(0.0, min(actual.values()))
            selected.append(selected_value)
            prior_values.append(prior_value)
            oracle.append(oracle_value)
            gaps.append(abs(predictions[(group.group_id, chosen, horizon)][0] - selected_value))
            decisions.append(
                {
                    "group": group.group_id,
                    "predicted_action": predicted_best,
                    "selected_action": chosen,
                    "selected_actual_delta": selected_value,
                    "predicted_delta": predictions[(group.group_id, predicted_best, horizon)][0],
                    "prediction_uncertainty": predictions[(group.group_id, predicted_best, horizon)][1],
                    "supported": group.group_id in supported_groups,
                    "support_distance": support_distances.get(group.group_id),
                    "oracle_delta": oracle_value,
                    "prior_action": prior_action,
                    "prior_actual_delta": prior_value,
                }
            )
        result[horizon] = {
            "groups": len(decisions),
            "selected_mean_delta": sum(selected) / max(1, len(selected)),
            "prior_mean_delta": sum(prior_values) / max(1, len(prior_values)),
            "oracle_mean_delta": sum(oracle) / max(1, len(oracle)),
            "prediction_reality_gap": sum(gaps) / max(1, len(gaps)),
            "selected_action_rate": sum(
                item["selected_action"] != "noop" for item in decisions
            )
            / max(1, len(decisions)),
            "raw_top1_agreement": raw_top1 / max(1, len(decisions)),
            "supported_group_rate": sum(item["supported"] for item in decisions)
            / max(1, len(decisions)),
            "support_radius": support_radius,
            "decisions": decisions,
        }
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.atlas or not args.holdout_root:
        raise ValueError("atlas and at least one holdout root are required")
    groups, actions = _load_groups(Path(args.atlas), args.history)
    examples = _examples(groups, actions, args.history)
    holdouts = set(args.holdout_root)
    train_groups = [group for group in groups if group.root not in holdouts]
    test_groups = [group for group in groups if group.root in holdouts]
    if not train_groups or not test_groups:
        raise ValueError("holdout split produced an empty train or test set")
    train_ids = {group.group_id for group in train_groups}
    train = [example for example in examples if example.group_id in train_ids]
    test = [example for example in examples if example.group_id not in train_ids]
    if not train or not test:
        raise ValueError("holdout split produced no response examples")
    train_values = _matrix(train, "values")
    test_values = _matrix(test, "values")
    train_targets = torch.tensor([example.target for example in train], dtype=torch.float32)
    test_targets = torch.tensor([example.target for example in test], dtype=torch.float32)
    device = torch.device(args.device)
    models, member_predictions, mean, scale, target_mean, target_scale = _fit_ensemble(
        train_values,
        train_targets,
        test_values,
        hidden=args.hidden,
        epochs=args.epochs,
        ensemble=args.ensemble,
        device=device,
        seed=args.seed,
    )
    prediction = member_predictions.mean(dim=0)
    train_predictions: list[torch.Tensor] = []
    normalized_train = ((train_values - mean) / scale)
    for model in models:
        with torch.no_grad():
            train_predictions.append(model(normalized_train).cpu() * target_scale + target_mean)
    train_prediction = torch.stack(train_predictions).mean(dim=0)
    risk_radius = {
        horizon: _quantile(
            (train_prediction[[index for index, example in enumerate(train) if example.horizon == horizon]]
             - train_targets[[index for index, example in enumerate(train) if example.horizon == horizon]]).abs(),
            args.risk_quantile,
        )
        for horizon in HORIZONS
    }
    support_radius, support_distances = _support_radius(
        train, test_groups, args.history, args.support_quantile
    )
    supported_groups = {
        group_id for group_id, distance in support_distances.items() if distance <= support_radius
    }
    predictions = _predict_by_group(member_predictions, test)
    priors = {
        (action, horizon): sum(
            example.target
            for example in train
            if example.action == action and example.horizon == horizon
        )
        / max(1, sum(example.action == action and example.horizon == horizon for example in train))
        for action in actions
        for horizon in HORIZONS
    }
    metrics: dict[str, Any] = {}
    for horizon in HORIZONS:
        indices = [index for index, example in enumerate(test) if example.horizon == horizon]
        residual = prediction[indices] - test_targets[indices]
        metrics[horizon] = {
            "examples": len(indices),
            "rmse": float(residual.square().mean().sqrt().item()),
            "mae": float(residual.abs().mean().item()),
            "mean_prediction": float(prediction[indices].mean().item()),
            "mean_actual": float(test_targets[indices].mean().item()),
            "risk_radius": risk_radius[horizon],
        }
    report = {
        "schema": "landscape-driver.response-atlas-model.v1",
        "atlas": str(args.atlas),
        "actions": list(actions),
        "horizons": list(HORIZONS),
        "history": args.history,
        "hidden": args.hidden,
        "ensemble": args.ensemble,
        "epochs": args.epochs,
        "device": str(device),
        "holdout_roots": sorted(holdouts),
        "train_roots": sorted({group.root for group in train_groups}),
        "test_roots": sorted({group.root for group in test_groups}),
        "groups": len(groups),
        "train_groups": len(train_groups),
        "test_groups": len(test_groups),
        "examples": len(examples),
        "train_examples": len(train),
        "test_examples": len(test),
        "support_radius": support_radius,
        "support_distances": support_distances,
        "supported_group_rate": len(supported_groups) / max(1, len(test_groups)),
        "prediction": metrics,
        "ranking": _ranking(
            test_groups,
            HORIZONS,
            predictions,
            priors,
            risk_radius,
            support_distances,
            support_radius,
            supported_groups,
        ),
    }
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"choose a fresh output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    torch.save(
        {
            "models": [model.state_dict() for model in models],
            "feature_mean": mean,
            "feature_scale": scale,
            "target_mean": target_mean,
            "target_scale": target_scale,
            "actions": actions,
            "horizons": HORIZONS,
            "history": args.history,
        },
        output / "world_model.pt",
    )
    print(json.dumps(report, indent=2))
    return report


def _self_check() -> None:
    row = {
        "case": {"width": 768, "layers": 12, "heads": 12, "context": 128},
        "parent": {"features": {name: 1.0 for name in STATE_FIELDS}, "history": []},
        "action": {"kind": "noop", "strength": 1.0},
    }
    state = _state_vector(row, 2)
    values = _feature_vector(row, "noop", "final", ("noop", "pulse"), 2)
    assert len(state) == len(STATE_FIELDS) + len(ARCHITECTURE_FIELDS) + 2 * len(HISTORY_FIELDS)
    assert len(values) == len(state) + 2 + 1 + len(HORIZONS)
    model = Predictor(len(values), 8)
    prediction = model(torch.zeros(1, len(values)))
    assert prediction.shape == (1,)
    assert torch.isfinite(prediction).all()
    print("response-atlas-model self-check passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas")
    parser.add_argument("--holdout-root", action="append", default=[])
    parser.add_argument("--output", default="runs/response-atlas-model")
    parser.add_argument("--history", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--ensemble", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--risk-quantile", type=float, default=0.9)
    parser.add_argument("--support-quantile", type=float, default=0.95)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--self-check", action="store_true")
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    if arguments.self_check:
        _self_check()
    else:
        run(arguments)
