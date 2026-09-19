"""Train a conservative action selector over history-enabled decoder branches.

The selector predicts a candidate's final validation-loss delta relative to the
matched no-op branch.  No-op remains legal when the prediction is not safely
better than zero.  This is an offline ranking probe, not an online policy.
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
from .trajectory_world_model import FEATURE_NAMES, _features


ACTIONS = (
    "trajectory_average",
    "trajectory_extrapolate",
    "trajectory_shadow_average",
)


@dataclass(frozen=True)
class Example:
    root: str
    group: tuple[str, int, int]
    action: str
    values: tuple[float, ...]
    target: float


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _history_vector(
    history: list[dict[str, Any]], parent_step: int, action: str, horizon: int
) -> tuple[float, ...]:
    if len(history) < 1:
        raise ValueError("parent history must contain at least one row")
    history = history[-16:]
    values: list[float] = []
    total = max(1, parent_step)
    for row in history:
        step = max(0, int(_finite(row.get("step"))))
        values.extend(_features(row, step, total))
    values.extend(
        (
            parent_step / max(1, parent_step + horizon),
            horizon / max(1, parent_step + horizon),
        )
    )
    values.extend(float(action == candidate) for candidate in ACTIONS)
    return tuple(values)


def _load_examples(paths: list[Path], history: int) -> list[Example]:
    if history != 16:
        raise ValueError("the persisted campaign contract currently stores 16 parent rows")
    examples: list[Example] = []
    for result in paths:
        rows = _read_jsonl(result / "transitions.jsonl")
        noop = next((row for row in rows if row["action"]["kind"] == "noop"), None)
        if noop is None or noop.get("after") is None:
            continue
        parent_step = int(noop["before"]["step"])
        horizon = int(noop["after"]["step"]) - parent_step
        for row in rows:
            action = row["action"]["kind"]
            if action not in ACTIONS or row.get("after") is None:
                continue
            parent_history = row.get("metadata", {}).get("parent_history")
            if not isinstance(parent_history, list) or len(parent_history) < history:
                continue
            values = _history_vector(parent_history[-history:], parent_step, action, horizon)
            examples.append(
                Example(
                    root=result.name,
                    group=(result.name, parent_step, horizon),
                    action=action,
                    values=values,
                    target=float(row["after"]["loss"] - noop["after"]["loss"]),
                )
            )
    if not examples:
        raise ValueError("no history-enabled action examples found")
    return examples


def _matrix(examples: list[Example]) -> torch.Tensor:
    return torch.tensor([example.values for example in examples], dtype=torch.float32)


def _fit(
    train_values: torch.Tensor,
    train_targets: torch.Tensor,
    test_values: torch.Tensor,
    *,
    hidden: int,
    epochs: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Predictor]:
    if hidden < 1 or epochs < 1:
        raise ValueError("hidden and epochs must be positive")
    mean = train_values.mean(dim=0)
    scale = train_values.std(dim=0, unbiased=False).clamp_min(1e-6)
    target_mean = train_targets.mean()
    target_scale = train_targets.std(unbiased=False).clamp_min(1e-6)
    train_values = ((train_values - mean) / scale).to(device)
    test_values = ((test_values - mean) / scale).to(device)
    normalized_targets = ((train_targets - target_mean) / target_scale).to(device)
    torch.manual_seed(seed)
    model = Predictor(train_values.shape[1], hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(train_values)
        loss = torch.nn.functional.smooth_l1_loss(prediction, normalized_targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("action selector training produced non-finite loss")
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        train_prediction = model(train_values).cpu() * target_scale + target_mean
        test_prediction = model(test_values).cpu() * target_scale + target_mean
    return train_prediction, test_prediction, mean, scale, model


def _support_radius(
    train_values: torch.Tensor,
    train_roots: list[str],
    test_values: torch.Tensor,
    requested: float | None,
) -> tuple[float, torch.Tensor]:
    mean = train_values.mean(dim=0)
    scale = train_values.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized_train = (train_values - mean) / scale
    normalized_test = (test_values - mean) / scale
    roots = sorted(set(train_roots))
    if len(roots) < 2:
        raise ValueError("support calibration needs at least two training roots")
    calibration: list[torch.Tensor] = []
    for root in roots:
        held = torch.tensor([item == root for item in train_roots], dtype=torch.bool)
        reference = normalized_train[~held]
        calibration.append(torch.cdist(normalized_train[held], reference).min(dim=1).values)
    default = float(torch.quantile(torch.cat(calibration), 0.95).item())
    radius = default if requested is None else float(requested)
    if radius < 0.0 or not math.isfinite(radius):
        raise ValueError("support radius must be finite and non-negative")
    distances = torch.cdist(normalized_test, normalized_train).min(dim=1).values
    return radius, distances


def _selection_report(
    test_examples: list[Example],
    predictions: torch.Tensor,
    train_examples: list[Example],
    risk_radius: float,
    supported_groups: dict[tuple[str, int, int], bool],
    fallback_action: str,
) -> dict[str, Any]:
    grouped: dict[tuple[str, int, int], list[tuple[Example, float]]] = defaultdict(list)
    for example, prediction in zip(test_examples, predictions.tolist()):
        grouped[example.group].append((example, float(prediction)))
    prior = {
        action: sum(example.target for example in train_examples if example.action == action)
        / max(1, len([example for example in train_examples if example.action == action]))
        for action in ACTIONS
    }
    selected: list[float] = []
    prior_values: list[float] = []
    oracle: list[float] = []
    decisions: list[dict[str, Any]] = []
    for group, candidates in grouped.items():
        predicted = min(candidates, key=lambda item: item[1] + risk_radius)
        predicted_score = predicted[1] + risk_radius
        supported = supported_groups.get(group, False)
        fallback = next(
            (item for item in candidates if item[0].action == fallback_action), None
        )
        chosen = (
            predicted
            if supported and predicted_score < 0.0
            else fallback
            if fallback is not None
            else (None, 0.0)
        )
        selected_value = chosen[0].target if chosen[0] is not None else 0.0
        prior_action = min(ACTIONS, key=lambda action: prior[action])
        prior_value = prior[prior_action] if prior[prior_action] < 0.0 else 0.0
        oracle_value = min(0.0, *(example.target for example, _ in candidates))
        selected.append(selected_value)
        prior_values.append(prior_value)
        oracle.append(oracle_value)
        decisions.append(
            {
                "group": group,
                "predicted_action": predicted[0].action,
                "predicted_delta": predicted[1],
                "risk_adjusted_delta": predicted_score,
                "selected_action": chosen[0].action if chosen[0] is not None else "noop",
                "selected_actual_delta": selected_value,
                "supported": supported,
                "prior_action": prior_action if prior_value < 0.0 else "noop",
                "oracle_delta": oracle_value,
            }
        )
    return {
        "groups": len(grouped),
        "selected_mean_delta": sum(selected) / max(1, len(selected)),
        "prior_mean_delta": sum(prior_values) / max(1, len(prior_values)),
        "oracle_mean_delta": sum(oracle) / max(1, len(oracle)),
        "selected_action_rate": sum(item["selected_action"] != "noop" for item in decisions)
        / max(1, len(decisions)),
        "decisions": decisions,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.results or not args.holdout_root:
        raise ValueError("results and at least one holdout root are required")
    examples = _load_examples([Path(value) for value in args.results], args.history)
    holdouts = set(args.holdout_root)
    train = [example for example in examples if example.root not in holdouts]
    test = [example for example in examples if example.root in holdouts]
    if not train or not test:
        raise ValueError("holdout split produced an empty train or test set")
    train_values = _matrix(train)
    test_values = _matrix(test)
    train_targets = torch.tensor([example.target for example in train])
    test_targets = torch.tensor([example.target for example in test])
    device = torch.device(args.device)
    train_prediction, prediction, mean, scale, model = _fit(
        train_values,
        train_targets,
        test_values,
        hidden=args.hidden,
        epochs=args.epochs,
        device=device,
        seed=args.seed,
    )
    residual = (train_prediction - train_targets).abs()
    risk_radius = (
        float(torch.quantile(residual, 0.9).item())
        if args.risk_radius is None
        else float(args.risk_radius)
    )
    if risk_radius < 0.0 or not math.isfinite(risk_radius):
        raise ValueError("risk radius must be finite and non-negative")
    support_radius, support_distances = _support_radius(
        train_values,
        [example.root for example in train],
        test_values,
        args.support_radius,
    )
    supported_groups: dict[tuple[str, int, int], bool] = {}
    for index, example in enumerate(test):
        supported_groups[example.group] = supported_groups.get(example.group, True) and bool(
            support_distances[index] <= support_radius
        )
    report = {
        "schema": "landscape-driver.trajectory-action-selector.v1",
        "results": args.results,
        "holdout_roots": sorted(holdouts),
        "train_roots": sorted({example.root for example in train}),
        "test_roots": sorted({example.root for example in test}),
        "history": args.history,
        "hidden": args.hidden,
        "epochs": args.epochs,
        "device": str(device),
        "examples": len(examples),
        "train_examples": len(train),
        "test_examples": len(test),
        "prediction_rmse": float((prediction - test_targets).square().mean().sqrt()),
        "calibration_radius_90": risk_radius,
        "support_radius_95": support_radius,
        "test_support_rate": float(
            sum(supported_groups.values()) / max(1, len(supported_groups))
        ),
        "fallback_action": args.fallback_action,
        "metrics": _selection_report(
            test,
            prediction,
            train,
            risk_radius,
            supported_groups,
            args.fallback_action,
        ),
    }
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"choose a fresh output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    torch.save(
        {
            "model": model.state_dict(),
            "mean": mean,
            "scale": scale,
            "actions": ACTIONS,
            "history": args.history,
        },
        output / "selector.pt",
    )
    print(json.dumps(report, indent=2))
    return report


def _self_check() -> None:
    row = {name: 1.0 for name in ("loss", "gradient_norm", "update_norm")}
    values = _features(row, 1, 16)
    assert len(values) == len(FEATURE_NAMES)
    model = Predictor(len(values) * 16 + 5, 8)
    prediction = model(torch.zeros(1, len(values) * 16 + 5))
    assert prediction.shape == (1,)
    assert torch.isfinite(prediction).all()
    print("trajectory action-selector self-check passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", action="append", default=[])
    parser.add_argument("--holdout-root", action="append", default=[])
    parser.add_argument("--output", default="runs/trajectory-action-selector")
    parser.add_argument("--history", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--risk-radius", type=float)
    parser.add_argument("--support-radius", type=float)
    parser.add_argument("--fallback-action", choices=("noop", *ACTIONS), default="noop")
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
