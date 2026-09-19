"""Fit a small history-conditioned predictor over decoder telemetry."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .history_jump_selector import Predictor


HISTORY_FIELDS = (
    "loss",
    "gradient_norm",
    "update_norm",
    "momentum_alignment",
    "parameter_norm",
    "adam_momentum_norm",
    "adam_variance_norm",
    "global_lr_multiplier",
)
FEATURE_NAMES = (
    "step_fraction",
    "loss",
    "log_gradient_norm",
    "log_update_norm",
    "momentum_alignment",
    "log_parameter_norm",
    "log_adam_momentum_norm",
    "log_adam_variance_norm",
    "relative_gradient_norm",
    "relative_update_norm",
    "relative_momentum_norm",
)


@dataclass(frozen=True)
class Example:
    root: str
    current: tuple[float, ...]
    history: tuple[float, ...]
    target: float


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _features(row: dict[str, Any], index: int, total: int) -> tuple[float, ...]:
    parameter_norm = max(_finite(row.get("parameter_norm")), 1e-6)
    gradient_norm = max(_finite(row.get("gradient_norm")), 0.0)
    update_norm = max(_finite(row.get("update_norm")), 0.0)
    momentum_norm = max(_finite(row.get("adam_momentum_norm")), 0.0)
    variance_norm = max(_finite(row.get("adam_variance_norm")), 0.0)
    return (
        index / max(1, total - 1),
        _finite(row.get("loss")),
        math.log1p(gradient_norm),
        math.log1p(update_norm),
        _finite(row.get("momentum_alignment")),
        math.log1p(parameter_norm),
        math.log1p(momentum_norm),
        math.log1p(variance_norm),
        gradient_norm / parameter_norm,
        update_norm / parameter_norm,
        momentum_norm / parameter_norm,
    )


def _read_examples(paths: list[Path], history: int) -> list[Example]:
    if history < 1:
        raise ValueError("history must be positive")
    examples: list[Example] = []
    for result in paths:
        root = result.name
        transitions = result / "transitions.jsonl"
        if not transitions.is_file():
            raise FileNotFoundError(f"missing transitions.jsonl: {transitions}")
        for line in transitions.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            telemetry = row.get("metadata", {}).get("telemetry")
            if not isinstance(telemetry, list) or len(telemetry) <= history:
                continue
            values = [_features(item, index, len(telemetry)) for index, item in enumerate(telemetry)]
            for index in range(history - 1, len(values) - 1):
                target = _finite(telemetry[index + 1].get("loss")) - _finite(
                    telemetry[index].get("loss")
                )
                current = values[index]
                sequence = tuple(value for row_values in values[index - history + 1 : index + 1] for value in row_values)
                if math.isfinite(target):
                    examples.append(Example(root, current, sequence, target))
    if not examples:
        raise ValueError("no telemetry examples found")
    return examples


def _matrix(examples: list[Example], field: str) -> torch.Tensor:
    return torch.tensor([getattr(example, field) for example in examples], dtype=torch.float32)


def _correlation(prediction: torch.Tensor, target: torch.Tensor) -> float | None:
    if prediction.numel() < 2 or prediction.std() == 0 or target.std() == 0:
        return None
    return float(torch.corrcoef(torch.stack((prediction, target)))[0, 1].item())


def _fit(
    train_values: torch.Tensor,
    train_targets: torch.Tensor,
    test_values: torch.Tensor,
    *,
    hidden: int,
    epochs: int,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    if hidden < 1 or epochs < 1:
        raise ValueError("hidden and epochs must be positive")
    feature_mean = train_values.mean(dim=0)
    feature_scale = train_values.std(dim=0, unbiased=False).clamp_min(1e-6)
    target_mean = train_targets.mean()
    target_scale = train_targets.std(unbiased=False).clamp_min(1e-6)
    train_values = ((train_values - feature_mean) / feature_scale).to(device)
    train_targets = ((train_targets - target_mean) / target_scale).to(device)
    test_values = ((test_values - feature_mean) / feature_scale).to(device)
    torch.manual_seed(seed)
    model = Predictor(train_values.shape[1], hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(train_values)
        loss = torch.nn.functional.smooth_l1_loss(prediction, train_targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("world-model training produced non-finite loss")
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        prediction = model(test_values).cpu() * target_scale + target_mean
    payload = {
        "model": model.state_dict(),
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "target_mean": target_mean,
        "target_scale": target_scale,
    }
    return {"prediction": prediction}, payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    results = [Path(value) for value in args.results]
    holdouts = set(args.holdout_root)
    if not results or not holdouts:
        raise ValueError("results and at least one holdout root are required")
    examples = _read_examples(results, args.history)
    train = [example for example in examples if example.root not in holdouts]
    test = [example for example in examples if example.root in holdouts]
    if not train or not test:
        raise ValueError("holdout split produced an empty train or test set")
    train_targets = torch.tensor([example.target for example in train], dtype=torch.float32)
    test_targets = torch.tensor([example.target for example in test], dtype=torch.float32)
    current_train = _matrix(train, "current")
    current_test = _matrix(test, "current")
    history_train = _matrix(train, "history")
    history_test = _matrix(test, "history")
    device = torch.device(args.device)
    current_result, _ = _fit(
        current_train,
        train_targets,
        current_test,
        hidden=args.hidden,
        epochs=args.epochs,
        device=device,
        seed=args.seed,
    )
    history_result, payload = _fit(
        history_train,
        train_targets,
        history_test,
        hidden=args.hidden,
        epochs=args.epochs,
        device=device,
        seed=args.seed,
    )
    baseline = train_targets.mean()
    metrics = {
        "examples": len(examples),
        "train_examples": len(train),
        "test_examples": len(test),
        "train_roots": sorted({example.root for example in train}),
        "test_roots": sorted({example.root for example in test}),
        "baseline_rmse": float((test_targets - baseline).square().mean().sqrt()),
    }
    for name, result in (("current", current_result), ("history", history_result)):
        prediction = result["prediction"]
        metrics[f"{name}_rmse"] = float((prediction - test_targets).square().mean().sqrt())
        metrics[f"{name}_correlation"] = _correlation(prediction, test_targets)
    report = {
        "schema": "landscape-driver.trajectory-world-model.v1",
        "history": args.history,
        "hidden": args.hidden,
        "epochs": args.epochs,
        "device": str(device),
        "features": FEATURE_NAMES,
        "metrics": metrics,
    }
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"choose a fresh output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    payload.update({"history": args.history, "hidden": args.hidden, "features": FEATURE_NAMES})
    torch.save(payload, output / "world_model.pt")
    print(json.dumps(report, indent=2))
    return report


def _self_check() -> None:
    row = {name: 1.0 for name in HISTORY_FIELDS}
    values = _features(row, 1, 4)
    assert len(values) == 11
    model = Predictor(len(values) * 2, 8)
    prediction = model(torch.zeros(1, len(values) * 2))
    assert prediction.shape == (1,)
    assert torch.isfinite(prediction).all()
    print("trajectory world-model self-check passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", action="append", default=[])
    parser.add_argument("--holdout-root", action="append", default=[])
    parser.add_argument("--output", default="runs/trajectory-world-model")
    parser.add_argument("--history", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=400)
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
