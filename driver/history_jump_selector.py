"""Train a small supervised selector over recorded history-jump branches.

The selector is deliberately offline and conservative: it predicts the final
worst-horizon loss delta of a candidate jump, while ``noop`` is always
available at zero delta. This tests whether the saved telemetry contains
regime information before introducing online learning or imagined rollouts.
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
from torch import nn
from torch.nn import functional as F


ROLES = ("attention", "embedding", "head", "mlp", "norm")
SCALARS = (
    "step",
    "loss",
    "validation_loss",
    "loss_slope",
    "loss_second_difference",
    "gradient_norm",
    "update_norm",
    "momentum_alignment",
    "parameter_norm",
    "adam_momentum_norm",
    "adam_variance_norm",
    "data_cursor",
)
NESTED = (
    ("gradient", "role_gradient_norms"),
    ("update", "role_update_norms"),
    ("parameter", "role_parameter_norms"),
    ("momentum", "adam_momentum_norms"),
    ("variance", "adam_variance_norms"),
)


@dataclass(frozen=True)
class Example:
    case_id: str
    landscape: str
    parent_step: int
    horizon: int
    blend: float
    features: dict[str, float]
    final_delta: float
    immediate_delta: float
    recovery_delta: float

    @property
    def safe_delta(self) -> float:
        return max(self.immediate_delta, self.recovery_delta, self.final_delta)


class Predictor(nn.Module):
    def __init__(self, inputs: int, hidden: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(inputs, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, max(32, hidden // 2)),
            nn.GELU(),
            nn.Linear(max(32, hidden // 2), 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _feature_map(row: dict[str, Any], horizon: int, blend: float) -> dict[str, float]:
    features = {
        "step": _finite(row.get("step")),
        "loss": _finite(row.get("loss")),
        "validation_loss": _finite(row.get("validation_loss"), _finite(row.get("loss"))),
        "loss_slope": _finite(row.get("loss_slope")),
        "loss_second_difference": _finite(row.get("loss_second_difference")),
        "gradient_norm": _finite(row.get("gradient_norm")),
        "update_norm": _finite(row.get("update_norm")),
        "momentum_alignment": _finite(row.get("momentum_alignment")),
        "parameter_norm": _finite(row.get("parameter_norm")),
        "adam_momentum_norm": _finite(row.get("adam_momentum_norm")),
        "adam_variance_norm": _finite(row.get("adam_variance_norm")),
        "data_cursor": _finite(row.get("data_cursor")),
        "action_horizon": float(horizon),
        "action_blend": float(blend),
    }
    for prefix, key in NESTED:
        values = row.get(key, {})
        if not isinstance(values, dict):
            values = {}
        for role in ROLES:
            features[f"{prefix}_{role}"] = _finite(values.get(role))
    return features


def _landscape(case_id: str) -> str:
    return case_id.rsplit("-", 1)[0]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_examples(
    result_dirs: list[Path], trace_dirs: list[Path] | None = None
) -> list[Example]:
    examples: list[Example] = []
    for index, result_dir in enumerate(result_dirs):
        result_manifest = json.loads(
            (result_dir / "manifest.json").read_text(encoding="utf-8")
        )
        trace_root = (
            trace_dirs[index]
            if trace_dirs is not None
            else Path(result_manifest["source"])
        )
        if not trace_root.exists():
            trace_root = result_dir.parent / trace_root.name
        branches = _read_jsonl(result_dir / result_manifest["files"]["branches"])
        traces: dict[str, dict[int, dict[str, Any]]] = {}
        for branch in branches:
            case_id = str(branch["case_id"])
            if case_id in traces:
                continue
            trace_path = trace_root / "cases" / case_id / "trace.jsonl"
            traces[case_id] = {
                int(row["step"]): row for row in _read_jsonl(trace_path)
            }
        grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
        for branch in branches:
            if not branch.get("failed"):
                grouped[
                    (str(branch["case_id"]), int(branch["parent_step"]), int(branch["horizon"]))
                ].append(branch)
        for (case_id, parent_step, horizon), rows in grouped.items():
            noop = next((row for row in rows if row["action"] == "noop"), None)
            parent = traces.get(case_id, {}).get(parent_step)
            if noop is None or parent is None:
                continue
            for row in rows:
                if row["action"] != "momentum_jump":
                    continue
                examples.append(
                    Example(
                        case_id=case_id,
                        landscape=_landscape(case_id),
                        parent_step=parent_step,
                        horizon=horizon,
                        blend=float(row["blend"]),
                        features=_feature_map(parent, horizon, float(row["blend"])),
                        final_delta=float(row["final_loss"] - noop["final_loss"]),
                        immediate_delta=float(
                            row["immediate_loss"] - noop["immediate_loss"]
                        ),
                        recovery_delta=float(
                            row["recovery_loss"] - noop["recovery_loss"]
                        ),
                    )
                )
    if not examples:
        raise ValueError("no valid jump examples found")
    return examples


def _feature_names(examples: list[Example]) -> list[str]:
    names = sorted({name for example in examples for name in example.features})
    return names


def _matrix(examples: list[Example], names: list[str]) -> torch.Tensor:
    return torch.tensor(
        [[example.features.get(name, 0.0) for name in names] for example in examples],
        dtype=torch.float32,
    )


def _standardize(
    train: torch.Tensor, test: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = train.mean(dim=0)
    scale = train.std(dim=0, unbiased=False).clamp_min(1e-6)
    return (train - mean) / scale, (test - mean) / scale, mean, scale


def _support_distances(
    train_examples: list[Example],
    test_examples: list[Example],
    names: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return leave-one-case-out calibration and test nearest-neighbor distances."""
    cases = sorted({example.case_id for example in train_examples})
    if len(cases) < 2:
        raise ValueError("support calibration needs at least two training cases")
    calibration: list[torch.Tensor] = []
    for case in cases:
        reference = [example for example in train_examples if example.case_id != case]
        held_out = [example for example in train_examples if example.case_id == case]
        reference_values = _matrix(reference, names)
        held_out_values = _matrix(held_out, names)
        mean = reference_values.mean(dim=0)
        scale = reference_values.std(dim=0, unbiased=False).clamp_min(1e-6)
        calibration.append(
            torch.cdist(
                (held_out_values - mean) / scale,
                (reference_values - mean) / scale,
            ).min(dim=1).values
        )
    train_values = _matrix(train_examples, names)
    test_values = _matrix(test_examples, names)
    mean = train_values.mean(dim=0)
    scale = train_values.std(dim=0, unbiased=False).clamp_min(1e-6)
    test_distances = torch.cdist(
        (test_values - mean) / scale,
        (train_values - mean) / scale,
    ).min(dim=1).values
    return torch.cat(calibration), test_distances


def _group_key(example: Example) -> tuple[str, int, int]:
    return example.case_id, example.parent_step, example.horizon


def _predict(model: Predictor, values: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(values)


def _selector_report(
    examples: list[Example],
    predictions: torch.Tensor,
    decision_predictions: torch.Tensor,
    train_examples: list[Example],
    risk_radius: float,
) -> dict[str, Any]:
    indexed = [
        {
            "example": example,
            "prediction": float(prediction),
            "decision": float(decision),
        }
        for example, prediction, decision in zip(
            examples, predictions.tolist(), decision_predictions.tolist()
        )
    ]
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for item in indexed:
        groups[_group_key(item["example"])].append(item)
    train_action_mean: dict[tuple[int, float], float] = defaultdict(list)
    for example in train_examples:
        train_action_mean[(example.horizon, example.blend)].append(example.safe_delta)
    fixed_scores = {
        key: sum(values) / len(values) for key, values in train_action_mean.items()
    }
    selected: list[float] = []
    fixed: list[float] = []
    oracle: list[float] = []
    noop: list[float] = []
    selected_safe: list[float] = []
    fixed_safe: list[float] = []
    oracle_safe: list[float] = []
    selected_jump = 0
    top1 = 0
    for candidates in groups.values():
        predicted = min(candidates, key=lambda item: item["decision"])
        chosen_prediction = min(0.0, predicted["decision"])
        if chosen_prediction < -risk_radius:
            selected.append(predicted["example"].final_delta)
            selected_safe.append(predicted["example"].safe_delta)
            selected_jump += 1
        else:
            selected.append(0.0)
            selected_safe.append(0.0)
        fixed_action = min(
            candidates,
            key=lambda item: fixed_scores.get(
                (item["example"].horizon, item["example"].blend), math.inf
            ),
        )
        fixed_score = fixed_scores.get(
            (fixed_action["example"].horizon, fixed_action["example"].blend), math.inf
        )
        fixed.append(fixed_action["example"].final_delta if fixed_score < 0.0 else 0.0)
        fixed_safe.append(
            fixed_action["example"].safe_delta if fixed_score < 0.0 else 0.0
        )
        oracle.append(min(0.0, *(item["example"].final_delta for item in candidates)))
        oracle_safe.append(
            min(0.0, *(item["example"].safe_delta for item in candidates))
        )
        noop.append(0.0)
        actual_best = min(
            0.0, *(item["example"].safe_delta for item in candidates)
        )
        selected_actual = selected_safe[-1]
        if math.isclose(selected_actual, actual_best, rel_tol=0.0, abs_tol=1e-6):
            top1 += 1
    errors = [
        item["prediction"] - item["example"].safe_delta for item in indexed
    ]
    return {
        "target": "safe_delta",
        "groups": len(groups),
        "jump_examples": len(examples),
        "prediction_rmse": math.sqrt(sum(error * error for error in errors) / len(errors)),
        "selected_mean_final_delta": sum(selected) / len(selected),
        "fixed_mean_final_delta": sum(fixed) / len(fixed),
        "oracle_mean_final_delta": sum(oracle) / len(oracle),
        "selected_mean_safe_delta": sum(selected_safe) / len(selected_safe),
        "fixed_mean_safe_delta": sum(fixed_safe) / len(fixed_safe),
        "oracle_mean_safe_delta": sum(oracle_safe) / len(oracle_safe),
        "noop_mean_final_delta": sum(noop) / len(noop),
        "selected_jump_rate": selected_jump / len(selected),
        "selected_top1_rate": top1 / len(selected),
        "risk_radius": risk_radius,
    }


def _train(
    train_values: torch.Tensor,
    train_targets: torch.Tensor,
    *,
    hidden: int,
    epochs: int,
    device: torch.device,
    seed: int | None = None,
) -> Predictor:
    if seed is not None:
        torch.manual_seed(seed)
    model = Predictor(train_values.shape[1], hidden).to(device)
    values = train_values.to(device)
    targets = train_targets.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(values)
        loss = F.smooth_l1_loss(prediction, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("selector training produced non-finite loss")
        loss.backward()
        optimizer.step()
    return model


def _train_ensemble(
    train_values: torch.Tensor,
    train_targets: torch.Tensor,
    train_examples: list[Example],
    *,
    hidden: int,
    epochs: int,
    device: torch.device,
    size: int,
    seed: int,
) -> list[Predictor]:
    cases = sorted({example.case_id for example in train_examples})
    if size < 1 or len(cases) < 2:
        raise ValueError("an ensemble needs at least two training cases")
    models = []
    for member in range(size):
        generator = torch.Generator().manual_seed(seed + member)
        sampled = [
            cases[int(torch.randint(len(cases), (), generator=generator))]
            for _ in cases
        ]
        indices = [
            index
            for case in sampled
            for index, example in enumerate(train_examples)
            if example.case_id == case
        ]
        models.append(
            _train(
                train_values[indices],
                train_targets[indices],
                hidden=hidden,
                epochs=epochs,
                device=device,
                seed=seed + member,
            )
        )
    return models


def _predict_ensemble(
    models: list[Predictor], values: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    predictions = torch.stack([_predict(model, values.to(device)) for model in models])
    return predictions.mean(dim=0).cpu(), predictions.std(dim=0, unbiased=False).cpu()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    if args.ensemble_size < 1:
        raise ValueError("ensemble size must be positive")
    if args.uncertainty_multiplier < 0.0 or not math.isfinite(
        args.uncertainty_multiplier
    ):
        raise ValueError("uncertainty multiplier must be finite and non-negative")
    torch.manual_seed(args.seed)
    result_dirs = [Path(value) for value in args.results]
    trace_dirs = None
    if args.traces:
        if len(args.traces) != len(result_dirs):
            raise ValueError("--traces must be supplied once per --results directory")
        trace_dirs = [Path(value) for value in args.traces]
    examples = _load_examples(result_dirs, trace_dirs)
    holdout_landscapes = set(args.holdout_landscape or ())
    holdout_seeds = set(args.holdout_seed or ())
    if not holdout_landscapes and not holdout_seeds:
        holdout_landscapes = {"text_shard"}
    test_examples = [
        example
        for example in examples
        if example.landscape in holdout_landscapes
        or any(example.case_id.endswith(f"-{seed}") for seed in holdout_seeds)
    ]
    train_examples = [example for example in examples if example not in test_examples]
    if not train_examples or not test_examples:
        raise ValueError("whole-landscape split produced an empty train or test set")
    names = _feature_names(examples)
    train_values, test_values, mean, scale = _standardize(
        _matrix(train_examples, names), _matrix(test_examples, names)
    )
    train_targets = torch.tensor(
        [example.safe_delta for example in train_examples], dtype=torch.float32
    )
    test_targets = torch.tensor(
        [example.safe_delta for example in test_examples], dtype=torch.float32
    )
    support_calibration, support_distances = _support_distances(
        train_examples, test_examples, names
    )
    support_calibration_radius = float(
        torch.quantile(support_calibration, 0.95).item()
    )
    if args.support_radius is None:
        support_radius = support_calibration_radius
    else:
        if args.support_radius < 0.0 or not math.isfinite(args.support_radius):
            raise ValueError("support radius must be finite and non-negative")
        support_radius = args.support_radius
    device = torch.device(args.device)
    if args.ensemble_size == 1:
        model = _train(
            train_values,
            train_targets,
            hidden=args.hidden,
            epochs=args.epochs,
            device=device,
            seed=args.seed,
        )
        train_predictions = _predict(model, train_values.to(device)).cpu()
        predictions = _predict(model, test_values.to(device)).cpu()
        prediction_std = torch.zeros_like(predictions)
        models = [model]
    else:
        models = _train_ensemble(
            train_values,
            train_targets,
            train_examples,
            hidden=args.hidden,
            epochs=args.epochs,
            device=device,
            size=args.ensemble_size,
            seed=args.seed,
        )
        train_predictions, _ = _predict_ensemble(models, train_values, device)
        predictions, prediction_std = _predict_ensemble(models, test_values, device)
    residuals = (train_predictions - train_targets).abs()
    calibration_radius = float(torch.quantile(residuals, 0.9).item())
    if args.risk_radius is None:
        risk_radius = calibration_radius
    else:
        if args.risk_radius < 0.0 or not math.isfinite(args.risk_radius):
            raise ValueError("risk radius must be finite and non-negative")
        risk_radius = args.risk_radius
    decision_predictions = predictions + args.uncertainty_multiplier * prediction_std
    decision_predictions = decision_predictions.clone()
    decision_predictions[support_distances > support_radius] = float("inf")
    report = {
        "schema": "landscape-driver.history-jump-selector.v2",
        "target": "safe_delta",
        "results": args.results,
        "holdout_landscapes": sorted(holdout_landscapes),
        "holdout_seeds": sorted(holdout_seeds),
        "train_cases": sorted({example.case_id for example in train_examples}),
        "test_cases": sorted({example.case_id for example in test_examples}),
        "features": names,
        "hidden": args.hidden,
        "epochs": args.epochs,
        "ensemble_size": args.ensemble_size,
        "uncertainty_multiplier": args.uncertainty_multiplier,
        "device": str(device),
        "calibration_radius_90": calibration_radius,
        "support_calibration_radius_95": support_calibration_radius,
        "support_radius": support_radius,
        "test_support_rate": float(
            (support_distances <= support_radius).float().mean().item()
        ),
        "test_prediction_std_mean": float(prediction_std.mean().item()),
        "metrics": _selector_report(
            test_examples,
            predictions,
            decision_predictions,
            train_examples,
            risk_radius,
        ),
    }
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"choose a fresh --output; {output} is not empty")
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    payload = {
        "feature_names": names,
        "mean": mean,
        "scale": scale,
        "holdout_landscapes": sorted(holdout_landscapes),
        "holdout_seeds": sorted(holdout_seeds),
        "ensemble_size": args.ensemble_size,
    }
    if args.ensemble_size == 1:
        payload["model"] = models[0].state_dict()
    else:
        payload["models"] = [model.state_dict() for model in models]
    torch.save(payload, output / "predictor.pt")
    print(json.dumps(report, indent=2))
    return report


def _self_check() -> None:
    torch.manual_seed(0)
    model = Predictor(4, 16)
    values = torch.randn(3, 4)
    prediction = model(values)
    assert prediction.shape == (3,)
    assert torch.isfinite(prediction).all()
    print("history-jump selector self-check passed: numerical predictor")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", action="append")
    parser.add_argument("--traces", action="append")
    parser.add_argument("--output", default="runs/history-jump-selector")
    parser.add_argument("--holdout-landscape", action="append")
    parser.add_argument("--holdout-seed", action="append", type=int)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--ensemble-size", type=int, default=1)
    parser.add_argument("--uncertainty-multiplier", type=float, default=1.0)
    parser.add_argument("--support-radius", type=float)
    parser.add_argument("--risk-radius", type=float)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--self-check", action="store_true")
    return parser


def main(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.self_check:
        _self_check()
        return None
    if not args.results:
        raise ValueError("at least one --results directory is required")
    return run(args)


if __name__ == "__main__":
    main(build_parser().parse_args())
