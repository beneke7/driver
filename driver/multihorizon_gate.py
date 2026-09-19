"""Evaluate a fixed, fail-closed multi-horizon response-model gate.

The gate loads an existing ``response_atlas_model`` checkpoint. It never
trains, imagines outcomes, or changes the model; it only applies the frozen
policy in ``research/MULTIHORIZON_GATE_PREREGISTRATION.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch

from .response_atlas_model import (
    HORIZONS,
    Predictor,
    _examples,
    _finite,
    _load_groups,
    _matrix,
    _predict_by_group,
)


METRICS = ("wall_seconds", "flops", "tokens")
RISK_TOLERANCE = 1e-3
UNCERTAINTY_SCALE = 3.0


def _geomean(values: list[float]) -> float:
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _load_predictions(
    atlas: Path, model_path: Path, source_report: dict[str, Any]
) -> tuple[list[Any], list[Any], dict[tuple[str, str, str], tuple[float, float]]]:
    bundle = torch.load(model_path, map_location="cpu", weights_only=False)
    groups, actions = _load_groups(atlas, int(bundle["history"]))
    train_roots = set(source_report["train_roots"])
    test_roots = set(source_report["test_roots"])
    train_groups = [group for group in groups if group.root in train_roots]
    test_groups = [group for group in groups if group.root in test_roots]
    if not train_groups or not test_groups:
        raise ValueError("checkpoint report has an empty train or test split")
    examples = _examples(groups, actions, int(bundle["history"]))
    train_ids = {group.group_id for group in train_groups}
    train = [item for item in examples if item.group_id in train_ids]
    test = [item for item in examples if item.group_id not in train_ids]
    train_values = _matrix(train, "values")
    test_values = _matrix(test, "values")
    mean = bundle["feature_mean"]
    scale = bundle["feature_scale"]
    target_mean = bundle["target_mean"]
    target_scale = bundle["target_scale"]
    normalized_train = (train_values - mean) / scale
    normalized_test = (test_values - mean) / scale
    input_width = int(mean.numel())
    hidden = int(bundle["models"][0]["network.0.weight"].shape[0])
    test_predictions: list[torch.Tensor] = []
    for state in bundle["models"]:
        model = Predictor(input_width, hidden)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            test_predictions.append(
                model(normalized_test) * target_scale + target_mean
            )
    predictions = _predict_by_group(torch.stack(test_predictions), test)
    return train_groups, test_groups, predictions


def _risk_margin(
    predictions: dict[tuple[str, str, str], tuple[float, float]],
    key: tuple[str, str, str],
    risk_radius: dict[str, float],
) -> float:
    mean, standard_deviation = predictions[key]
    horizon = key[2]
    return mean + risk_radius[horizon] + UNCERTAINTY_SCALE * standard_deviation


def evaluate(atlas: Path, model_path: Path, source_report_path: Path) -> dict[str, Any]:
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    train_groups, test_groups, predictions = _load_predictions(
        atlas, model_path, source_report
    )
    all_groups, actions = _load_groups(atlas, int(source_report["history"]))
    train_ids = {group.group_id for group in train_groups}
    examples = _examples(all_groups, actions, int(source_report["history"]))
    priors = {
        (action, horizon): statistics.fmean(
            item.target
            for item in examples
            if item.group_id in train_ids
            and item.action == action
            and item.horizon == horizon
        )
        for action in actions
        for horizon in HORIZONS
    }
    risk_radius = {
        horizon: float(source_report["prediction"][horizon]["risk_radius"])
        for horizon in HORIZONS
    }
    support_radius = float(source_report["support_radius"])
    support_distances = source_report["support_distances"]
    selected_rows: list[dict[str, Any]] = []
    for group in test_groups:
        noop = group.rows.get("noop")
        if noop is None:
            raise ValueError(f"test group {group.group_id} has no noop")
        noop_cost = noop["cost"]
        supported = (
            group.group_id in support_distances
            and float(support_distances[group.group_id]) <= support_radius
        )
        qualified: list[tuple[float, str, dict[str, float]]] = []
        for action, row in group.rows.items():
            if action == "noop":
                continue
            if not all(
                float(row["cost"][metric]) <= float(noop_cost[metric]) + 1e-9
                for metric in METRICS
            ):
                continue
            margins = {
                horizon: _risk_margin(
                    predictions, (group.group_id, action, horizon), risk_radius
                )
                for horizon in HORIZONS
            }
            if (
                supported
                and margins["final"] < 0.0
                and margins["recovery"] <= RISK_TOLERANCE
                and margins["immediate"] <= RISK_TOLERANCE
            ):
                score = margins["final"] / max(1e-12, float(row["cost"]["wall_seconds"]))
                qualified.append((score, action, margins))
        selected_action = (
            min(qualified, key=lambda item: (item[0], item[1]))[1]
            if qualified
            else "noop"
        )
        selected_margins = next(
            (margins for _, action, margins in qualified if action == selected_action),
            {horizon: 0.0 for horizon in HORIZONS},
        )
        selected_prediction = {
            horizon: (
                0.0
                if selected_action == "noop"
                else predictions[(group.group_id, selected_action, horizon)][0]
            )
            for horizon in HORIZONS
        }
        actual = {
            action: {
                horizon: _finite(
                    row.get("outcomes", {}).get(horizon, {}).get(
                        "loss_delta_vs_noop"
                    ),
                    math.nan,
                )
                for horizon in HORIZONS
            }
            for action, row in group.rows.items()
        }
        selected = actual[selected_action]
        prior_candidates = [action for action in group.rows if action in actions]
        prior_action = min(
            prior_candidates, key=lambda action: priors[(action, "final")]
        )
        if priors[(prior_action, "final")] >= 0.0:
            prior_action = "noop"
        prior = actual[prior_action]
        oracle = {
            horizon: min(0.0, min(values[horizon] for values in actual.values()))
            for horizon in HORIZONS
        }
        selected_cost = group.rows[selected_action]["cost"]
        selected_rows.append(
            {
                "group": group.group_id,
                "supported": supported,
                "support_distance": support_distances.get(group.group_id),
                "qualified_actions": [action for _, action, _ in qualified],
                "selected_action": selected_action,
                "selected_prediction": selected_prediction,
                "selected_margins": selected_margins,
                "selected": selected,
                "prior_action": prior_action,
                "prior": prior,
                "oracle": oracle,
                "cost_speedup": {
                    metric: float(noop_cost[metric]) / float(selected_cost[metric])
                    for metric in METRICS
                },
                "recovery_violation": selected_action != "noop"
                and selected["recovery"] > RISK_TOLERANCE,
                "final_regression": selected["final"] > 0.0,
            }
        )

    ranking: dict[str, Any] = {}
    for horizon in HORIZONS:
        selected = [row["selected"][horizon] for row in selected_rows]
        prior = [row["prior"][horizon] for row in selected_rows]
        oracle = [row["oracle"][horizon] for row in selected_rows]
        ranking[horizon] = {
            "groups": len(selected_rows),
            "selected_mean_delta": statistics.fmean(selected),
            "prior_mean_delta": statistics.fmean(prior),
            "oracle_mean_delta": statistics.fmean(oracle),
            "selected_action_rate": statistics.fmean(
                row["selected_action"] != "noop" for row in selected_rows
            ),
            "supported_group_rate": statistics.fmean(
                row["supported"] for row in selected_rows
            ),
            "prediction_reality_gap": statistics.fmean(
                abs(
                    row["selected"][horizon]
                    - row["selected_prediction"][horizon]
                )
                for row in selected_rows
            ),
        }
    speedups = {
        metric: [row["cost_speedup"][metric] for row in selected_rows]
        for metric in METRICS
    }
    result = {
        "schema": "landscape-driver.multihorizon-gate.v1",
        "atlas": str(atlas),
        "model": str(model_path),
        "source_report": str(source_report_path),
        "uncertainty_scale": UNCERTAINTY_SCALE,
        "risk_tolerance": RISK_TOLERANCE,
        "train_groups": len(train_groups),
        "test_groups": len(test_groups),
        "recovery_violations": sum(row["recovery_violation"] for row in selected_rows),
        "final_regressions": sum(row["final_regression"] for row in selected_rows),
        "selected_speedup": {
            metric: {
                "geometric_mean": _geomean(values),
                "median": statistics.median(values),
            }
            for metric, values in speedups.items()
        },
        "ranking": ranking,
        "decisions": selected_rows,
    }
    return result


def _self_check() -> None:
    assert math.isclose(_geomean([1.0, 4.0]), 2.0)
    assert _risk_margin({("g", "a", "final"): (-0.01, 0.0)}, ("g", "a", "final"), {"final": 0.0, "recovery": 0.0, "immediate": 0.0}) < 0.0
    print("multihorizon_gate self-check: ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas")
    parser.add_argument("--model")
    parser.add_argument("--source-report")
    parser.add_argument("--output")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        _self_check()
        return
    if not all((args.atlas, args.model, args.source_report, args.output)):
        parser.error("--atlas, --model, --source-report, and --output are required")
    result = evaluate(Path(args.atlas), Path(args.model), Path(args.source_report))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "ranking": result["ranking"]}, indent=2))


if __name__ == "__main__":
    main()
