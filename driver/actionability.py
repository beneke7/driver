"""Derive preregistered actionability labels from a causal response atlas.

The labeler consumes only observed matched branches.  It does not train a
policy, fill missing outcomes, or turn passive data into intervention labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HORIZONS = ("immediate", "recovery", "final")
DEFAULT_TOLERANCE = 1e-3


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_atlas(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"atlas row {line_number} is not an object")
        rows.append(value)
    if not rows:
        raise ValueError("atlas has no rows")
    return rows


def _outcome_deltas(row: dict[str, Any]) -> dict[str, float] | None:
    outcomes = row.get("outcomes", {})
    if not isinstance(outcomes, dict):
        return None
    deltas: dict[str, float] = {}
    for horizon in HORIZONS:
        item = outcomes.get(horizon)
        if not isinstance(item, dict):
            return None
        value = _finite(item.get("loss_delta_vs_noop"))
        if value is None:
            return None
        deltas[horizon] = value
    return deltas


def _jump_horizon(
    row: dict[str, Any], noop: dict[str, Any], tolerance: float
) -> str | None:
    noop_outcomes = noop.get("outcomes", {})
    final = noop_outcomes.get("final", {}) if isinstance(noop_outcomes, dict) else {}
    noop_final_loss = _finite(final.get("loss")) if isinstance(final, dict) else None
    noop_cost = _finite((noop.get("cost") or {}).get("wall_seconds"))
    row_cost = row.get("cost") or {}
    prefix_wall = (_finite(row_cost.get("wall_seconds")) or 0.0) - (
        _finite(row_cost.get("branch_wall_seconds")) or 0.0
    )
    if noop_final_loss is None or noop_cost is None:
        return None
    outcomes = row.get("outcomes", {})
    for horizon in ("immediate", "recovery"):
        item = outcomes.get(horizon) if isinstance(outcomes, dict) else None
        if not isinstance(item, dict):
            continue
        loss = _finite(item.get("loss"))
        phase_wall = _finite(item.get("wall_seconds"))
        if loss is None or phase_wall is None:
            continue
        if loss <= noop_final_loss + tolerance and prefix_wall + phase_wall < noop_cost:
            return horizon
    return None


def _label(row: dict[str, Any], noop: dict[str, Any], tolerance: float) -> dict[str, Any]:
    action = row.get("action") or {}
    action_kind = str(action.get("kind", ""))
    deltas = _outcome_deltas(row)
    risk = row.get("risk") or {}
    failed = bool(risk.get("failed")) or bool(risk.get("catastrophic_failure"))
    labels: list[str] = []
    jump_horizon = None
    if action_kind == "noop":
        labels.append("noop_baseline")
        primary = "noop_baseline"
    elif failed:
        labels.append("dangerous")
        primary = "dangerous"
    elif deltas is None:
        labels.append("stalled")
        primary = "stalled"
    else:
        early_max = max(deltas["immediate"], deltas["recovery"])
        final_delta = deltas["final"]
        productive = final_delta <= -tolerance and max(deltas.values()) <= tolerance
        if productive:
            labels.append("productive")
            jump_horizon = _jump_horizon(row, noop, tolerance)
            if jump_horizon is not None:
                labels.append("jumpable")
        if final_delta <= tolerance and early_max > tolerance:
            labels.append("recoverable")
        if final_delta > tolerance:
            labels.append("dangerous")
        if not labels:
            labels.append("neutral")
        primary = (
            "jumpable" if "jumpable" in labels else labels[0]
        )
    return {
        "atlas_id": row.get("atlas_id"),
        "group_id": row.get("group_id"),
        "root_id": row.get("root_id"),
        "case_id": row.get("case_id"),
        "action": action_kind,
        "primary_label": primary,
        "labels": labels,
        "jump_horizon": jump_horizon,
        "deltas": deltas,
        "tolerance": tolerance,
        "risk": risk,
        "cost": row.get("cost", {}),
        "provenance": row.get("provenance", {}),
    }


def label_rows(rows: list[dict[str, Any]], tolerance: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        group = str(row.get("group_id", "")).strip()
        action = row.get("action") or {}
        kind = str(action.get("kind", "")).strip()
        if not group or not kind or kind in groups[group]:
            raise ValueError(f"atlas needs unique group/action rows: {group!r}/{kind!r}")
        groups[group][kind] = row
    labeled: list[dict[str, Any]] = []
    missing_noop = 0
    for group, actions in sorted(groups.items()):
        noop = actions.get("noop")
        if noop is None:
            missing_noop += 1
            continue
        for row in actions.values():
            labeled.append(_label(row, noop, tolerance))
    counts = Counter(row["primary_label"] for row in labeled)
    summary = {
        "schema": "landscape-driver.actionability.v1",
        "rows": len(labeled),
        "groups": len(groups),
        "matched_groups": len(groups) - missing_noop,
        "unmatched_groups": missing_noop,
        "tolerance": tolerance,
        "primary_label_counts": dict(sorted(counts.items())),
        "action_counts": dict(sorted(Counter(row["action"] for row in labeled).items())),
    }
    return labeled, summary


def _self_check() -> None:
    common = {
        "action": {"kind": "noop"},
        "outcomes": {
            "immediate": {"loss": 2.0, "loss_delta_vs_noop": 0.0, "wall_seconds": 1.0},
            "recovery": {"loss": 2.0, "loss_delta_vs_noop": 0.0, "wall_seconds": 2.0},
            "final": {"loss": 2.0, "loss_delta_vs_noop": 0.0, "wall_seconds": 3.0},
        },
        "cost": {"wall_seconds": 4.0, "branch_wall_seconds": 3.0},
        "risk": {"failed": False, "catastrophic_failure": False},
    }
    noop = dict(common)
    noop["group_id"] = "g"
    productive = json.loads(json.dumps(common))
    productive.update(
        {
            "group_id": "g",
            "action": {"kind": "probe"},
            "outcomes": {
                "immediate": {"loss": 1.8, "loss_delta_vs_noop": -0.2, "wall_seconds": 1.0},
                "recovery": {"loss": 1.7, "loss_delta_vs_noop": -0.3, "wall_seconds": 2.0},
                "final": {"loss": 1.8, "loss_delta_vs_noop": -0.2, "wall_seconds": 3.0},
            },
        }
    )
    rows, summary = label_rows([noop, productive], DEFAULT_TOLERANCE)
    assert summary["matched_groups"] == 1
    assert next(row for row in rows if row["action"] == "probe")["primary_label"] == "jumpable"
    print("actionability self-check: ok")


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.self_check:
        _self_check()
        return None
    atlas = Path(args.atlas)
    rows = _read_atlas(atlas)
    labeled, summary = label_rows(rows, args.tolerance)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"choose a fresh output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        **summary,
        "atlas": str(atlas),
        "atlas_sha256": _sha256(atlas),
        "files": {"labels": "labels.jsonl"},
    }
    with (output / "labels.jsonl").open("w", encoding="utf-8") as handle:
        for row in labeled:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    (output / "manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas")
    parser.add_argument("--output", default="runs/actionability")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--self-check", action="store_true")
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    if not arguments.self_check and not arguments.atlas:
        raise SystemExit("--atlas is required unless --self-check is used")
    run(arguments)
