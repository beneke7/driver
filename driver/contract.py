"""Fail-closed evaluator for robust landscape-driver speedup claims.

The evaluator consumes one aggregate JSON campaign file, or several files with
the same declared contract.  The preferred input shape is::

    {
      "schema": "landscape-driver.contract.v1",
      "contract": {
        "preregistered": true,
        "thresholds": ["easy", "middle", "hard"],
        "target_speedup": 10,
        "primary_metric": "wall_seconds",
        "gated_metrics": ["wall_seconds"],
        "evidence_metrics": ["wall_seconds", "flops", "tokens"],
        "min_landscapes": 5,
        "min_seeds": 3,
        "case_floor": 1,
        "confidence": 0.95,
        "bootstrap_samples": 4000,
        "random_seed": 0,
        "amortization_deployments": 1
      },
      "development_cases": [{"case_id": "dev-0", "landscape": "...", "seed": 0}],
      "heldout_cases": [{"case_id": "test-0", "landscape": "...", "seed": 0}],
      "one_time_costs": {"baseline": COST, "driver": COST},
      "rows": [{
        "threshold": "easy",
        "case_id": "test-0",
        "split": "heldout",
        "landscape": "...",
        "seed": 0,
        "match_id": "...",
        "provenance": {
          "source_checkpoint_sha256": "...",
          "data_sha256": "...",
          "objective_sha256": "...",
          "config_sha256": "..."
        },
        "baseline": {"reached": true, "failure": null, "cost": COST},
        "driver": {"reached": true, "failure": null, "cost": COST}
      }]
    }

``COST`` has ``scope`` (``deployment`` or ``one_time``), measured
    ``wall_seconds`` with ``wall_scope: "end_to_end"`` for deployments,
    consumed ``tokens``, and an explicit non-negative value for every component
    in ``FLOP_COMPONENTS``.  Wall time is the primary end-to-end
metric; FLOPs and tokens are reported as secondary accounting metrics unless
the declared contract gates them too.

The report requires every declared held-out case at every threshold.  Failed
or missing branches remain failures rather than disappearing from a ratio.
For each threshold and gated metric it requires: complete coverage, a minimum
case floor, every landscape median at least the target, every seed median at
least the target, a geometric mean at least the target, and a one-sided
cluster bootstrap lower bound at the declared confidence.  Landscapes are the
bootstrap clusters, so seeds cannot be counted as independent landscapes.

This module is intentionally stdlib-only.  It does not infer missing costs
from legacy ``estimated_speedup`` fields.  ``python -m driver.contract
--self-check`` exercises both a passing campaign and failing gates.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "landscape-driver.contract.v1"
REPORT_SCHEMA = "landscape-driver.contract-report.v1"
FLOP_COMPONENTS = (
    "target_flops",
    "driver_inference_flops",
    "driver_update_flops",
    "probe_flops",
    "recovery_flops",
    "evaluation_flops",
    "rejected_branch_flops",
    "meta_training_flops",
    "search_flops",
    "other_flops",
)
METRICS = ("wall_seconds", "flops", "tokens")
PROVENANCE_FIELDS = (
    "source_checkpoint_sha256",
    "data_sha256",
    "objective_sha256",
    "config_sha256",
)


class ContractError(ValueError):
    """Invalid contract input; callers should report this as unevaluable."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be an object")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be a non-empty string")
    return value.strip()


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ContractError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ContractError(f"{name} must be finite")
    if result < 0 or (positive and result <= 0):
        qualifier = "positive" if positive else "non-negative"
        raise ContractError(f"{name} must be {qualifier}")
    return result


def _integer(value: Any, name: str, *, positive: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ContractError(f"{name} must be an integer")
    if value < 0 or (positive and value < 1):
        qualifier = "positive" if positive else "non-negative"
        raise ContractError(f"{name} must be {qualifier}")
    return value


def _threshold_key(value: Any) -> str:
    if isinstance(value, bool):
        raise ContractError("threshold identifiers cannot be booleans")
    if isinstance(value, str):
        return _text(value, "threshold")
    if isinstance(value, (int, float)):
        return format(_number(value, "threshold"), ".17g")
    raise ContractError("threshold must be a string or finite number")


def _unique_texts(values: Any, name: str) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ContractError(f"{name} must be a list")
    result = tuple(_text(value, f"{name} item") for value in values)
    if len(set(result)) != len(result):
        raise ContractError(f"{name} must not contain duplicates")
    return result


@dataclass(frozen=True)
class ContractSpec:
    thresholds: tuple[str, ...]
    target_speedup: float
    primary_metric: str
    gated_metrics: tuple[str, ...]
    evidence_metrics: tuple[str, ...]
    min_landscapes: int
    min_seeds: int
    case_floor: float
    confidence: float
    bootstrap_samples: int
    random_seed: int
    amortization_deployments: int

    @classmethod
    def from_dict(cls, raw: Any) -> "ContractSpec":
        value = _mapping(raw, "contract")
        if value.get("preregistered") is not True:
            raise ContractError(
                "contract.preregistered must be true; the evaluator cannot infer a "
                "contract after seeing results"
            )
        thresholds = tuple(_threshold_key(item) for item in value.get("thresholds", ()))
        if len(thresholds) < 2:
            raise ContractError("contract.thresholds must contain at least two thresholds")
        if len(set(thresholds)) != len(thresholds):
            raise ContractError("contract.thresholds must not contain duplicates")
        target_speedup = _number(value.get("target_speedup"), "contract.target_speedup", positive=True)
        primary_metric = _text(value.get("primary_metric"), "contract.primary_metric")
        gated_metrics = _unique_texts(value.get("gated_metrics"), "contract.gated_metrics")
        evidence_metrics = _unique_texts(
            value.get("evidence_metrics"), "contract.evidence_metrics"
        )
        for name, metrics in (("gated_metrics", gated_metrics), ("evidence_metrics", evidence_metrics)):
            unknown = set(metrics) - set(METRICS)
            if unknown:
                raise ContractError(f"contract.{name} contains unknown metrics: {sorted(unknown)}")
        if not gated_metrics or primary_metric not in gated_metrics:
            raise ContractError("primary_metric must be one of the gated_metrics")
        if not set(gated_metrics).issubset(evidence_metrics):
            raise ContractError("every gated metric must also be an evidence metric")
        min_landscapes = _integer(
            value.get("min_landscapes"), "contract.min_landscapes", positive=True
        )
        min_seeds = _integer(value.get("min_seeds"), "contract.min_seeds", positive=True)
        case_floor = _number(value.get("case_floor"), "contract.case_floor")
        confidence = _number(value.get("confidence"), "contract.confidence")
        if not 0.5 < confidence < 1.0:
            raise ContractError("contract.confidence must be between 0.5 and 1")
        bootstrap_samples = _integer(
            value.get("bootstrap_samples"), "contract.bootstrap_samples", positive=True
        )
        if bootstrap_samples < 100:
            raise ContractError("contract.bootstrap_samples must be at least 100")
        random_seed = _integer(value.get("random_seed"), "contract.random_seed")
        amortization_deployments = _integer(
            value.get("amortization_deployments"),
            "contract.amortization_deployments",
            positive=True,
        )
        return cls(
            thresholds=thresholds,
            target_speedup=target_speedup,
            primary_metric=primary_metric,
            gated_metrics=gated_metrics,
            evidence_metrics=evidence_metrics,
            min_landscapes=min_landscapes,
            min_seeds=min_seeds,
            case_floor=case_floor,
            confidence=confidence,
            bootstrap_samples=bootstrap_samples,
            random_seed=random_seed,
            amortization_deployments=amortization_deployments,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "thresholds": list(self.thresholds),
            "target_speedup": self.target_speedup,
            "primary_metric": self.primary_metric,
            "gated_metrics": list(self.gated_metrics),
            "evidence_metrics": list(self.evidence_metrics),
            "min_landscapes": self.min_landscapes,
            "min_seeds": self.min_seeds,
            "case_floor": self.case_floor,
            "confidence": self.confidence,
            "bootstrap_samples": self.bootstrap_samples,
            "random_seed": self.random_seed,
            "amortization_deployments": self.amortization_deployments,
        }


@dataclass(frozen=True)
class Case:
    case_id: str
    landscape: str
    seed: int


def _cases(raw: Any, name: str) -> tuple[Case, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ContractError(f"{name} must be a list")
    result: list[Case] = []
    seen: set[str] = set()
    seen_pairs: set[tuple[str, int]] = set()
    for index, item in enumerate(raw):
        value = _mapping(item, f"{name}[{index}]")
        case = Case(
            case_id=_text(value.get("case_id"), f"{name}[{index}].case_id"),
            landscape=_text(value.get("landscape"), f"{name}[{index}].landscape"),
            seed=_integer(value.get("seed"), f"{name}[{index}].seed"),
        )
        if case.case_id in seen:
            raise ContractError(f"duplicate case_id in {name}: {case.case_id}")
        pair = (case.landscape, case.seed)
        if pair in seen_pairs:
            raise ContractError(
                f"duplicate landscape/seed pair in {name}: {case.landscape!r}/{case.seed}"
            )
        seen.add(case.case_id)
        seen_pairs.add(pair)
        result.append(case)
    if not result:
        raise ContractError(f"{name} must not be empty")
    return tuple(result)


@dataclass(frozen=True)
class Cost:
    scope: str
    wall_seconds: float
    tokens: int
    components: dict[str, float]

    @classmethod
    def from_dict(cls, raw: Any, expected_scope: str) -> "Cost":
        value = _mapping(raw, f"{expected_scope} cost")
        scope = _text(value.get("scope"), f"{expected_scope} cost.scope")
        if scope != expected_scope:
            raise ContractError(
                f"{expected_scope} cost.scope must be {expected_scope!r}, got {scope!r}"
            )
        expected_wall_scope = "end_to_end" if expected_scope == "deployment" else "one_time"
        wall_scope = _text(value.get("wall_scope"), f"{expected_scope} cost.wall_scope")
        if wall_scope != expected_wall_scope:
            raise ContractError(
                f"{expected_scope} cost.wall_scope must be {expected_wall_scope!r}"
            )
        wall_seconds = _number(
            value.get("wall_seconds"), f"{expected_scope} cost.wall_seconds"
        )
        tokens = _integer(value.get("tokens"), f"{expected_scope} cost.tokens")
        components_raw = _mapping(
            value.get("components"), f"{expected_scope} cost.components"
        )
        missing = set(FLOP_COMPONENTS) - set(components_raw)
        unknown = set(components_raw) - set(FLOP_COMPONENTS)
        if missing:
            raise ContractError(
                f"{expected_scope} cost.components missing: {sorted(missing)}"
            )
        if unknown:
            raise ContractError(
                f"{expected_scope} cost.components unknown: {sorted(unknown)}"
            )
        components = {
            name: _number(components_raw[name], f"{expected_scope} cost.components.{name}")
            for name in FLOP_COMPONENTS
        }
        return cls(scope, wall_seconds, tokens, components)

    @property
    def flops(self) -> float:
        return sum(self.components.values())

    def metric_values(self) -> dict[str, float]:
        return {
            "wall_seconds": self.wall_seconds,
            "flops": self.flops,
            "tokens": float(self.tokens),
        }

    def plus_amortized(self, one_time: "Cost", deployments: int) -> dict[str, float]:
        divisor = float(deployments)
        left = self.metric_values()
        right = one_time.metric_values()
        return {name: left[name] + right[name] / divisor for name in METRICS}


@dataclass(frozen=True)
class RunEvidence:
    reached: bool
    failure: str | None
    cost: Cost


@dataclass(frozen=True)
class EvidenceRow:
    threshold: str
    case_id: str
    split: str
    landscape: str
    seed: int
    match_id: str
    provenance: dict[str, str]
    baseline: RunEvidence
    driver: RunEvidence


def _run_evidence(raw: Any, name: str) -> RunEvidence:
    value = _mapping(raw, name)
    reached = value.get("reached")
    if not isinstance(reached, bool):
        raise ContractError(f"{name}.reached must be a boolean")
    failure = value.get("failure")
    if failure is not None:
        failure = _text(failure, f"{name}.failure")
    if reached and failure is not None:
        raise ContractError(f"{name} cannot be reached and failed")
    if not reached and failure is None:
        raise ContractError(f"{name} must name a failure when reached is false")
    return RunEvidence(
        reached=reached,
        failure=failure,
        cost=Cost.from_dict(value.get("cost"), "deployment"),
    )


def _row(raw: Any, index: int) -> EvidenceRow:
    value = _mapping(raw, f"rows[{index}]")
    provenance_raw = _mapping(value.get("provenance"), f"rows[{index}].provenance")
    provenance = {
        name: _text(provenance_raw.get(name), f"rows[{index}].provenance.{name}")
        for name in PROVENANCE_FIELDS
    }
    return EvidenceRow(
        threshold=_threshold_key(value.get("threshold")),
        case_id=_text(value.get("case_id"), f"rows[{index}].case_id"),
        split=_text(value.get("split"), f"rows[{index}].split"),
        landscape=_text(value.get("landscape"), f"rows[{index}].landscape"),
        seed=_integer(value.get("seed"), f"rows[{index}].seed"),
        match_id=_text(value.get("match_id"), f"rows[{index}].match_id"),
        provenance=provenance,
        baseline=_run_evidence(value.get("baseline"), f"rows[{index}].baseline"),
        driver=_run_evidence(value.get("driver"), f"rows[{index}].driver"),
    )


def _one_time_costs(payload: Mapping[str, Any]) -> tuple[Cost, Cost]:
    raw = _mapping(payload.get("one_time_costs"), "one_time_costs")
    return (
        Cost.from_dict(raw.get("baseline"), "one_time"),
        Cost.from_dict(raw.get("driver"), "one_time"),
    )


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot calculate a quantile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _geometric_mean(values: Sequence[float]) -> float | None:
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _cluster_bootstrap_lower(
    grouped_ratios: Mapping[str, Sequence[float]],
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> float | None:
    """Bootstrap landscapes as clusters, not individual seeds."""

    if len(grouped_ratios) < 2 or any(not values for values in grouped_ratios.values()):
        return None
    cluster_logs = [
        sum(math.log(value) for value in values) / len(values)
        for values in grouped_ratios.values()
    ]
    generator = random.Random(seed)
    count = len(cluster_logs)
    estimates = [
        math.exp(
            sum(generator.choice(cluster_logs) for _ in range(count)) / count
        )
        for _ in range(samples)
    ]
    return _quantile(estimates, 1.0 - confidence)


def _case_key(row: EvidenceRow) -> tuple[str, str]:
    return row.threshold, row.case_id


def _metric_report(
    rows: Sequence[EvidenceRow],
    spec: ContractSpec,
    metric: str,
    baseline_one_time: Cost,
    driver_one_time: Cost,
) -> dict[str, Any]:
    ratios: list[dict[str, Any]] = []
    grouped_landscape: dict[str, list[float]] = defaultdict(list)
    grouped_seed: dict[str, list[float]] = defaultdict(list)
    failures: list[dict[str, Any]] = []
    invalid_costs: list[dict[str, Any]] = []
    for row in rows:
        baseline = row.baseline.cost.plus_amortized(
            baseline_one_time, spec.amortization_deployments
        )[metric]
        driver = row.driver.cost.plus_amortized(
            driver_one_time, spec.amortization_deployments
        )[metric]
        if not row.baseline.reached or not row.driver.reached:
            failures.append(
                {
                    "case_id": row.case_id,
                    "landscape": row.landscape,
                    "seed": row.seed,
                    "baseline_failure": row.baseline.failure,
                    "driver_failure": row.driver.failure,
                }
            )
            continue
        if baseline <= 0 or driver <= 0 or not math.isfinite(baseline + driver):
            invalid_costs.append(
                {
                    "case_id": row.case_id,
                    "metric": metric,
                    "baseline_cost": baseline,
                    "driver_cost": driver,
                }
            )
            continue
        speedup = baseline / driver
        ratios.append(
            {
                "case_id": row.case_id,
                "landscape": row.landscape,
                "seed": row.seed,
                "speedup": speedup,
            }
        )
        grouped_landscape[row.landscape].append(speedup)
        grouped_seed[str(row.seed)].append(speedup)

    landscape_stats = {
        name: {
            "n": len(values),
            "median": statistics.median(values),
            "minimum": min(values),
            "pass": len(values) >= spec.min_seeds
            and statistics.median(values) >= spec.target_speedup,
        }
        for name, values in sorted(grouped_landscape.items())
    }
    seed_stats = {
        name: {
            "n": len(values),
            "median": statistics.median(values),
            "minimum": min(values),
            "pass": len(values) >= spec.min_landscapes
            and statistics.median(values) >= spec.target_speedup,
        }
        for name, values in sorted(grouped_seed.items())
    }
    speedup_values = [item["speedup"] for item in ratios]
    geometric_mean = _geometric_mean(speedup_values)
    cluster_lower = _cluster_bootstrap_lower(
        grouped_landscape,
        samples=spec.bootstrap_samples,
        confidence=spec.confidence,
        seed=spec.random_seed,
    )
    complete = not failures and not invalid_costs
    case_floor_pass = bool(speedup_values) and complete and min(speedup_values) >= spec.case_floor
    pass_value = bool(
        complete
        and len(grouped_landscape) >= spec.min_landscapes
        and len(grouped_seed) >= spec.min_seeds
        and case_floor_pass
        and landscape_stats
        and all(item["pass"] for item in landscape_stats.values())
        and seed_stats
        and all(item["pass"] for item in seed_stats.values())
        and geometric_mean is not None
        and geometric_mean >= spec.target_speedup
        and cluster_lower is not None
        and cluster_lower >= spec.target_speedup
    )
    return {
        "metric": metric,
        "n_expected": len(rows),
        "n_scored": len(ratios),
        "failures": failures,
        "invalid_costs": invalid_costs,
        "ratios": ratios,
        "case_floor": spec.case_floor,
        "case_floor_pass": case_floor_pass,
        "per_landscape": landscape_stats,
        "per_seed": seed_stats,
        "geometric_mean": geometric_mean,
        "cluster_bootstrap_lower_bound": cluster_lower,
        "confidence": spec.confidence,
        "target_speedup": spec.target_speedup,
        "pass": pass_value,
    }


def evaluate_campaign(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate one declared campaign and return a JSON-serializable report."""

    report: dict[str, Any] = {
        "report_schema": REPORT_SCHEMA,
        "eligible": False,
        "errors": [],
        "warnings": [],
    }
    if not isinstance(payload, Mapping):
        report["errors"].append("campaign must be an object")
        return report
    if payload.get("schema") != SCHEMA:
        report["errors"].append(
            f"unsupported or missing schema; expected {SCHEMA!r}"
        )
        report["legacy_hint"] = (
            "legacy summaries need per-case baseline/driver costs, matched provenance, "
            "and a declared multi-threshold contract"
        )
        return report
    try:
        spec = ContractSpec.from_dict(payload.get("contract"))
        development = _cases(payload.get("development_cases"), "development_cases")
        heldout = _cases(payload.get("heldout_cases"), "heldout_cases")
        baseline_one_time, driver_one_time = _one_time_costs(payload)
        raw_rows = payload.get("rows")
        if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
            raise ContractError("rows must be a list")
    except ContractError as exc:
        report["errors"].append(str(exc))
        return report

    report["contract"] = spec.to_dict()
    report["one_time_costs"] = {
        "deployments": spec.amortization_deployments,
        "baseline": baseline_one_time.metric_values(),
        "driver": driver_one_time.metric_values(),
    }
    report["coverage"] = {
        "development_cases": len(development),
        "heldout_cases": len(heldout),
        "landscapes": sorted({case.landscape for case in heldout}),
        "seeds": sorted({case.seed for case in heldout}),
        "thresholds": list(spec.thresholds),
        "expected_rows": len(heldout) * len(spec.thresholds),
        "observed_rows": len(raw_rows),
    }

    development_ids = {case.case_id for case in development}
    heldout_by_id = {case.case_id: case for case in heldout}
    if development_ids & set(heldout_by_id):
        report["errors"].append("development and heldout case IDs overlap")
    if len(report["coverage"]["landscapes"]) < spec.min_landscapes:
        report["errors"].append(
            f"heldout roster has {len(report['coverage']['landscapes'])} landscapes; "
            f"contract requires {spec.min_landscapes}"
        )
    if len(report["coverage"]["seeds"]) < spec.min_seeds:
        report["errors"].append(
            f"heldout roster has {len(report['coverage']['seeds'])} seeds; "
            f"contract requires {spec.min_seeds}"
        )
    for landscape in report["coverage"]["landscapes"]:
        count = sum(case.landscape == landscape for case in heldout)
        if count < spec.min_seeds:
            report["errors"].append(
                f"landscape {landscape!r} has {count} heldout cases; "
                f"requires at least {spec.min_seeds}"
            )
    for seed in report["coverage"]["seeds"]:
        count = sum(case.seed == seed for case in heldout)
        if count < spec.min_landscapes:
            report["errors"].append(
                f"seed {seed!r} spans {count} landscapes; "
                f"requires at least {spec.min_landscapes}"
            )

    rows: list[EvidenceRow] = []
    for index, raw in enumerate(raw_rows):
        try:
            rows.append(_row(raw, index))
        except ContractError as exc:
            report["errors"].append(str(exc))

    seen: set[tuple[str, str]] = set()
    provenance_by_case: dict[str, tuple[str, ...]] = {}
    valid_rows: list[EvidenceRow] = []
    declared_thresholds = set(spec.thresholds)
    for row in rows:
        key = _case_key(row)
        if key in seen:
            report["errors"].append(f"duplicate row for threshold/case {key}")
            continue
        seen.add(key)
        if row.threshold not in declared_thresholds:
            report["errors"].append(
                f"row {row.case_id} uses undeclared threshold {row.threshold!r}"
            )
        expected = heldout_by_id.get(row.case_id)
        if expected is None:
            report["errors"].append(f"row {row.case_id!r} is not in heldout_cases")
        else:
            if row.split != "heldout":
                report["errors"].append(f"row {row.case_id!r} is not marked heldout")
            if row.landscape != expected.landscape or row.seed != expected.seed:
                report["errors"].append(
                    f"row {row.case_id!r} disagrees with the heldout roster"
                )
        signature = (
            row.match_id,
            row.provenance["source_checkpoint_sha256"],
            row.provenance["data_sha256"],
            row.provenance["objective_sha256"],
            row.provenance["config_sha256"],
        )
        previous = provenance_by_case.setdefault(row.case_id, signature)
        if previous != signature:
            report["errors"].append(
                f"matched checkpoint/provenance changed across thresholds for {row.case_id!r}"
            )
        valid_rows.append(row)

    expected_keys = {
        (threshold, case.case_id)
        for threshold in spec.thresholds
        for case in heldout
    }
    missing = sorted(expected_keys - seen)
    extra = sorted(seen - expected_keys)
    if missing:
        report["errors"].append(f"missing threshold/case rows: {missing[:8]}")
    if extra:
        report["errors"].append(f"unexpected threshold/case rows: {extra[:8]}")
    if len(valid_rows) != len(raw_rows):
        report["errors"].append("one or more rows could not be parsed")

    if report["errors"]:
        report["eligible"] = False
        return report

    report["thresholds"] = {}
    all_gates_pass = True
    for threshold in spec.thresholds:
        threshold_rows = [row for row in valid_rows if row.threshold == threshold]
        threshold_report: dict[str, Any] = {"metrics": {}, "pass": True}
        for metric in spec.evidence_metrics:
            metric_result = _metric_report(
                threshold_rows,
                spec,
                metric,
                baseline_one_time,
                driver_one_time,
            )
            threshold_report["metrics"][metric] = metric_result
            if metric in spec.gated_metrics and not metric_result["pass"]:
                threshold_report["pass"] = False
        report["thresholds"][threshold] = threshold_report
        all_gates_pass = all_gates_pass and threshold_report["pass"]

    report["eligible"] = all_gates_pass
    if not all_gates_pass:
        report["errors"].append("one or more declared threshold/metric gates failed")
    return report


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON from {path}: {exc}") from exc


def _load_source(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if source.is_dir():
        summary_path = source / "summary.json"
        if not summary_path.exists():
            raise ContractError(f"{source} has no summary.json")
        summary = _read_json(summary_path)
        manifest_path = source / "manifest.json"
        manifest = _read_json(manifest_path) if manifest_path.exists() else None
        if isinstance(summary, Mapping) and summary.get("schema") == SCHEMA:
            payload = dict(summary)
            if manifest is not None:
                payload.setdefault("manifest", manifest)
            return payload
        return {
            "_legacy_source": str(source),
            "_legacy_summary": summary,
            "_legacy_manifest": manifest,
        }
    value = _read_json(source)
    if not isinstance(value, Mapping):
        raise ContractError(f"{source} must contain a JSON object")
    return dict(value)


def _legacy_report(payloads: Sequence[Mapping[str, Any]], errors: Sequence[str]) -> dict[str, Any]:
    observed: list[dict[str, Any]] = []
    for payload in payloads:
        summary = payload.get("_legacy_summary", payload)
        manifest = payload.get("_legacy_manifest")
        if not isinstance(summary, Mapping):
            continue
        rows = summary.get("rows", [])
        observed.append(
            {
                "source": payload.get("_legacy_source", "input"),
                "threshold": summary.get("threshold"),
                "rows": len(rows) if isinstance(rows, list) else None,
                "wall_speedup": summary.get("wall_speedup"),
                "estimated_speedups": (
                    [row.get("estimated_speedup") for row in rows if isinstance(row, Mapping)]
                    if isinstance(rows, list)
                    else []
                ),
                "manifest_has_development_cases": isinstance(manifest, Mapping)
                and bool(manifest.get("development_cases")),
            }
        )
    return {
        "report_schema": REPORT_SCHEMA,
        "eligible": False,
        "mode": "legacy_diagnostic",
        "errors": list(errors)
        + [
            "legacy evidence is diagnostic only: per-case end-to-end costs, explicit "
            "failure rows, matched provenance, and a preregistered multi-threshold "
            "contract are required for a claim"
        ],
        "observed": observed,
    }


def evaluate_paths(paths: Sequence[str | Path]) -> dict[str, Any]:
    """Load one aggregate campaign or merge threshold slices with one contract."""

    if not paths:
        return {
            "report_schema": REPORT_SCHEMA,
            "eligible": False,
            "errors": ["at least one input is required"],
        }
    loaded: list[dict[str, Any]] = []
    load_errors: list[str] = []
    for path in paths:
        try:
            loaded.append(_load_source(path))
        except ContractError as exc:
            load_errors.append(str(exc))
    if load_errors:
        return _legacy_report(loaded, load_errors)
    if any(payload.get("schema") != SCHEMA for payload in loaded):
        return _legacy_report(
            loaded,
            [f"input {index} is not {SCHEMA!r}" for index, payload in enumerate(loaded) if payload.get("schema") != SCHEMA],
        )
    if len(loaded) == 1:
        return evaluate_campaign(loaded[0])

    first = loaded[0]
    merged = dict(first)
    first_contract = first.get("contract")
    first_development = first.get("development_cases")
    first_heldout = first.get("heldout_cases")
    all_rows: list[Any] = []
    one_time_payloads: list[Any] = []
    for payload in loaded:
        if payload.get("contract") != first_contract:
            return {
                "report_schema": REPORT_SCHEMA,
                "eligible": False,
                "errors": ["multi-file inputs do not declare identical contracts"],
            }
        if payload.get("development_cases") != first_development:
            return {
                "report_schema": REPORT_SCHEMA,
                "eligible": False,
                "errors": ["multi-file inputs do not declare identical development rosters"],
            }
        if payload.get("heldout_cases") != first_heldout:
            return {
                "report_schema": REPORT_SCHEMA,
                "eligible": False,
                "errors": ["multi-file inputs do not declare identical heldout rosters"],
            }
        rows = payload.get("rows", [])
        if not isinstance(rows, list):
            return {
                "report_schema": REPORT_SCHEMA,
                "eligible": False,
                "errors": ["multi-file input rows must be lists"],
            }
        all_rows.extend(rows)
        if "one_time_costs" in payload:
            one_time_payloads.append(payload["one_time_costs"])
    if len(one_time_payloads) > 1 and any(
        payload != one_time_payloads[0] for payload in one_time_payloads[1:]
    ):
        return {
            "report_schema": REPORT_SCHEMA,
            "eligible": False,
            "errors": [
                "multi-file one_time_costs differ; use one aggregate campaign file "
                "to make amortization unambiguous"
            ],
        }
    merged["rows"] = all_rows
    return evaluate_campaign(merged)


def _zero_cost(scope: str, wall_seconds: float, tokens: int, flops: float) -> dict[str, Any]:
    components = {name: 0.0 for name in FLOP_COMPONENTS}
    components["target_flops"] = flops
    return {
        "scope": scope,
        "wall_scope": "end_to_end" if scope == "deployment" else "one_time",
        "wall_seconds": wall_seconds,
        "tokens": tokens,
        "components": components,
    }


def _self_check_campaign() -> dict[str, Any]:
    landscapes = [f"landscape-{index}" for index in range(5)]
    seeds = [0, 1, 2]
    thresholds = ["easy", "middle", "hard"]
    heldout = [
        {"case_id": f"{landscape}-{seed}", "landscape": landscape, "seed": seed}
        for landscape in landscapes
        for seed in seeds
    ]
    development = [{"case_id": "development-0", "landscape": "dev", "seed": 0}]
    rows = []
    for threshold in thresholds:
        for case in heldout:
            rows.append(
                {
                    "threshold": threshold,
                    **case,
                    "split": "heldout",
                    "match_id": f"match-{case['case_id']}",
                    "provenance": {
                        name: f"{name}-{case['case_id']}" for name in PROVENANCE_FIELDS
                    },
                    "baseline": {
                        "reached": True,
                        "failure": None,
                        "cost": _zero_cost("deployment", 20.0, 200, 2_000.0),
                    },
                    "driver": {
                        "reached": True,
                        "failure": None,
                        "cost": _zero_cost("deployment", 1.0, 10, 100.0),
                    },
                }
            )
    return {
        "schema": SCHEMA,
        "contract": {
            "preregistered": True,
            "thresholds": thresholds,
            "target_speedup": 10.0,
            "primary_metric": "wall_seconds",
            "gated_metrics": ["wall_seconds", "flops", "tokens"],
            "evidence_metrics": list(METRICS),
            "min_landscapes": 5,
            "min_seeds": 3,
            "case_floor": 1.0,
            "confidence": 0.95,
            "bootstrap_samples": 200,
            "random_seed": 0,
            "amortization_deployments": 1,
        },
        "development_cases": development,
        "heldout_cases": heldout,
        "one_time_costs": {
            "baseline": _zero_cost("one_time", 0.0, 0, 0.0),
            "driver": _zero_cost("one_time", 0.0, 0, 0.0),
        },
        "rows": rows,
    }


def self_check() -> None:
    passing = evaluate_campaign(_self_check_campaign())
    if not passing["eligible"]:
        raise AssertionError(f"valid contract did not pass: {passing}")
    if passing["thresholds"]["hard"]["metrics"]["wall_seconds"]["cluster_bootstrap_lower_bound"] < 10:
        raise AssertionError("cluster lower bound was not calculated")

    failing_campaign = copy.deepcopy(_self_check_campaign())
    for row in failing_campaign["rows"]:
        if row["landscape"] == "landscape-0":
            row["driver"]["cost"]["wall_seconds"] = 20.0
            row["driver"]["cost"]["components"]["target_flops"] = 2_000.0
            row["driver"]["cost"]["tokens"] = 200
    failing = evaluate_campaign(failing_campaign)
    if failing["eligible"]:
        raise AssertionError("per-landscape regression was not rejected")
    for threshold in failing["thresholds"].values():
        if threshold["metrics"]["wall_seconds"]["per_landscape"]["landscape-0"]["pass"]:
            raise AssertionError("landscape gate unexpectedly passed")

    failed_row_campaign = copy.deepcopy(_self_check_campaign())
    failed_row_campaign["rows"][0]["driver"]["reached"] = False
    failed_row_campaign["rows"][0]["driver"]["failure"] = "diverged"
    failed = evaluate_campaign(failed_row_campaign)
    if failed["eligible"]:
        raise AssertionError("failed branch was silently dropped")
    if not failed["thresholds"]["easy"]["metrics"]["wall_seconds"]["failures"]:
        raise AssertionError("failed branch was not reported")
    print("contract self-check: PASS")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", help="aggregate JSON or evidence directories")
    parser.add_argument("--output", help="optional path for the JSON report")
    parser.add_argument("--self-check", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.self_check:
        self_check()
        return 0
    if not args.input:
        _parser().error("--input is required unless --self-check is used")
    report = evaluate_paths(args.input)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0 if report.get("eligible") else 1


if __name__ == "__main__":
    raise SystemExit(main())
