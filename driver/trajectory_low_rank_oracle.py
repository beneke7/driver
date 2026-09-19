"""Measure a hindsight low-rank trajectory-transport ceiling.

The basis contains only model-state differences observed before the parent
checkpoint.  Hindsight is used to fit the projection coefficients onto that
basis, so the result is an oracle upper bound and never a deployable policy.
The projected state is then evaluated through the existing matched-branch
transport harness with the parent AdamW state preserved and the data cursor
advanced explicitly.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from .decoder_campaign import _sync
from .trajectory_transport_oracle import (
    TRANSPORT_ROLES,
    _campaign_and_config,
    _file_sha256,
    _future_snapshot,
    _repo_path,
    _run_variant,
    _snapshot_state,
    _source_transition,
    _streams,
)


DEFAULT_BASIS_STEPS = (1152, 1280, 1408)


def _history_path(case: dict[str, Any], step: int) -> Path:
    for value in case.get("trajectory_snapshots", ()):
        path = _repo_path(value)
        if path.name == f"step-{step:06d}.pt":
            return path
    raise FileNotFoundError(f"missing pre-parent trajectory snapshot at step {step}")


def _parent_state(path: Path, *, config_sha: str, data_sha: str) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"parent checkpoint is missing metadata: {path}")
    if metadata.get("config_sha256") != config_sha or metadata.get("data_sha256") != data_sha:
        raise ValueError(f"parent checkpoint provenance mismatch: {path}")
    state = payload.get("model")
    if not isinstance(state, dict):
        raise ValueError(f"parent checkpoint is missing model state: {path}")
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def _project_state(
    parent: dict[str, torch.Tensor],
    basis_vectors: tuple[dict[str, torch.Tensor], ...],
    future: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], torch.Tensor, float, float, list[float], float]:
    if not basis_vectors:
        raise ValueError("at least one basis vector is required")
    names = tuple(parent)
    if any(set(state) != set(names) for state in (*basis_vectors, future)):
        raise ValueError("parent, basis-vector, and future state keys differ")
    rank = len(basis_vectors)
    gram = torch.zeros((rank, rank), dtype=torch.float64)
    rhs = torch.zeros(rank, dtype=torch.float64)
    basis_energy = 0.0
    target_energy = 0.0
    for name in names:
        parent_value = parent[name].float()
        differences = [state[name].float() for state in basis_vectors]
        target_delta = future[name].float() - parent_value
        for left in range(rank):
            basis_energy += float(differences[left].square().sum(dtype=torch.float64))
            rhs[left] += (differences[left] * target_delta).sum(dtype=torch.float64)
            for right in range(left, rank):
                value = (differences[left] * differences[right]).sum(dtype=torch.float64)
                gram[left, right] += value
                if right != left:
                    gram[right, left] += value
        target_energy += float(target_delta.square().sum(dtype=torch.float64))
    singular_values_squared = torch.linalg.svdvals(gram)
    tolerance = max(float(singular_values_squared.max()), 1.0) * 1e-8
    inverse = torch.where(
        singular_values_squared > tolerance,
        singular_values_squared.reciprocal(),
        torch.zeros_like(singular_values_squared),
    )
    gram_u, _, gram_vh = torch.linalg.svd(gram)
    coefficients = gram_vh.transpose(0, 1) @ (
        inverse * (gram_u.transpose(0, 1) @ rhs)
    )
    effective_rank = float((singular_values_squared > tolerance).sum().item())
    projected: dict[str, torch.Tensor] = {}
    residual_energy = 0.0
    for name in names:
        parent_value = parent[name].float()
        target_delta = future[name].float() - parent_value
        projected_delta = sum(
            coefficients[index].float() * state[name].float()
            for index, state in enumerate(basis_vectors)
        )
        residual_energy += float((target_delta - projected_delta).square().sum(dtype=torch.float64))
        projected[name] = (parent_value + projected_delta).to(dtype=parent[name].dtype)
    residual_ratio = residual_energy / max(target_energy, 1e-20)
    singular_values = [
        float(value.sqrt().item()) for value in singular_values_squared
    ]
    positive = [value for value in singular_values if value > math.sqrt(tolerance)]
    condition_number = max(positive) / min(positive) if positive else float("inf")
    return (
        projected,
        coefficients,
        residual_ratio,
        basis_energy,
        singular_values,
        condition_number,
    )


def _write_snapshot(
    path: Path,
    *,
    state: dict[str, torch.Tensor],
    config_sha: str,
    data_sha: str,
    step: int,
) -> None:
    torch.save(
        {
            "metadata": {
                "config_sha256": config_sha,
                "data_sha256": data_sha,
                "step": step,
            },
            "model": state,
        },
        path,
    )


def _run_case(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    case: dict[str, Any],
    basis_steps: tuple[int, ...],
    future_step: int,
    recovery_after: int,
    device: torch.device,
) -> dict[str, Any]:
    campaign, target = _campaign_and_config(manifest["config"], case)
    if future_step <= target.prefix_steps or future_step >= target.prefix_steps + target.final_steps:
        raise ValueError("future_step must be strictly inside the recorded continuation")
    if tuple(sorted(basis_steps)) != basis_steps or any(
        step >= target.prefix_steps for step in basis_steps
    ):
        raise ValueError("basis steps must be sorted and strictly pre-parent")
    train_values, validation_values, data_sha = _streams(
        campaign, landscape=target.landscape, seed=target.seed
    )
    if data_sha != case["data_sha256"]:
        raise ValueError(f"reconstructed data hash does not match {case['case_id']}")
    parent_path = _repo_path(case["parent_checkpoint"])
    config_sha = str(case["config_sha256"])
    future_path = _future_snapshot(parent_path, future_step)
    source_transition = _source_transition(manifest_path, case["case_id"])
    started = time.perf_counter()
    parent: dict[str, torch.Tensor] | None = None
    basis_vectors: tuple[dict[str, torch.Tensor], ...] = ()
    future: dict[str, torch.Tensor] | None = None
    projected: dict[str, torch.Tensor] | None = None
    try:
        parent = _parent_state(parent_path, config_sha=config_sha, data_sha=data_sha)
        basis_states = tuple(
            _snapshot_state(
                _history_path(case, step), config_sha=config_sha, data_sha=data_sha
            )[0]
            for step in basis_steps
        )
        future = _snapshot_state(
            future_path, config_sha=config_sha, data_sha=data_sha
        )[0]
        ordered_states = (basis_states + (parent,))
        basis_vectors = tuple(
            {
                name: right[name].float() - left[name].float()
                for name in parent
            }
            for left, right in zip(ordered_states[:-1], ordered_states[1:])
        )
        (
            projected,
            coefficients,
            residual_ratio,
            basis_energy,
            singular_values,
            condition_number,
        ) = _project_state(
            parent, basis_vectors, future
        )
        projection_seconds = time.perf_counter() - started
        parameter_count = sum(value.numel() for value in parent.values())
        projection_flops = 6.0 * len(basis_steps) * parameter_count
        with tempfile.TemporaryDirectory(prefix="trajectory-low-rank-") as temporary:
            projected_path = Path(temporary) / "projected.pt"
            _write_snapshot(
                projected_path,
                state=projected,
                config_sha=config_sha,
                data_sha=data_sha,
                step=future_step,
            )
            result = _run_variant(
                parent=parent_path,
                future=projected_path,
                source_case={
                    **case,
                    "baseline_snapshot_losses": {},
                },
                source_transition=source_transition,
                campaign=campaign,
                target=target,
                train_values=train_values,
                validation_values=validation_values,
                config_sha=config_sha,
                data_sha=data_sha,
                future_step=future_step,
                recovery_after=recovery_after,
                cursor_policy="skip",
                optimizer_state_policy="preserve",
                transport_roles=None,
                device=device,
            )
        result["basis_steps"] = list(basis_steps)
        result["parent_checkpoint_sha256"] = case["parent_checkpoint_sha256"]
        result["future_snapshot_sha256"] = _file_sha256(future_path)
        result["basis_snapshot_sha256"] = {
            str(step): _file_sha256(_history_path(case, step)) for step in basis_steps
        }
        result["transport_kind"] = "hindsight_low_rank_preparent_projection"
        result["projection_coefficients"] = [float(value) for value in coefficients]
        result["projection_residual_energy_ratio"] = residual_ratio
        result["basis_energy"] = basis_energy
        result["basis_singular_values"] = singular_values
        result["basis_condition_number"] = condition_number
        result["basis_effective_rank"] = sum(
            value > max(singular_values, default=0.0) * 1e-4
            for value in singular_values
        )
        result["projection_seconds"] = projection_seconds
        result["projection_flops"] = projection_flops
        if result.get("failed"):
            return result
        result["candidate_total_wall_seconds"] += projection_seconds
        result["wall_speedup"] = result["baseline_total_wall_seconds"] / max(
            result["candidate_total_wall_seconds"], 1e-9
        )
        result["candidate_total_flops"] += projection_flops
        result["candidate_total_flops_with_skipped_work"] += projection_flops
        result["flop_speedup"] = result["baseline_total_flops"] / max(
            result["candidate_total_flops"], 1.0
        )
        result["conservative_flop_speedup"] = result["baseline_total_flops"] / max(
            result["candidate_total_flops_with_skipped_work"], 1.0
        )
        result["oracle_violations"] = [
            "hindsight_future_coefficients",
            "preparent_basis_only_but_future_target_used_for_projection",
            "future_optimizer_state_unavailable",
        ]
        return result
    except Exception as exc:
        _sync(device)
        return {
            "case_id": case["case_id"],
            "basis_steps": list(basis_steps),
            "transport_kind": "hindsight_low_rank_preparent_projection",
            "failed": True,
            "failure": f"{type(exc).__name__}: {exc}",
            "oracle_only": True,
        }
    finally:
        del parent, basis_vectors, future, projected
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _self_check() -> None:
    parent = {"x": torch.zeros(4), "y": torch.zeros(2)}
    basis = ({"x": torch.ones(4), "y": torch.ones(2)},)
    future = {"x": torch.full((4,), 3.0), "y": torch.full((2,), 3.0)}
    projected, coefficients, residual, _, singular_values, condition_number = _project_state(
        parent, basis, future
    )
    assert torch.allclose(projected["x"], future["x"])
    assert torch.allclose(projected["y"], future["y"])
    assert torch.allclose(coefficients, torch.tensor([3.0], dtype=torch.float64))
    assert residual < 1e-10
    assert abs(singular_values[0] - math.sqrt(6.0)) < 1e-12
    assert condition_number == 1.0
    assert DEFAULT_BASIS_STEPS == (1152, 1280, 1408)
    assert set(TRANSPORT_ROLES) == {"embedding", "attention", "mlp", "norm", "head"}
    print("trajectory low-rank oracle self-check: ok")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.self_check:
        _self_check()
        return {"self_check": "ok"}
    if not args.source_manifest:
        raise ValueError("at least one --source-manifest is required")
    source_manifests = tuple(_repo_path(value) for value in args.source_manifest)
    basis_steps = tuple(args.basis_step or DEFAULT_BASIS_STEPS)
    if len(set(basis_steps)) != len(basis_steps) or any(step <= 0 for step in basis_steps):
        raise ValueError("basis steps must be distinct positive integers")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"choose a fresh --output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cases: list[dict[str, Any]] = []
    for manifest_path in source_manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for case in manifest["cases"]:
            cases.append(
                _run_case(
                    manifest_path=manifest_path,
                    manifest=manifest,
                    case=case,
                    basis_steps=basis_steps,
                    future_step=args.future_step,
                    recovery_after=args.recovery_after,
                    device=device,
                )
            )
    failures = [case for case in cases if case["failed"]]
    result = {
        "schema": "landscape-driver.trajectory-low-rank-oracle.v1",
        "stage": "oracle_ceiling",
        "promotion_status": "diagnostic_only",
        "oracle_only": True,
        "hypothesis": "pre-parent update geometry contains a useful low-rank approximation to a future training displacement",
        "basis_steps": list(basis_steps),
        "future_step": args.future_step,
        "recovery_after": args.recovery_after,
        "cursor_policy": "skip",
        "optimizer_state_policy": "preserve",
        "cost_definition": {
            "baseline": "recorded AdamW/no-op continuation plus shared prefix",
            "candidate": "projected transport, projection solve, serialized checkpoint load, continuation, evaluation, and shared prefix",
            "conservative": "declared skipped exposure is charged separately in each result",
        },
        "device": str(device),
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "source_manifests": [
            {"path": str(path), "sha256": _file_sha256(path)}
            for path in source_manifests
        ],
        "coverage": {
            "cases": len(cases),
            "variants": len(cases),
            "failed_variants": len(failures),
            "quality_passes": sum(bool(not case["failed"] and case["quality_pass"]) for case in cases),
        },
        "cases": cases,
        "failures": failures,
    }
    (output / "manifest.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": str(output / "manifest.json"), **result["coverage"]}, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", action="append")
    parser.add_argument("--output", default="runs/trajectory-low-rank-oracle")
    parser.add_argument("--basis-step", action="append", type=int)
    parser.add_argument("--future-step", type=int, default=2048)
    parser.add_argument("--recovery-after", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--self-check", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
