"""Evaluate a small learned role-wise trajectory transport baseline.

The fit is intentionally simple: it learns one coefficient per tensor role
for a recent checkpoint-difference basis.  Coefficients are fit only from
the training manifests and applied to complete held-out roots.  The future
checkpoint is used only to create supervised targets and to score the held-out
branch; it is never loaded by the candidate action.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from .checkpoints import load_checkpoint
from .decoder_benchmark import TokenStream, _flops, _role
from .decoder_campaign import (
    _evaluate,
    _model_and_optimizer,
    _sync,
    _train_step,
)
from .trajectory_transport_oracle import (
    _campaign_and_config,
    _file_sha256,
    _future_snapshot,
    _load_snapshot_loss,
    _repo_path,
    _snapshot_state,
    _source_transition,
    _streams,
)


ROLES = ("embedding", "attention", "mlp", "norm", "head")


def _code_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _history_snapshot(parent: Path, step: int) -> Path:
    path = parent.parent / "trajectory" / f"step-{step:06d}.pt"
    if not path.exists():
        raise FileNotFoundError(f"missing history snapshot: {path}")
    return path


def _fit_coefficients(
    manifests: tuple[Path, ...], *, history_step: int, future_step: int
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    totals = {role: {"dot": 0.0, "recent": 0.0} for role in ROLES}
    rows: list[dict[str, Any]] = []
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for case in manifest["cases"]:
            campaign, target = _campaign_and_config(manifest["config"], case)
            _, _, data_sha = _streams(
                campaign, landscape=target.landscape, seed=target.seed
            )
            if data_sha != case["data_sha256"]:
                raise ValueError(f"data hash mismatch in {case['case_id']}")
            parent = _repo_path(case["parent_checkpoint"])
            config_sha = str(case["config_sha256"])
            previous, previous_step = _snapshot_state(
                _history_snapshot(parent, history_step),
                config_sha=config_sha,
                data_sha=data_sha,
            )
            current, current_step = _snapshot_state(
                _history_snapshot(parent, target.prefix_steps),
                config_sha=config_sha,
                data_sha=data_sha,
            )
            future, observed_future = _snapshot_state(
                _future_snapshot(parent, future_step),
                config_sha=config_sha,
                data_sha=data_sha,
            )
            if previous_step != history_step or current_step != target.prefix_steps:
                raise ValueError(f"history snapshot step mismatch in {case['case_id']}")
            if observed_future != future_step:
                raise ValueError(f"future snapshot step mismatch in {case['case_id']}")
            row_totals = {role: {"dot": 0.0, "recent": 0.0} for role in ROLES}
            for name, value in current.items():
                role = _role(name)
                recent = value.float() - previous[name].float()
                target_delta = future[name].float() - value.float()
                dot = float((recent * target_delta).sum())
                recent_energy = float((recent * recent).sum())
                totals[role]["dot"] += dot
                totals[role]["recent"] += recent_energy
                row_totals[role]["dot"] += dot
                row_totals[role]["recent"] += recent_energy
            rows.append(
                {
                    "case_id": case["case_id"],
                    "manifest": str(manifest_path),
                    "width": target.width,
                    "history_step": history_step,
                    "parent_step": target.prefix_steps,
                    "future_step": future_step,
                    "coefficients": {
                        role: row_totals[role]["dot"]
                        / max(row_totals[role]["recent"], 1e-20)
                        for role in ROLES
                    },
                }
            )
            del previous, current, future
            gc.collect()
    if not rows:
        raise ValueError("no training cases were found")
    coefficients = {
        role: totals[role]["dot"] / max(totals[role]["recent"], 1e-20)
        for role in ROLES
    }
    return coefficients, rows


@torch.no_grad()
def _apply_transport(
    model: torch.nn.Module,
    previous: dict[str, torch.Tensor],
    coefficients: dict[str, float],
) -> dict[str, Any]:
    energy = torch.zeros((), device=next(model.parameters()).device, dtype=torch.float64)
    parameter_count = 0
    for name, parameter in model.named_parameters():
        basis = parameter - previous[name].to(
            device=parameter.device, dtype=parameter.dtype
        )
        coefficient = float(coefficients[_role(name)])
        parameter.add_(coefficient * basis)
        energy += (coefficient * basis).float().square().sum(dtype=torch.float64) / (
            parameter.float().square().sum(dtype=torch.float64) + 1e-12
        )
        parameter_count += parameter.numel()
    return {
        "kind": "rolewise_recent_delta",
        "coefficients": dict(coefficients),
        "energy": float(energy.item()),
        "flops": 2.0 * parameter_count,
        "optimizer_state": "preserved_parent",
        "data_policy": "declared_skip",
    }


def _run_case(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    case: dict[str, Any],
    coefficients: dict[str, float],
    history_step: int,
    future_step: int,
    recovery_after: int,
    device: torch.device,
) -> dict[str, Any]:
    campaign, target = _campaign_and_config(manifest["config"], case)
    train_values, validation_values, data_sha = _streams(
        campaign, landscape=target.landscape, seed=target.seed
    )
    if data_sha != case["data_sha256"]:
        raise ValueError(f"data hash mismatch in {case['case_id']}")
    parent = _repo_path(case["parent_checkpoint"])
    config_sha = str(case["config_sha256"])
    future = _future_snapshot(parent, future_step)
    source_transition = _source_transition(manifest_path, case["case_id"])
    baseline_model, baseline_optimizer = _model_and_optimizer(target, device)
    load_checkpoint(
        parent,
        model=baseline_model,
        optimizer=baseline_optimizer,
        config_sha256=config_sha,
        data_sha256=data_sha,
    )
    baseline_snapshot_losses: dict[str, float] = {}
    for step in (future_step, future_step + recovery_after, target.prefix_steps + target.final_steps):
        path = _future_snapshot(parent, step)
        loss, observed_step = _load_snapshot_loss(
            baseline_model,
            path,
            config_sha=config_sha,
            data_sha=data_sha,
            validation_values=validation_values,
            target=target,
            campaign=campaign,
            device=device,
        )
        if observed_step != step:
            raise ValueError(f"snapshot step mismatch: {path}")
        baseline_snapshot_losses[str(step)] = loss
    del baseline_model, baseline_optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    previous, previous_step = _snapshot_state(
        _history_snapshot(parent, history_step),
        config_sha=config_sha,
        data_sha=data_sha,
    )
    if previous_step != history_step:
        raise ValueError(f"history snapshot step mismatch: {parent}")
    started = time.perf_counter()
    model = None
    optimizer = None
    try:
        model, optimizer = _model_and_optimizer(target, device)
        metadata = load_checkpoint(
            parent,
            model=model,
            optimizer=optimizer,
            config_sha256=config_sha,
            data_sha256=data_sha,
        )
        parent_cursor = int(metadata["data_state"]["cursor"])
        continuation_steps = target.prefix_steps + target.final_steps - future_step
        if continuation_steps <= recovery_after:
            raise ValueError("recovery horizon must precede the endpoint")
        maneuver = _apply_transport(model, previous, coefficients)
        transport_loss = _evaluate(
            model,
            validation_values,
            target_config=target,
            campaign=campaign,
            device=device,
        )
        skip_steps = future_step - target.prefix_steps
        stream = TokenStream(
            train_values,
            cursor=parent_cursor + skip_steps * target.batch_size * (target.context + 1),
        )
        phase_metrics = [
            {"name": "transport", "global_step": future_step, "loss": transport_loss},
        ]
        for local_step in range(continuation_steps):
            _train_step(
                model,
                optimizer,
                stream,
                target,
                campaign,
                device,
                step=local_step,
                schedule="noop",
                runtime={},
            )
            if local_step + 1 == recovery_after:
                phase_metrics.append(
                    {
                        "name": "recovery",
                        "global_step": future_step + local_step + 1,
                        "loss": _evaluate(
                            model,
                            validation_values,
                            target_config=target,
                            campaign=campaign,
                            device=device,
                        ),
                    }
                )
        final_loss = _evaluate(
            model,
            validation_values,
            target_config=target,
            campaign=campaign,
            device=device,
        )
        phase_metrics.append(
            {
                "name": "final",
                "global_step": target.prefix_steps + target.final_steps,
                "loss": final_loss,
            }
        )
        _sync(device)
        wall_seconds = time.perf_counter() - started
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        target_tokens = continuation_steps * target.batch_size * target.context
        skipped_tokens = skip_steps * target.batch_size * target.context
        evaluation_flops = 2.0 * parameter_count * target.batch_size * target.context * 3
        target_flops = _flops(target, model, target_tokens)
        branch_flops = target_flops + maneuver["flops"] + evaluation_flops
        conservative_branch_flops = branch_flops + _flops(target, model, skipped_tokens)
        prefix_flops = float(source_transition["before"]["compute_flops"])
        baseline_total_flops = prefix_flops + float(source_transition["compute_flops"])
        candidate_total_flops = prefix_flops + branch_flops
        prefix_wall = float(source_transition["before"]["features"].get("prefix_seconds", 0.0))
        baseline_wall = prefix_wall + float(source_transition["wall_seconds"])
        candidate_wall = prefix_wall + wall_seconds
        baseline_final = float(source_transition["after"]["loss"])
        return {
            "case_id": case["case_id"],
            "manifest": str(manifest_path),
            "coefficients": dict(coefficients),
            "history_step": history_step,
            "future_step": future_step,
            "recovery_after": recovery_after,
            "parent_cursor": parent_cursor,
            "candidate_cursor": stream.cursor,
            "skipped_tokens": skipped_tokens,
            "phase_metrics": phase_metrics,
            "baseline_snapshot_losses": baseline_snapshot_losses,
            "baseline_final_loss": baseline_final,
            "final_delta_vs_baseline": final_loss - baseline_final,
            "quality_pass": final_loss <= baseline_final,
            "wall_seconds": wall_seconds,
            "baseline_total_wall_seconds": baseline_wall,
            "candidate_total_wall_seconds": candidate_wall,
            "wall_speedup": baseline_wall / candidate_wall,
            "target_tokens": target_tokens,
            "target_flops": target_flops,
            "transport_flops": maneuver["flops"],
            "evaluation_flops": evaluation_flops,
            "actual_branch_flops": branch_flops,
            "conservative_branch_flops_with_skipped_work": conservative_branch_flops,
            "baseline_total_flops": baseline_total_flops,
            "candidate_total_flops": candidate_total_flops,
            "candidate_total_flops_with_skipped_work": prefix_flops + conservative_branch_flops,
            "flop_speedup": baseline_total_flops / candidate_total_flops,
            "conservative_flop_speedup": baseline_total_flops
            / (prefix_flops + conservative_branch_flops),
            "maneuver": maneuver,
            "oracle_only": False,
            "failed": False,
        }
    except Exception as exc:
        _sync(device)
        return {
            "case_id": case["case_id"],
            "manifest": str(manifest_path),
            "failed": True,
            "failure": f"{type(exc).__name__}: {exc}",
            "oracle_only": False,
        }
    finally:
        del model, optimizer, previous
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _self_check() -> None:
    assert set(ROLES) == {"embedding", "attention", "mlp", "norm", "head"}
    assert _role("blocks.0.mlp_in.weight") == "mlp"


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.self_check:
        _self_check()
        print("trajectory transport probe self-check: ok")
        return {"self_check": "ok"}
    if not args.train_manifest or not args.test_manifest:
        raise ValueError("at least one --train-manifest and --test-manifest are required")
    train_manifests = tuple(_repo_path(path) for path in args.train_manifest)
    test_manifests = tuple(_repo_path(path) for path in args.test_manifest)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"choose a fresh --output; {output} is not empty")
    output.mkdir(parents=True, exist_ok=True)
    coefficients, fit_rows = _fit_coefficients(
        train_manifests, history_step=args.history_step, future_step=args.future_step
    )
    device = torch.device(args.device)
    cases: list[dict[str, Any]] = []
    for manifest_path in test_manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for case in manifest["cases"]:
            random.seed(int(case["seed"]))
            torch.manual_seed(int(case["seed"]))
            if device.type == "cuda":
                torch.cuda.manual_seed_all(int(case["seed"]))
            cases.append(
                _run_case(
                    manifest_path=manifest_path,
                    manifest=manifest,
                    case=case,
                    coefficients=coefficients,
                    history_step=args.history_step,
                    future_step=args.future_step,
                    recovery_after=args.recovery_after,
                    device=device,
                )
            )
    failures = [result for case in cases for result in [case] if result.get("failed")]
    output_manifest = {
        "schema": "landscape-driver.trajectory-transport-probe.v1",
        "stage": "learned_structured_transport",
        "promotion_status": "diagnostic_baseline",
        "oracle_only": False,
        "action": {
            "kind": "rolewise_recent_delta",
            "basis": "step_history_to_parent",
            "optimizer_state": "preserved_parent",
            "data_policy": "declared_skip_to_future_step",
        },
        "device": str(device),
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "history_step": args.history_step,
        "future_step": args.future_step,
        "recovery_after": args.recovery_after,
        "coefficients": coefficients,
        "fit_rows": fit_rows,
        "training_manifests": [
            {"path": str(path), "sha256": _file_sha256(path)} for path in train_manifests
        ],
        "test_manifests": [
            {"path": str(path), "sha256": _file_sha256(path)} for path in test_manifests
        ],
        "immutable": {
            "git_commit": _code_revision(),
            "code_sha256": _file_sha256(Path(__file__).resolve()),
        },
        "coverage": {
            "training_cases": len(fit_rows),
            "test_cases": len(cases),
            "failed_cases": len(failures),
            "quality_passes": sum(case.get("quality_pass", False) for case in cases),
        },
        "cases": cases,
        "failures": failures,
    }
    (output / "manifest.json").write_text(
        json.dumps(output_manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": str(output / "manifest.json"), **output_manifest["coverage"]}, indent=2))
    return output_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", action="append")
    parser.add_argument("--test-manifest", action="append")
    parser.add_argument("--output", default="runs/trajectory-transport-probe")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--history-step", type=int, default=1408)
    parser.add_argument("--future-step", type=int, default=2048)
    parser.add_argument("--recovery-after", type=int, default=128)
    parser.add_argument("--self-check", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
