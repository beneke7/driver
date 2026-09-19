"""Verify complete-checkpoint transport state consistency.

This is a diagnostic control, not a speedup claim.  A no-op continuation is
run to a future checkpoint, saved with the repository checkpoint primitive,
and resumed from that complete state.  Matching the direct continuation shows
that parameters, AdamW state, RNG, and data cursor are sufficient to resume;
the conservative ledger still charges the work used to create the future
state.
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from .checkpoints import load_checkpoint, save_checkpoint
from .decoder_benchmark import TokenStream, _flops
from .decoder_campaign import (
    _autocast,
    _evaluate,
    _model_and_optimizer,
    _set_group_lrs,
    _sync,
)
from .trajectory_transport_oracle import (
    _campaign_and_config as _transport_campaign_and_config,
    _file_sha256,
    _repo_path,
    _source_transition,
    _streams,
)


def _code_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _fast_noop_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    stream: TokenStream,
    target: Any,
    campaign: Any,
    device: torch.device,
) -> int:
    _set_group_lrs(optimizer, campaign)
    tokens, targets = stream.batch(
        batch_size=target.batch_size, context=target.context, device=device
    )
    optimizer.zero_grad(set_to_none=True)
    with _autocast(campaign, device):
        loss = model(tokens, targets)
    loss.backward()
    optimizer.step()
    return target.batch_size * target.context


def _advance(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    stream: TokenStream,
    target: Any,
    campaign: Any,
    device: torch.device,
    steps: int,
) -> int:
    consumed = 0
    for _ in range(steps):
        consumed += _fast_noop_step(
            model, optimizer, stream, target, campaign, device
        )
    return consumed


def _case(
    *,
    source_manifest: Path,
    manifest: dict[str, Any],
    case: dict[str, Any],
    future_step: int,
    recovery_after: int,
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    campaign, target = _transport_campaign_and_config(manifest["config"], case)
    parent_step = target.prefix_steps
    endpoint_step = target.prefix_steps + target.final_steps
    continuation_steps = future_step - parent_step
    remaining_steps = endpoint_step - future_step
    if not 0 < continuation_steps < target.final_steps:
        raise ValueError("future_step must lie inside the recorded continuation")
    if not 0 < recovery_after < remaining_steps:
        raise ValueError("recovery_after must lie inside the future continuation")
    train_values, validation_values, data_sha = _streams(
        campaign, landscape=target.landscape, seed=target.seed
    )
    if data_sha != case["data_sha256"]:
        raise ValueError(f"source data hash mismatch for {case['case_id']}")
    parent = _repo_path(case["parent_checkpoint"])
    config_sha = str(case["config_sha256"])
    source_transition = _source_transition(source_manifest, case["case_id"])
    case_key = f"{source_manifest.parent.name}-{case['case_id']}"
    case_output = output / case_key
    case_output.mkdir(parents=True, exist_ok=False)
    future_path = case_output / "future-complete.pt"
    code_sha = _code_revision()
    started = time.perf_counter()
    direct_model = direct_optimizer = None
    resumed_model = resumed_optimizer = None
    try:
        direct_model, direct_optimizer = _model_and_optimizer(target, device)
        parent_metadata = load_checkpoint(
            parent,
            model=direct_model,
            optimizer=direct_optimizer,
            config_sha256=config_sha,
            data_sha256=data_sha,
        )
        parent_cursor = int(parent_metadata["data_state"]["cursor"])
        direct_stream = TokenStream(train_values, cursor=parent_cursor)
        prefix_loss = _evaluate(
            direct_model,
            validation_values,
            target_config=target,
            campaign=campaign,
            device=device,
        )
        future_tokens = _advance(
            direct_model,
            direct_optimizer,
            direct_stream,
            target,
            campaign,
            device,
            continuation_steps,
        )
        future_loss = _evaluate(
            direct_model,
            validation_values,
            target_config=target,
            campaign=campaign,
            device=device,
        )
        future_checkpoint = save_checkpoint(
            future_path,
            model=direct_model,
            optimizer=direct_optimizer,
            data_state=direct_stream.state(),
            config_sha256=config_sha,
            data_sha256=data_sha,
            code_sha=code_sha,
            seed=target.seed,
            parent_sha256=case["parent_checkpoint_sha256"],
        )
        recovery_tokens = _advance(
            direct_model,
            direct_optimizer,
            direct_stream,
            target,
            campaign,
            device,
            recovery_after,
        )
        direct_recovery_loss = _evaluate(
            direct_model,
            validation_values,
            target_config=target,
            campaign=campaign,
            device=device,
        )
        direct_final_tokens = _advance(
            direct_model,
            direct_optimizer,
            direct_stream,
            target,
            campaign,
            device,
            remaining_steps - recovery_after,
        )
        direct_final_loss = _evaluate(
            direct_model,
            validation_values,
            target_config=target,
            campaign=campaign,
            device=device,
        )
        direct_wall = time.perf_counter() - started
        parameter_count = sum(parameter.numel() for parameter in direct_model.parameters())
        direct_tokens = future_tokens + recovery_tokens + direct_final_tokens
        direct_train_flops = _flops(target, direct_model, direct_tokens)
        evaluation_flops = 2.0 * parameter_count * target.batch_size * target.context
        direct_evaluation_flops = 4.0 * evaluation_flops
        resumed_evaluation_flops = 3.0 * evaluation_flops
        del direct_model, direct_optimizer
        direct_model = direct_optimizer = None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        resumed_model, resumed_optimizer = _model_and_optimizer(target, device)
        resume_started = time.perf_counter()
        future_metadata = load_checkpoint(
            future_checkpoint,
            model=resumed_model,
            optimizer=resumed_optimizer,
            config_sha256=config_sha,
            data_sha256=data_sha,
        )
        resumed_cursor = int(future_metadata["data_state"]["cursor"])
        resumed_stream = TokenStream(train_values, cursor=resumed_cursor)
        resumed_future_loss = _evaluate(
            resumed_model,
            validation_values,
            target_config=target,
            campaign=campaign,
            device=device,
        )
        _advance(
            resumed_model,
            resumed_optimizer,
            resumed_stream,
            target,
            campaign,
            device,
            recovery_after,
        )
        resumed_recovery_loss = _evaluate(
            resumed_model,
            validation_values,
            target_config=target,
            campaign=campaign,
            device=device,
        )
        _advance(
            resumed_model,
            resumed_optimizer,
            resumed_stream,
            target,
            campaign,
            device,
            remaining_steps - recovery_after,
        )
        resumed_final_loss = _evaluate(
            resumed_model,
            validation_values,
            target_config=target,
            campaign=campaign,
            device=device,
        )
        _sync(device)
        resume_wall = time.perf_counter() - resume_started
        resumed_tokens = remaining_steps * target.batch_size * target.context
        resumed_train_flops = _flops(target, resumed_model, resumed_tokens)
        conservative_candidate_flops = resumed_train_flops + _flops(
            target, resumed_model, continuation_steps * target.batch_size * target.context
        )
        prefix_flops = float(source_transition["before"]["compute_flops"])
        baseline_flops = prefix_flops + direct_train_flops + direct_evaluation_flops
        ideal_flops = prefix_flops + resumed_train_flops + resumed_evaluation_flops
        candidate_flops = prefix_flops + conservative_candidate_flops + resumed_evaluation_flops
        result = {
            "case_id": case["case_id"],
            "case_key": case_key,
            "landscape": case["landscape"],
            "seed": case["seed"],
            "target_parameters": case["target_parameters"],
            "source_manifest": str(source_manifest),
            "parent_checkpoint": str(parent),
            "parent_checkpoint_sha256": case["parent_checkpoint_sha256"],
            "future_checkpoint": str(future_path),
            "future_checkpoint_sha256": future_checkpoint.sha256,
            "future_checkpoint_metadata": future_checkpoint.metadata,
            "source_data_sha256": data_sha,
            "config_sha256": config_sha,
            "parent_cursor": parent_cursor,
            "future_cursor": int(future_metadata["data_state"]["cursor"]),
            "future_step": future_step,
            "recovery_after": recovery_after,
            "endpoint_step": endpoint_step,
            "direct": {
                "losses": {
                    "parent": prefix_loss,
                    "future": future_loss,
                    "recovery": direct_recovery_loss,
                    "final": direct_final_loss,
                },
                "tokens": direct_tokens,
                "wall_seconds": direct_wall,
                "train_flops": direct_train_flops,
            },
            "resumed": {
                "losses": {
                    "future": resumed_future_loss,
                    "recovery": resumed_recovery_loss,
                    "final": resumed_final_loss,
                },
                "tokens": resumed_tokens,
                "wall_seconds": resume_wall,
                "train_flops": resumed_train_flops,
            },
            "absolute_loss_gaps": {
                "future": resumed_future_loss - future_loss,
                "recovery": resumed_recovery_loss - direct_recovery_loss,
                "final": resumed_final_loss - direct_final_loss,
            },
            "max_absolute_loss_gap": max(
                abs(resumed_future_loss - future_loss),
                abs(resumed_recovery_loss - direct_recovery_loss),
                abs(resumed_final_loss - direct_final_loss),
            ),
            "resume_pass": max(
                abs(resumed_future_loss - future_loss),
                abs(resumed_recovery_loss - direct_recovery_loss),
                abs(resumed_final_loss - direct_final_loss),
            ) <= 1e-5,
            "baseline_total_flops": baseline_flops,
            "ideal_resume_total_flops": ideal_flops,
            "conservative_resume_total_flops": candidate_flops,
            "ideal_resume_flop_ratio": baseline_flops
            / max(1.0, prefix_flops + resumed_train_flops + evaluation_flops),
            "conservative_resume_flop_ratio": baseline_flops / max(1.0, candidate_flops),
            "oracle_only": True,
            "oracle_violations": ["future_state_created_by_real_noop_replay"],
            "failed": False,
        }
        del resumed_model, resumed_optimizer
        resumed_model = resumed_optimizer = None
        return result
    except Exception as exc:
        _sync(device)
        return {
            "case_id": case["case_id"],
            "source_manifest": str(source_manifest),
            "parent_checkpoint": str(parent),
            "failed": True,
            "failure": f"{type(exc).__name__}: {exc}",
        }
    finally:
        del direct_model, direct_optimizer
        del resumed_model, resumed_optimizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _self_check() -> None:
    assert 2304 - 1536 == 768
    assert 512 * 128 * 128 == 8_388_608
    assert _repo_path("research/OPERATING_CONTRACT.md").is_file()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.self_check:
        _self_check()
        print("full-state transport sanity self-check: ok")
        return {"self_check": "ok"}
    if not args.source_manifest:
        raise ValueError("at least one --source-manifest is required")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"choose a fresh output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cases: list[dict[str, Any]] = []
    for source_manifest_value in args.source_manifest:
        source_manifest = _repo_path(source_manifest_value)
        manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
        for case in manifest["cases"]:
            cases.append(
                _case(
                    source_manifest=source_manifest,
                    manifest=manifest,
                    case=case,
                    future_step=args.future_step,
                    recovery_after=args.recovery_after,
                    output=output,
                    device=device,
                )
            )
    failures = [case for case in cases if case.get("failed")]
    result_manifest = {
        "schema": "landscape-driver.full-state-transport-sanity.v1",
        "stage": "oracle_state_consistency",
        "promotion_status": "diagnostic_only",
        "oracle_only": True,
        "contract": {
            "baseline": "direct AdamW/no-op continuation from immutable parent",
            "candidate": "resume from a complete checkpoint created at future_step",
            "required_horizons": ["future", "recovery", "final"],
            "resume_tolerance": 1e-5,
            "cost_includes": [
                "shared prefix",
                "future-state creation",
                "resume continuation",
                "evaluation",
                "conservative charge for skipped future work",
            ],
        },
        "device": str(device),
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "future_step": args.future_step,
        "recovery_after": args.recovery_after,
        "source_manifests": [
            {
                "path": str(_repo_path(value)),
                "sha256": _file_sha256(_repo_path(value)),
            }
            for value in args.source_manifest
        ],
        "immutable": {
            "git_commit": _code_revision(),
            "code_sha256": _file_sha256(Path(__file__).resolve()),
        },
        "coverage": {
            "cases": len(cases),
            "failed_cases": len(failures),
            "resume_passes": sum(bool(case.get("resume_pass")) for case in cases),
        },
        "cases": cases,
        "failures": failures,
    }
    (output / "manifest.json").write_text(
        json.dumps(result_manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": str(output / "manifest.json"), **result_manifest["coverage"]}, indent=2))
    return result_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", action="append")
    parser.add_argument("--output", default="runs/full-state-transport-sanity")
    parser.add_argument("--future-step", type=int, default=2048)
    parser.add_argument("--recovery-after", type=int, default=128)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--self-check", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
