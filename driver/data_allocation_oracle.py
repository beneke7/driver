"""Measure a simple, cost-matched data-allocation oracle.

Each candidate starts from an immutable AdamW parent and consumes the same
number of tokens as a source-data no-op branch.  The only intervention is
switching the training stream between the two versioned local byte corpora.
This is an oracle screen for a data/work hypothesis, not a deployment claim:
the choice is made without a learned state model and the candidate still pays
for every token, evaluation, and branch.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from .checkpoints import load_checkpoint
from .decoder_benchmark import (
    TokenStream,
    _flops,
    _hash_bytes,
    _hash_tensor,
    make_byte_stream,
)
from .decoder_campaign import (
    _autocast,
    _evaluate,
    _model_and_optimizer,
    _set_group_lrs,
    _sync,
)
from .trajectory_transport_oracle import (
    _campaign_and_config,
    _repo_path,
    _source_transition,
)


HORIZONS = (128, 512, 768)
MIX_BLOCK_TOKENS = 16_512
DATA_FILES = {
    "data/raw/FineWeb-Edu-train-40m.txt": "data/raw/TinyStories-train-prefix.txt",
    "data/raw/TinyStories-train-prefix.txt": "data/raw/FineWeb-Edu-train-40m.txt",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _alternate_train_file(path: str) -> str:
    normalized = path.replace("\\", "/")
    try:
        return DATA_FILES[normalized]
    except KeyError as exc:
        raise ValueError(
            "data-allocation oracle only supports the two versioned local "
            f"byte corpora, got {path!r}"
        ) from exc


def _data_sha(train_values: torch.Tensor, validation_values: torch.Tensor) -> str:
    return _hash_bytes(
        _hash_tensor(train_values).encode() + _hash_tensor(validation_values).encode()
    )


def _mix_train_values(
    source_values: torch.Tensor,
    alternate_values: torch.Tensor,
    alternate_fraction: float,
) -> tuple[torch.Tensor, int]:
    if source_values.ndim != 1 or alternate_values.ndim != 1:
        raise ValueError("mixture streams must be one-dimensional")
    if source_values.dtype != torch.uint8 or alternate_values.dtype != torch.uint8:
        raise ValueError("mixture streams must contain uint8 tokens")
    if not 0.0 <= alternate_fraction <= 1.0:
        raise ValueError("alternate_fraction must lie in [0, 1]")
    if alternate_fraction in (0.0, 1.0):
        if alternate_fraction == 0.0:
            return source_values.clone(), 0
        if alternate_values.numel() < source_values.numel():
            raise ValueError("full alternate stream is shorter than source stream")
        return alternate_values[: source_values.numel()].clone(), source_values.numel()
    # Deliberate ceiling: fixed 16-step blocks test allocation without a
    # per-example utility model; upgrade only if this oracle has durable signal.
    period = 16
    alternate_blocks = round(period * alternate_fraction)
    if alternate_blocks <= 0 or alternate_blocks >= period:
        raise ValueError("fraction does not select a nontrivial block schedule")
    block_ids = torch.arange(source_values.numel()) // MIX_BLOCK_TOKENS
    mask = (block_ids % period) >= period - alternate_blocks
    selected = int(mask.sum().item())
    if alternate_values.numel() < selected:
        raise ValueError("alternate stream is shorter than selected mixture tokens")
    mixed = source_values.clone()
    mixed[mask] = alternate_values[:selected]
    return mixed, selected


def _fast_noop_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    stream: TokenStream,
    target: Any,
    campaign: Any,
    device: torch.device,
) -> int:
    """Run the noop step without telemetry that this oracle does not use."""
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


def _train_branch(
    *,
    parent: Path,
    campaign: Any,
    target: Any,
    train_values: torch.Tensor,
    validation_values: torch.Tensor,
    checkpoint_data_sha: str,
    config_sha: str,
    data_source: str,
    data_source_sha: str,
    start_cursor: int | None,
    device: torch.device,
) -> dict[str, Any]:
    started = time.perf_counter()
    model = optimizer = None
    try:
        model, optimizer = _model_and_optimizer(target, device)
        metadata = load_checkpoint(
            parent,
            model=model,
            optimizer=optimizer,
            config_sha256=config_sha,
            data_sha256=checkpoint_data_sha,
        )
        parent_cursor = int(metadata["data_state"]["cursor"])
        cursor = parent_cursor if start_cursor is None else int(start_cursor)
        stream = TokenStream(train_values, cursor=cursor)
        phase_metrics: list[dict[str, Any]] = []
        consumed_tokens = 0
        for step in range(HORIZONS[-1]):
            consumed = _fast_noop_step(
                model,
                optimizer,
                stream,
                target,
                campaign,
                device,
            )
            consumed_tokens += consumed
            if step + 1 in HORIZONS:
                phase_metrics.append(
                    {
                        "name": {128: "immediate", 512: "recovery", 768: "final"}[step + 1],
                        "global_step": target.prefix_steps + step + 1,
                        "loss": _evaluate(
                            model,
                            validation_values,
                            target_config=target,
                            campaign=campaign,
                            device=device,
                        ),
                        "tokens": consumed_tokens,
                    }
                )
        _sync(device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        evaluation_flops = (
            2.0
            * parameter_count
            * target.batch_size
            * target.context
            * len(HORIZONS)
        )
        target_flops = _flops(target, model, consumed_tokens)
        wall_seconds = time.perf_counter() - started
        return {
            "data_source": data_source,
            "data_source_sha256": data_source_sha,
            "start_cursor": cursor,
            "end_cursor": stream.cursor,
            "parent_cursor": parent_cursor,
            "consumed_tokens": consumed_tokens,
            "phase_metrics": phase_metrics,
            "target_flops": target_flops,
            "evaluation_flops": evaluation_flops,
            "compute_flops": target_flops + evaluation_flops,
            "wall_seconds": wall_seconds,
            "failed": False,
        }
    except Exception as exc:
        _sync(device)
        return {
            "data_source": data_source,
            "data_source_sha256": data_source_sha,
            "failed": True,
            "failure": f"{type(exc).__name__}: {exc}",
        }
    finally:
        del model, optimizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _run_case(
    *,
    source_manifest: Path,
    manifest: dict[str, Any],
    case: dict[str, Any],
    alternate_fraction: float,
    device: torch.device,
) -> dict[str, Any]:
    campaign, target = _campaign_and_config(manifest["config"], case)
    if tuple(HORIZONS) != (128, 512, campaign.final_steps):
        raise ValueError(
            "the oracle expects the existing 128/512/768 continuation; "
            f"got final_steps={campaign.final_steps}"
        )
    parent = _repo_path(case["parent_checkpoint"])
    config_sha = str(case["config_sha256"])
    source_train_file = str(campaign.train_file)
    alternate_train_file = _alternate_train_file(source_train_file)
    source_train_path = _repo_path(source_train_file)
    alternate_train_path = (
        source_train_path
        if alternate_fraction < 1.0
        else _repo_path(alternate_train_file)
    )
    train_tokens = (
        campaign.final_steps * target.batch_size * (target.context + 1) + 1
    )
    validation_tokens = campaign.validation_batches * target.batch_size * (
        target.context + 1
    )
    source_train_values = make_byte_stream(
        source_train_path,
        seed=target.seed,
        length=(campaign.prefix_steps + campaign.final_steps + 2)
        * target.batch_size
        * (target.context + 1),
    )
    validation_values = make_byte_stream(
        _repo_path(campaign.validation_file or source_train_file),
        seed=target.seed + 1_000_000,
        length=validation_tokens,
    )
    checkpoint_data_sha = _data_sha(source_train_values, validation_values)
    if checkpoint_data_sha != case["data_sha256"]:
        raise ValueError(f"reconstructed source data hash does not match {case['case_id']}")
    parent_cursor = campaign.prefix_steps * target.batch_size * (target.context + 1)
    alternate_interval: list[int] | None = None
    if alternate_fraction < 1.0:
        source_continuation = source_train_values[parent_cursor : parent_cursor + train_tokens]
        alternate_count = round(source_continuation.numel() * alternate_fraction)
        raw_length = alternate_train_path.stat().st_size
        alternate_offset = raw_length - alternate_count - 1
        baseline_end = (target.seed % raw_length) + parent_cursor + train_tokens
        if alternate_offset <= baseline_end or alternate_offset + alternate_count > raw_length:
            raise ValueError(
                "same-corpus alternate interval overlaps the target continuation "
                f"or exceeds the file: alternate=[{alternate_offset},{alternate_offset + alternate_count}), "
                f"target=[{target.seed % raw_length + parent_cursor},{baseline_end}), bytes={raw_length}"
            )
        alternate_train_values = make_byte_stream(
            alternate_train_path, seed=alternate_offset, length=alternate_count
        )
        mixed_continuation, selected = _mix_train_values(
            source_continuation, alternate_train_values, alternate_fraction
        )
        if selected != alternate_count:
            alternate_interval = [alternate_offset, alternate_offset + selected]
        else:
            alternate_interval = [alternate_offset, alternate_offset + alternate_count]
        candidate_train_values = torch.cat(
            (
                source_train_values[:parent_cursor],
                mixed_continuation,
                source_train_values[parent_cursor + train_tokens :],
            )
        )
        candidate_start_cursor = parent_cursor
    else:
        alternate_train_values = make_byte_stream(
            alternate_train_path, seed=target.seed, length=train_tokens
        )
        candidate_train_values = alternate_train_values
        candidate_start_cursor = 0
    alternate_data_sha = _data_sha(alternate_train_values, validation_values)
    candidate_data_sha = _data_sha(candidate_train_values, validation_values)
    source_transition = _source_transition(source_manifest, case["case_id"])
    baseline = _train_branch(
        parent=parent,
        campaign=campaign,
        target=target,
        train_values=source_train_values,
        validation_values=validation_values,
        checkpoint_data_sha=checkpoint_data_sha,
        config_sha=config_sha,
        data_source=source_train_file,
        data_source_sha=checkpoint_data_sha,
        start_cursor=None,
        device=device,
    )
    candidate = _train_branch(
        parent=parent,
        campaign=campaign,
        target=target,
        train_values=candidate_train_values,
        validation_values=validation_values,
        checkpoint_data_sha=checkpoint_data_sha,
        config_sha=config_sha,
        data_source=(
            alternate_train_file
            if alternate_fraction == 1.0
            else f"{source_train_file}:disjoint_block_mixture"
        ),
        data_source_sha=candidate_data_sha,
        start_cursor=candidate_start_cursor,
        device=device,
    )
    shared_prefix_flops = float(source_transition["before"]["compute_flops"])
    shared_prefix_wall = float(
        source_transition["before"]["features"].get("prefix_seconds", 0.0)
    )
    for branch in (baseline, candidate):
        if not branch["failed"]:
            branch["total_compute_flops_with_shared_prefix"] = (
                shared_prefix_flops + branch["compute_flops"]
            )
            branch["total_wall_seconds_with_shared_prefix"] = (
                shared_prefix_wall + branch["wall_seconds"]
            )
    result: dict[str, Any] = {
        "case_id": case["case_id"],
        "landscape": case["landscape"],
        "seed": case["seed"],
        "target_parameters": case["target_parameters"],
        "source_manifest": str(source_manifest),
        "parent_checkpoint": str(parent),
        "parent_checkpoint_sha256": case["parent_checkpoint_sha256"],
        "source_data_sha256": checkpoint_data_sha,
        "alternate_data_sha256": alternate_data_sha,
        "candidate_data_sha256": candidate_data_sha,
        "alternate_fraction": alternate_fraction,
        "alternate_interval": alternate_interval,
        "candidate_start_cursor": candidate_start_cursor,
        "config_sha256": config_sha,
        "shared_prefix_flops": float(
            source_transition["before"]["compute_flops"]
        ),
        "shared_prefix_wall_seconds": float(
            source_transition["before"]["features"].get("prefix_seconds", 0.0)
        ),
        "baseline": baseline,
        "candidate": candidate,
        "recorded_noop_final_loss": source_transition["after"]["loss"],
    } 
    if not baseline["failed"] and not candidate["failed"]:
        base_by_name = {row["name"]: row for row in baseline["phase_metrics"]}
        candidate_by_name = {row["name"]: row for row in candidate["phase_metrics"]}
        deltas = {
            name: candidate_by_name[name]["loss"] - base_by_name[name]["loss"]
            for name in ("immediate", "recovery", "final")
        }
        result["deltas_vs_recomputed_noop"] = deltas
        result["quality_pass"] = deltas["final"] <= 0.0
        result["same_token_budget"] = (
            baseline["consumed_tokens"] == candidate["consumed_tokens"]
        )
    else:
        result["quality_pass"] = False
        result["same_token_budget"] = False
    return result


def _self_check() -> None:
    assert _alternate_train_file("data/raw/FineWeb-Edu-train-40m.txt").endswith(
        "TinyStories-train-prefix.txt"
    )
    assert _alternate_train_file("data/raw/TinyStories-train-prefix.txt").endswith(
        "FineWeb-Edu-train-40m.txt"
    )
    assert HORIZONS == (128, 512, 768)
    source = torch.zeros(16 * MIX_BLOCK_TOKENS, dtype=torch.uint8)
    alternate = torch.ones(4 * MIX_BLOCK_TOKENS, dtype=torch.uint8)
    mixed, selected = _mix_train_values(source, alternate, 0.25)
    assert selected == alternate.numel()
    assert int(mixed.sum()) == alternate.numel()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.self_check:
        _self_check()
        print("data allocation oracle self-check: ok")
        return {"self_check": "ok"}
    if not args.source_manifest:
        raise ValueError("at least one --source-manifest is required")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"choose a fresh --output; {output} is not empty")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cases: list[dict[str, Any]] = []
    for source_manifest_value in args.source_manifest:
        source_manifest = _repo_path(source_manifest_value)
        manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
        for case in manifest["cases"]:
            cases.append(
                _run_case(
                    source_manifest=source_manifest,
                    manifest=manifest,
                    case=case,
                    alternate_fraction=args.alternate_fraction,
                    device=device,
                )
            )
    failures = [
        result
        for case in cases
        for result in (case["baseline"], case["candidate"])
        if result["failed"]
    ]
    output_manifest = {
        "schema": "landscape-driver.data-allocation-oracle.v1",
        "stage": "oracle_ceiling",
        "promotion_status": "diagnostic_only",
        "oracle_only": True,
        "hypothesis": (
            "switching among the two local training corpora can improve durable "
            "capability at equal token cost"
            if args.alternate_fraction == 1.0
            else "a fixed same-corpus block mixture can improve durable capability at equal token cost"
        ),
        "alternate_fraction": args.alternate_fraction,
        "mixture_block_tokens": MIX_BLOCK_TOKENS,
        "contract": {
            "baseline": "recomputed AdamW/no-op continuation from each immutable parent",
            "quality": "candidate final validation loss <= recomputed no-op final loss",
            "horizons": ["immediate", "recovery", "final"],
            "cost_includes": [
                "shared prefix",
                "candidate training",
                "all candidate evaluations",
                "all baseline evaluations",
                "data exposure",
            ],
        },
        "device": str(device),
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
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
            "failed_branches": len(failures),
            "candidate_passes": sum(bool(case["quality_pass"]) for case in cases),
        },
        "cases": cases,
        "failures": failures,
    }
    (output / "manifest.json").write_text(
        json.dumps(output_manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"manifest": str(output / "manifest.json"), **output_manifest["coverage"]}, indent=2))
    return output_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", action="append")
    parser.add_argument("--output", default="runs/data-allocation-oracle")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--alternate-fraction",
        type=float,
        default=1.0,
        help="fraction of candidate continuation replaced by same-corpus tail data; values below 1 use the disjoint block mixture screen",
    )
    parser.add_argument("--self-check", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
