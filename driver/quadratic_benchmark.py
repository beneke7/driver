"""Batched first experiment for the optimization track.

This is a mechanism benchmark, not a language-model result.  The target is a
family of positive-definite quadratic training landscapes.  AdamW is tuned on
development landscapes; the driver estimates a local Hessian from gradient
probes and applies a Newton correction.  Probe gradients and the linear solve
are charged to the driver.

Run with the CUDA-enabled Python environment, for example:

    /home/v/proj/bene/aerial-drop-tdk/.venv/bin/python \
        -m driver.quadratic_benchmark --output runs/quadratic-first
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError as exc:  # pragma: no cover - depends on the selected environment
    raise SystemExit(
        "PyTorch is required; use the existing CUDA environment under "
        "/home/v/proj/bene/aerial-drop-tdk/.venv"
    ) from exc

from .core import Action, Archive, Observation, Transition


FAMILIES = ("diagonal", "rotated", "two_block")
DEFAULT_LR_GRID = (0.3, 0.5, 0.7, 1.0)


@dataclass(frozen=True)
class Case:
    case_id: str
    split: str
    family: str
    seed: int
    condition: float


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def cases(split: str, seed_count: int) -> list[Case]:
    if split not in {"development", "heldout"}:
        raise ValueError("split must be development or heldout")
    result: list[Case] = []
    seed_offset = 0 if split == "development" else 10_000
    conditions = (3_000.0, 10_000.0, 30_000.0, 100_000.0)
    for family_index, family in enumerate(FAMILIES):
        for index in range(seed_count):
            seed = seed_offset + family_index * 1_000 + index
            condition = conditions[(index + family_index) % len(conditions)]
            result.append(
                Case(
                    case_id=f"{split}-{family}-{index}",
                    split=split,
                    family=family,
                    seed=seed,
                    condition=condition,
                )
            )
    return result


def _generator(seed: int, device: torch.device) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def make_batch(
    suite: list[Case], *, dimension: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    matrices: list[torch.Tensor] = []
    starts: list[torch.Tensor] = []
    for case in suite:
        generator = _generator(case.seed, device)
        if case.family == "diagonal":
            eigenvalues = torch.logspace(
                0.0,
                math.log10(case.condition),
                dimension,
                device=device,
                dtype=dtype,
            )
            matrix = torch.diag(eigenvalues)
        else:
            rotation, _ = torch.linalg.qr(
                torch.randn(
                    dimension,
                    dimension,
                    generator=generator,
                    device=device,
                    dtype=dtype,
                )
            )
            if case.family == "two_block":
                half = dimension // 2
                eigenvalues = torch.cat(
                    [
                        torch.ones(half, device=device, dtype=dtype),
                        torch.full(
                            (dimension - half,),
                            case.condition,
                            device=device,
                            dtype=dtype,
                        ),
                    ]
                )
            else:
                eigenvalues = torch.logspace(
                    0.0,
                    math.log10(case.condition),
                    dimension,
                    device=device,
                    dtype=dtype,
                )
            matrix = rotation @ torch.diag(eigenvalues) @ rotation.T
        matrices.append(matrix)
        starts.append(torch.randn(dimension, generator=generator, device=device, dtype=dtype))
    return torch.stack(matrices), torch.stack(starts)


def objective(matrix: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
    product = torch.bmm(matrix, parameters.unsqueeze(-1)).squeeze(-1)
    return 0.5 * (parameters * product).sum(dim=-1)


def gradient(matrix: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
    return torch.bmm(matrix, parameters.unsqueeze(-1)).squeeze(-1)


def _timed_start(device: torch.device) -> float:
    _sync(device)
    return time.perf_counter()


def _timed_end(device: torch.device, started: float) -> float:
    _sync(device)
    return time.perf_counter() - started


def adam_batch(
    matrix: torch.Tensor,
    initial: torch.Tensor,
    *,
    learning_rate: float | torch.Tensor,
    threshold: float,
    max_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return first threshold step, final loss, and synchronized wall time."""

    device = matrix.device
    parameters = initial.clone()
    first = objective(matrix, parameters)
    reached = torch.full((parameters.shape[0],), -1, dtype=torch.int64, device=device)
    moment = torch.zeros_like(parameters)
    second = torch.zeros_like(parameters)
    beta1, beta2 = 0.9, 0.999
    learning_rate_tensor = torch.as_tensor(
        learning_rate, device=device, dtype=parameters.dtype
    )
    if learning_rate_tensor.ndim == 0:
        learning_rate_tensor = learning_rate_tensor.reshape(1, 1)
    elif learning_rate_tensor.ndim == 1:
        learning_rate_tensor = learning_rate_tensor[:, None]
    else:
        raise ValueError("learning_rate must be a scalar or one value per batch item")
    if learning_rate_tensor.shape[0] not in {1, parameters.shape[0]}:
        raise ValueError("learning_rate batch size does not match parameters")
    started = _timed_start(device)
    with torch.no_grad():
        for step in range(1, max_steps + 1):
            active = reached < 0
            if not bool(active.any()):
                break
            current_gradient = gradient(matrix, parameters)
            next_moment = beta1 * moment + (1.0 - beta1) * current_gradient
            next_second = beta2 * second + (1.0 - beta2) * current_gradient.square()
            bias1 = 1.0 - beta1**step
            bias2 = 1.0 - beta2**step
            update = learning_rate_tensor * (next_moment / bias1) / (
                (next_second / bias2).sqrt() + 1e-8
            )
            parameters = torch.where(active[:, None], parameters - update, parameters)
            moment = torch.where(active[:, None], next_moment, moment)
            second = torch.where(active[:, None], next_second, second)
            current_loss = objective(matrix, parameters)
            newly_reached = active & (current_loss <= first * threshold)
            reached = torch.where(newly_reached, torch.as_tensor(step, device=device), reached)
    elapsed = _timed_end(device, started)
    return reached, objective(matrix, parameters), elapsed


def newton_probe_batch(
    matrix: torch.Tensor,
    initial: torch.Tensor,
    *,
    threshold: float,
    probe_scale: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use a cheap diagonal probe, then fall back to a full Hessian probe."""

    device = matrix.device
    dimension = initial.shape[-1]
    started = _timed_start(device)
    with torch.no_grad():
        initial_loss = objective(matrix, initial)
        current_gradient = gradient(matrix, initial)
        scale = initial.square().mean(dim=-1).sqrt().clamp_min(1.0) * probe_scale
        diagonal_probe = initial + scale[:, None]
        diagonal_gradient = gradient(matrix, diagonal_probe)
        diagonal = (diagonal_gradient - current_gradient) / scale[:, None]
        diagonal_ok = torch.isfinite(diagonal).all(dim=-1) & (diagonal > 0).all(dim=-1)
        diagonal_parameters = initial - current_gradient / diagonal.clamp_min(1e-8)
        diagonal_loss = objective(matrix, diagonal_parameters)
        diagonal_reached = (
            diagonal_ok
            & torch.isfinite(diagonal_loss)
            & (diagonal_loss <= initial_loss * threshold)
        )

        final_parameters = diagonal_parameters.clone()
        final_loss = diagonal_loss.clone()
        reached = diagonal_reached.clone()
        full_solve = ~diagonal_reached
        if bool(full_solve.any()):
            subset_matrix = matrix[full_solve]
            subset_initial = initial[full_solve]
            subset_scale = scale[full_solve]
            subset_gradient = current_gradient[full_solve]
            identity = torch.eye(dimension, device=device, dtype=initial.dtype)
            probes = subset_initial[:, None, :] + subset_scale[:, None, None] * identity[None, :, :]
            probe_gradients = torch.einsum("bij,bkj->bki", subset_matrix, probes)
            hessian = (probe_gradients - subset_gradient[:, None, :]).transpose(1, 2)
            hessian = hessian / subset_scale[:, None, None]
            hessian = 0.5 * (hessian + hessian.transpose(1, 2))
            correction = torch.linalg.solve(
                hessian, subset_gradient.unsqueeze(-1)
            ).squeeze(-1)
            subset_parameters = subset_initial - correction
            subset_loss = objective(subset_matrix, subset_parameters)
            subset_reached = torch.isfinite(subset_loss) & (
                subset_loss <= initial_loss[full_solve] * threshold
            )
            final_parameters[full_solve] = subset_parameters
            final_loss[full_solve] = subset_loss
            reached[full_solve] = subset_reached
    elapsed = _timed_end(device, started)
    calls = torch.where(
        full_solve,
        torch.as_tensor(dimension + 3, dtype=torch.int64, device=device),
        torch.as_tensor(2, dtype=torch.int64, device=device),
    )
    return calls, final_loss, reached, full_solve, torch.tensor(elapsed, device=device)


def estimated_flops(
    method: str,
    dimension: int,
    steps: torch.Tensor,
    *,
    full_solve: torch.Tensor | None = None,
) -> torch.Tensor:
    gradient_flops = 2.0 * dimension * dimension
    if method == "adamw":
        optimizer_flops = 20.0 * dimension
        return steps.to(torch.float64) * (gradient_flops + optimizer_flops)
    if method == "hessian_probe":
        solve_flops = (2.0 / 3.0) * dimension**3
        solve = (
            torch.ones_like(steps, dtype=torch.float64)
            if full_solve is None
            else full_solve.to(torch.float64)
        )
        return steps.to(torch.float64) * gradient_flops + solve * solve_flops
    raise ValueError(f"unknown method: {method}")


def threshold_steps(
    values: torch.Tensor, *, max_steps: int, threshold_step: int | None = None
) -> torch.Tensor:
    """Replace an unsuccessful sentinel with a pre-registered failure cap."""

    cap = max_steps if threshold_step is None else threshold_step
    return torch.where(values > 0, values, torch.as_tensor(cap, device=values.device))


def _median_score(values: torch.Tensor, cap: int) -> float:
    capped = threshold_steps(values, max_steps=cap).detach().cpu().tolist()
    return float(statistics.median(capped))


def tune_baseline(
    matrix: torch.Tensor,
    initial: torch.Tensor,
    *,
    learning_rates: tuple[float, ...],
    threshold: float,
    max_steps: int,
) -> tuple[float, dict[str, Any]]:
    count = len(learning_rates)
    tiled_matrix = matrix.repeat(count, 1, 1)
    tiled_initial = initial.repeat(count, 1)
    tiled_rates = torch.tensor(
        learning_rates, device=matrix.device, dtype=matrix.dtype
    ).repeat_interleave(matrix.shape[0])
    reached, _, _ = adam_batch(
        tiled_matrix,
        tiled_initial,
        learning_rate=tiled_rates,
        threshold=threshold,
        max_steps=max_steps,
    )
    reached = reached.reshape(count, matrix.shape[0])
    scores: dict[float, float] = {}
    per_rate_steps: dict[str, list[int]] = {}
    for index, learning_rate in enumerate(learning_rates):
        values = threshold_steps(reached[index], max_steps=max_steps).detach().cpu().tolist()
        scores[learning_rate] = float(statistics.median(values))
        per_rate_steps[str(learning_rate)] = [int(value) for value in values]
    selected = min(learning_rates, key=lambda value: (scores[value], value))
    return selected, {"scores": scores, "steps": per_rate_steps}


def _repeat_batch(
    matrix: torch.Tensor, initial: torch.Tensor, replicas: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if replicas < 1:
        raise ValueError("replicas must be positive")
    return (
        matrix.repeat_interleave(replicas, dim=0),
        initial.repeat_interleave(replicas, dim=0),
    )


def _observation(step: int, loss: float, flops: float) -> Observation:
    return Observation(step=step, tokens=0, loss=loss, compute_flops=flops)


def write_evidence(
    output: Path,
    suite: list[Case],
    *,
    threshold: float,
    baseline_reached: torch.Tensor,
    baseline_steps: torch.Tensor,
    baseline_losses: torch.Tensor,
    baseline_flops: torch.Tensor,
    driver_calls: torch.Tensor,
    driver_losses: torch.Tensor,
    driver_reached: torch.Tensor,
    driver_flops: torch.Tensor,
    initial_losses: torch.Tensor,
    baseline_wall: float,
    driver_wall: float,
    baseline_learning_rate: float,
    dimension: int,
    max_steps: int,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    archive = Archive(output / "transitions.jsonl")
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(suite):
        baseline_step = int(baseline_steps[index].item())
        driver_call = int(driver_calls[index].item())
        baseline_cost = float(baseline_flops[index].item())
        driver_cost = float(driver_flops[index].item())
        speedup = baseline_cost / driver_cost if driver_cost else 0.0
        row = {
            "case_id": case.case_id,
            "split": case.split,
            "family": case.family,
            "seed": case.seed,
            "condition": case.condition,
            "baseline_steps": baseline_step,
            "baseline_reached": bool(baseline_reached[index].item()),
            "driver_gradient_calls": driver_call,
            "baseline_loss": float(baseline_losses[index].item()),
            "driver_loss": float(driver_losses[index].item()),
            "driver_reached": bool(driver_reached[index].item()),
            "estimated_speedup": speedup,
        }
        rows.append(row)
        before = _observation(0, float(initial_losses[index].item()), 0.0)
        for method, action_kind, steps, loss, flops in (
            ("adamw", "noop", baseline_step, baseline_losses[index], baseline_cost),
            (
                "hessian_probe",
                "structured_correction",
                driver_call,
                driver_losses[index],
                driver_cost,
            ),
        ):
            archive.append(
                Transition(
                    transition_id=f"{case.case_id}:{method}",
                    run_id=case.case_id,
                    parent_id=None,
                    before=before,
                    action=Action(action_kind),
                    after=_observation(
                        steps,
                        float(loss.item()),
                        flops,
                    ),
                    compute_flops=flops,
                    wall_seconds=(baseline_wall if method == "adamw" else driver_wall)
                    / max(1, len(suite)),
                    reward_task=float(initial_losses[index].item() - loss.item()),
                    learning_progress=float(initial_losses[index].item() - loss.item()),
                    accepted=(
                        bool(baseline_reached[index].item())
                        if method == "adamw"
                        else bool(driver_reached[index].item())
                    ),
                    failure=(
                        None
                        if (
                            bool(baseline_reached[index].item())
                            if method == "adamw"
                            else bool(driver_reached[index].item())
                        )
                        else "threshold_not_reached"
                    ),
                    metadata={
                        "landscape": case.family,
                        "seed": case.seed,
                        "split": case.split,
                        "condition": case.condition,
                        "threshold": threshold,
                        "baseline_learning_rate": baseline_learning_rate,
                        "dimension": dimension,
                        "max_steps": max_steps,
                        "matched_group": case.case_id,
                        "source_checkpoint_sha256": "synthetic_initial_state",
                        "code_revision": _revision(),
                    },
                )
            )
    count = archive.validate()
    summary = {
        "threshold": threshold,
        "dimension": dimension,
        "baseline_learning_rate": baseline_learning_rate,
        "baseline_wall_seconds": baseline_wall,
        "driver_wall_seconds": driver_wall,
        "wall_speedup": baseline_wall / driver_wall if driver_wall else 0.0,
        "archive_records": count,
        "rows": rows,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    dtype = torch.float32
    development = cases("development", args.development_seeds)
    heldout = cases("heldout", args.heldout_seeds)
    development_matrix, development_initial = make_batch(
        development, dimension=args.dimension, device=device, dtype=dtype
    )
    learning_rate, tuning = tune_baseline(
        development_matrix,
        development_initial,
        learning_rates=tuple(args.learning_rates),
        threshold=args.threshold,
        max_steps=args.max_steps,
    )
    heldout_matrix, heldout_initial = make_batch(
        heldout, dimension=args.dimension, device=device, dtype=dtype
    )
    timed_matrix, timed_initial = _repeat_batch(
        heldout_matrix, heldout_initial, args.timing_replicas
    )
    baseline_steps_timed, baseline_losses_timed, baseline_wall = adam_batch(
        timed_matrix,
        timed_initial,
        learning_rate=learning_rate,
        threshold=args.threshold,
        max_steps=args.max_steps,
    )
    (
        driver_calls_timed,
        driver_losses_timed,
        driver_reached_timed,
        driver_full_solve_timed,
        driver_wall_tensor,
    ) = (
        newton_probe_batch(
            timed_matrix,
            timed_initial,
            threshold=args.threshold,
            probe_scale=args.probe_scale,
        )
    )
    del baseline_steps_timed
    representatives = torch.arange(
        0, len(heldout) * args.timing_replicas, args.timing_replicas, device=device
    )
    baseline_raw_steps, baseline_losses, _ = adam_batch(
        heldout_matrix,
        heldout_initial,
        learning_rate=learning_rate,
        threshold=args.threshold,
        max_steps=args.max_steps,
    )
    baseline_reached = baseline_raw_steps > 0
    baseline_steps = threshold_steps(baseline_raw_steps, max_steps=args.max_steps)
    driver_calls = driver_calls_timed[representatives]
    driver_losses = driver_losses_timed[representatives]
    driver_reached = driver_reached_timed[representatives]
    driver_full_solve = driver_full_solve_timed[representatives]
    initial_losses = objective(heldout_matrix, heldout_initial)
    baseline_flops = estimated_flops("adamw", args.dimension, baseline_steps)
    driver_flops = estimated_flops(
        "hessian_probe", args.dimension, driver_calls, full_solve=driver_full_solve
    )
    output = Path(args.output)
    summary = write_evidence(
        output,
        heldout,
        threshold=args.threshold,
        baseline_reached=baseline_reached,
        baseline_steps=baseline_steps,
        baseline_losses=baseline_losses,
        baseline_flops=baseline_flops,
        driver_calls=driver_calls,
        driver_losses=driver_losses,
        driver_reached=driver_reached,
        driver_flops=driver_flops,
        initial_losses=initial_losses,
        baseline_wall=baseline_wall / max(1, args.timing_replicas),
        driver_wall=float(driver_wall_tensor.item()) / max(1, args.timing_replicas),
        baseline_learning_rate=learning_rate,
        dimension=args.dimension,
        max_steps=args.max_steps,
    )
    manifest = {
        "code_revision": _revision(),
        "device": str(device),
        "torch": torch.__version__,
        "dimension": args.dimension,
        "threshold": args.threshold,
        "max_steps": args.max_steps,
        "development_cases": [asdict(case) for case in development],
        "heldout_cases": [asdict(case) for case in heldout],
        "timing_replicas": args.timing_replicas,
        "probe_scale": args.probe_scale,
        "tuning": tuning,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    ratios = [
        row["estimated_speedup"]
        for row in summary["rows"]
        if row["driver_reached"] and row["baseline_reached"]
    ]
    print(f"device={device} torch={torch.__version__}")
    print(f"baseline_learning_rate={learning_rate}")
    print(
        f"heldout_cases={len(summary['rows'])} "
        f"matched_threshold_cases={len(ratios)}"
    )
    if ratios:
        print(
            "estimated_flop_speedup="
            f"median={statistics.median(ratios):.2f} min={min(ratios):.2f} "
            f"fraction_ge_10={sum(r >= 10 for r in ratios) / len(ratios):.2f}"
        )
    print(f"wall_speedup={summary['wall_speedup']:.2f}")
    print(f"evidence={output}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/quadratic-first")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dimension", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=1e-6)
    parser.add_argument("--max-steps", type=int, default=5_000)
    parser.add_argument("--development-seeds", type=int, default=4)
    parser.add_argument("--heldout-seeds", type=int, default=3)
    parser.add_argument("--timing-replicas", type=int, default=64)
    parser.add_argument("--probe-scale", type=float, default=0.1)
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=list(DEFAULT_LR_GRID),
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
