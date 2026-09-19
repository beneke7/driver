"""Fit a small root-held-out model on the passive trajectory corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import torch
from torch import nn


ROLES = ("embedding", "attention", "mlp", "norm", "head")
ROLE_GROUPS = (
    "role_gradient_norms",
    "role_update_norms",
    "role_parameter_norms",
    "adam_momentum_norms",
    "adam_variance_norms",
)
SCALARS = (
    "global_step",
    "global_tokens",
    "step_fraction",
    "loss",
    "gradient_norm",
    "update_norm",
    "momentum_alignment",
    "parameter_norm",
    "adam_momentum_norm",
    "adam_variance_norm",
    "global_lr_multiplier",
)
PHASES = ("immediate", "recovery", "late")


def _sha256(path: Path) -> str:
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


def _vector(row: dict[str, Any]) -> list[float]:
    features = row["features"]
    values = [float(features[name]) for name in SCALARS]
    for group in ROLE_GROUPS:
        values.extend(float(features[group][role]) for role in ROLES)
    architecture = row["architecture"]
    values.extend(
        float(architecture[name]) for name in ("width", "layers", "heads", "vocab_size", "context")
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("non-finite passive feature")
    return values


def _read_corpus(corpus: Path, horizons: tuple[int, ...]) -> list[dict[str, Any]]:
    rows_path = corpus / "trajectories.jsonl"
    rows: list[dict[str, Any]] = []
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if all(f"loss_delta_h{horizon}" in row["targets"] for horizon in horizons):
            row["vector"] = _vector(row)
            row["phase_index"] = PHASES.index(row["phase"])
            rows.append(row)
    if not rows:
        raise ValueError("corpus contains no rows for all requested horizons")
    return rows


class PassiveModel(nn.Module):
    def __init__(self, input_size: int, hidden: int, horizon_count: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.loss_head = nn.Linear(hidden, horizon_count)
        self.phase_head = nn.Linear(hidden, len(PHASES))

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(values)
        return self.loss_head(hidden), self.phase_head(hidden)


def _corr(prediction: torch.Tensor, target: torch.Tensor) -> float | None:
    if prediction.numel() < 2 or prediction.std() == 0 or target.std() == 0:
        return None
    return float(torch.corrcoef(torch.stack((prediction, target)))[0, 1].item())


def _self_check() -> None:
    model = PassiveModel(8, 4, 2)
    losses, phases = model(torch.zeros(3, 8))
    assert losses.shape == (3, 2)
    assert phases.shape == (3, 3)
    assert torch.isfinite(losses).all() and torch.isfinite(phases).all()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.self_check:
        _self_check()
        print("passive trajectory model self-check: ok")
        return {"self_check": "ok"}
    corpus = Path(args.corpus)
    horizons = tuple(sorted(set(args.horizon or (1, 8, 32, 128))))
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError("horizons must be positive")
    if not args.holdout_root:
        raise ValueError("at least one complete holdout root is required")
    rows = _read_corpus(corpus, horizons)
    holdouts = set(args.holdout_root)
    train = [row for row in rows if row["root_id"] not in holdouts]
    test = [row for row in rows if row["root_id"] in holdouts]
    if not train or not test:
        raise ValueError("root holdout produced an empty train or test split")
    device = torch.device(args.device)
    train_x = torch.tensor([row["vector"] for row in train], dtype=torch.float32)
    test_x = torch.tensor([row["vector"] for row in test], dtype=torch.float32)
    train_y = torch.tensor(
        [[row["targets"][f"loss_delta_h{horizon}"] for horizon in horizons] for row in train],
        dtype=torch.float32,
    )
    test_y = torch.tensor(
        [[row["targets"][f"loss_delta_h{horizon}"] for horizon in horizons] for row in test],
        dtype=torch.float32,
    )
    train_phase = torch.tensor([row["phase_index"] for row in train], dtype=torch.long)
    test_phase = torch.tensor([row["phase_index"] for row in test], dtype=torch.long)
    mean = train_x.mean(dim=0)
    scale = train_x.std(dim=0, unbiased=False).clamp_min(1e-6)
    target_mean = train_y.mean(dim=0)
    target_scale = train_y.std(dim=0, unbiased=False).clamp_min(1e-6)
    train_x = ((train_x - mean) / scale).to(device)
    test_x = ((test_x - mean) / scale).to(device)
    normalized_y = ((train_y - target_mean) / target_scale).to(device)
    train_phase = train_phase.to(device)
    test_y = test_y.to(device)
    test_phase = test_phase.to(device)
    torch.manual_seed(args.seed)
    model = PassiveModel(train_x.shape[1], args.hidden, len(horizons)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    for _ in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        prediction, phase_logits = model(train_x)
        loss = nn.functional.smooth_l1_loss(prediction, normalized_y)
        loss = loss + args.phase_weight * nn.functional.cross_entropy(phase_logits, train_phase)
        if not torch.isfinite(loss):
            raise FloatingPointError("passive model training produced a non-finite loss")
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        prediction, phase_logits = model(test_x)
        prediction = prediction * target_scale.to(device) + target_mean.to(device)
        phase_prediction = phase_logits.argmax(dim=1)
    baseline = train_y.mean(dim=0).to(device)
    metrics: dict[str, Any] = {
        "rows": len(rows),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_roots": sorted({row["root_id"] for row in train}),
        "test_roots": sorted({row["root_id"] for row in test}),
        "phase_accuracy": float((phase_prediction == test_phase).float().mean().item()),
        "horizons": list(horizons),
    }
    for index, horizon in enumerate(horizons):
        error = prediction[:, index] - test_y[:, index]
        metrics[f"h{horizon}_baseline_rmse"] = float(
            (test_y[:, index] - baseline[index]).square().mean().sqrt().item()
        )
        metrics[f"h{horizon}_rmse"] = float(error.square().mean().sqrt().item())
        metrics[f"h{horizon}_correlation"] = _corr(prediction[:, index], test_y[:, index])
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"choose a fresh output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "landscape-driver.passive-trajectory-model.v1",
        "causal_status": "passive_prediction_only",
        "corpus": str(corpus),
        "corpus_manifest_sha256": _sha256(corpus / "manifest.json"),
        "hidden": args.hidden,
        "epochs": args.epochs,
        "device": str(device),
        "metrics": metrics,
        "immutable": {
            "git_commit": _code_revision(),
            "model_sha256": _sha256(Path(__file__).resolve()),
        },
        "limitations": [
            "no counterfactual action outcomes",
            "no future optimizer-state target",
            "root split is required for transfer claims",
        ],
    }
    torch.save(
        {
            "model": model.state_dict(),
            "feature_mean": mean,
            "feature_scale": scale,
            "target_mean": target_mean,
            "target_scale": target_scale,
            "horizons": horizons,
            "hidden": args.hidden,
            "phase_names": PHASES,
        },
        output / "passive_model.pt",
    )
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="runs/passive-trajectory-corpus-local-v1")
    parser.add_argument("--holdout-root", action="append", default=[])
    parser.add_argument("--horizon", action="append", type=int, default=[])
    parser.add_argument("--output", default="runs/passive-trajectory-model")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--phase-weight", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--self-check", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
