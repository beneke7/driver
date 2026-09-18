"""Immutable torch checkpoints for matched intervention branches."""

from __future__ import annotations

import hashlib
import os
import random
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


CHECKPOINT_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_hash(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be hexadecimal") from exc
    return value


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: Mapping[str, Any]) -> None:
    if not isinstance(state, Mapping) or "python" not in state or "torch" not in state:
        raise ValueError("checkpoint is missing complete RNG state")
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available():
        cuda_state = state.get("cuda")
        if cuda_state is None:
            raise ValueError("CUDA checkpoint is missing CUDA RNG state")
        torch.cuda.set_rng_state_all(cuda_state)


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    sha256: str
    metadata: dict[str, Any]


def save_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_state: Mapping[str, Any],
    config_sha256: str,
    data_sha256: str,
    code_sha: str,
    seed: int,
    parent_sha256: str | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
) -> Checkpoint:
    """Atomically write a complete branch source checkpoint."""

    config_sha256 = _required_hash(config_sha256, "config_sha256")
    data_sha256 = _required_hash(data_sha256, "data_sha256")
    if not isinstance(code_sha, str) or not code_sha.strip():
        raise ValueError("code_sha must be non-empty")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if parent_sha256 is not None:
        _required_hash(parent_sha256, "parent_sha256")
    if not isinstance(data_state, Mapping):
        raise ValueError("data_state must be a mapping")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "version": CHECKPOINT_VERSION,
        "config_sha256": config_sha256,
        "data_sha256": data_sha256,
        "code_sha": code_sha,
        "seed": seed,
        "parent_sha256": parent_sha256,
        "data_state": dict(data_state),
    }
    payload = {
        "metadata": metadata,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "rng": _rng_state(),
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return Checkpoint(target, _sha256(target), metadata)


def load_checkpoint(
    checkpoint: Checkpoint | str | os.PathLike[str],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config_sha256: str,
    data_sha256: str,
    scheduler: Any | None = None,
    scaler: Any | None = None,
) -> dict[str, Any]:
    """Validate and restore a checkpoint, failing closed on mismatched state."""

    record = checkpoint if isinstance(checkpoint, Checkpoint) else None
    path = record.path if record is not None else Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(path)
    actual_sha = _sha256(path)
    if record is not None and actual_sha != record.sha256:
        raise ValueError("checkpoint changed after it was recorded")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported or incomplete checkpoint metadata")
    if metadata.get("config_sha256") != _required_hash(config_sha256, "config_sha256"):
        raise ValueError("configuration hash mismatch")
    if metadata.get("data_sha256") != _required_hash(data_sha256, "data_sha256"):
        raise ValueError("data hash mismatch")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        if payload.get("scheduler") is None:
            raise ValueError("checkpoint is missing scheduler state")
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None:
        if payload.get("scaler") is None:
            raise ValueError("checkpoint is missing scaler state")
        scaler.load_state_dict(payload["scaler"])
    _restore_rng(payload["rng"])
    return dict(metadata)
