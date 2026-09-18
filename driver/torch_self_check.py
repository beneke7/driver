"""Verify exact matched-branch checkpoint restoration."""

from __future__ import annotations

import hashlib
import random
import tempfile
from pathlib import Path

import torch

from .checkpoints import load_checkpoint, save_checkpoint


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> None:
    random.seed(19)
    torch.manual_seed(19)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(19)
    config_sha = _digest("config-v1")
    data_sha = _digest("data-v1")
    with tempfile.TemporaryDirectory() as directory:
        checkpoint_path = Path(directory) / "parent.pt"
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        inputs = torch.randn(8, 4)
        target = torch.randn(8, 2)
        loss = (model(inputs) - target).square().mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        expected_model = {key: value.detach().clone() for key, value in model.state_dict().items()}
        checkpoint = save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            data_state={"cursor": 8},
            config_sha256=config_sha,
            data_sha256=data_sha,
            code_sha="self-check",
            seed=19,
        )
        expected_rng = torch.rand(4)

        model_a = torch.nn.Linear(4, 2)
        optimizer_a = torch.optim.AdamW(model_a.parameters(), lr=0.01)
        load_checkpoint(
            checkpoint,
            model=model_a,
            optimizer=optimizer_a,
            config_sha256=config_sha,
            data_sha256=data_sha,
        )
        actual_rng = torch.rand(4)
        assert torch.equal(actual_rng, expected_rng)
        for key, value in model_a.state_dict().items():
            assert torch.equal(value, expected_model[key])

        model_b = torch.nn.Linear(4, 2)
        optimizer_b = torch.optim.AdamW(model_b.parameters(), lr=0.01)
        load_checkpoint(
            checkpoint,
            model=model_b,
            optimizer=optimizer_b,
            config_sha256=config_sha,
            data_sha256=data_sha,
        )
        assert all(torch.equal(a, b) for a, b in zip(model_a.parameters(), model_b.parameters()))

        try:
            load_checkpoint(
                checkpoint,
                model=model_b,
                optimizer=optimizer_b,
                config_sha256=_digest("wrong-config"),
                data_sha256=data_sha,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("checkpoint hash mismatch must fail closed")
    print("torch self-check passed: immutable matched branch restore")


if __name__ == "__main__":
    main()
