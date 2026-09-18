"""Small matched-branch decoder experiment.

This is the first target-training loop, not a claim that the pulse is useful.
It intentionally keeps the action space to ``noop`` and ``role_pulse`` so a
negative result is interpretable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .checkpoints import Checkpoint, load_checkpoint, save_checkpoint
from .core import Action, Archive, Observation, Transition


@dataclass(frozen=True)
class Config:
    landscape: str
    seed: int
    vocab_size: int = 128
    context: int = 128
    batch_size: int = 64
    width: int = 256
    layers: int = 4
    heads: int = 4
    prefix_steps: int = 20
    immediate_steps: int = 4
    recovery_steps: int = 16
    final_steps: int = 40
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    pulse_attention: float = 1.05
    pulse_mlp: float = 0.95
    pulse_embedding: float = 1.0
    pulse_norm: float = 1.0
    pulse_head: float = 1.0


def _code_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_tensor(value: torch.Tensor) -> str:
    cpu_value = value.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    return _hash_bytes(bytes(cpu_value.tolist()))


def _hash_config(config: Config) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return _hash_bytes(payload.encode())


def make_stream(
    landscape: str, *, seed: int, length: int, vocab_size: int
) -> torch.Tensor:
    """Generate deterministic byte-like training landscapes without downloads."""

    if landscape not in {"delayed_copy", "phase_switch", "text_shard"}:
        raise ValueError(f"unknown landscape: {landscape}")
    generator = torch.Generator().manual_seed(seed)
    if landscape == "text_shard":
        text = (
            "the driver observes a training landscape and chooses a bounded "
            "intervention. measure recovery, cost, and transfer. "
        ).encode()
        offset = seed % len(text)
        rotated = text[offset:] + text[:offset]
        values = torch.tensor([byte % vocab_size for byte in rotated], dtype=torch.uint8)
        return values.repeat((length + len(values) - 1) // len(values))[:length]
    if landscape == "delayed_copy":
        values = torch.randint(0, vocab_size, (length,), generator=generator, dtype=torch.uint8)
        lag = 16
        mask = torch.rand(length - lag, generator=generator) < 0.8
        values[lag:] = torch.where(mask, values[:-lag], values[lag:])
        return values

    values = torch.empty(length, dtype=torch.uint8)
    state = int(torch.randint(0, vocab_size, (), generator=generator))
    switch = length // 2
    first = torch.randperm(vocab_size, generator=generator)
    second = torch.randperm(vocab_size, generator=generator)
    for index in range(length):
        mapping = first if index < switch else second
        state = int(mapping[state])
        values[index] = state
    return values


def make_byte_stream(path: str | Path, *, seed: int, length: int) -> torch.Tensor:
    """Load a deterministic cyclic byte stream without a tokenizer dependency."""

    if length <= 0:
        raise ValueError("byte stream length must be positive")
    raw = Path(path).read_bytes()
    if not raw:
        raise ValueError(f"byte stream is empty: {path}")
    values = torch.tensor(list(raw), dtype=torch.uint8)
    offset = seed % values.numel()
    if offset:
        values = torch.cat((values[offset:], values[:offset]))
    repeats = (length + values.numel() - 1) // values.numel()
    return values.repeat(repeats)[:length]


class TokenStream:
    def __init__(self, values: torch.Tensor, *, cursor: int = 0):
        if values.ndim != 1 or values.dtype != torch.uint8:
            raise ValueError("token stream must be a one-dimensional uint8 tensor")
        if cursor < 0 or cursor >= values.numel():
            raise ValueError("stream cursor is outside the token stream")
        self.values = values
        self.cursor = cursor

    def batch(self, *, batch_size: int, context: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        width = batch_size * (context + 1)
        end = self.cursor + width
        if end > self.values.numel():
            raise RuntimeError("training stream exhausted; increase generated length")
        chunk = self.values[self.cursor:end].view(batch_size, context + 1)
        self.cursor = end
        chunk = chunk.to(device=device, dtype=torch.long, non_blocking=True)
        return chunk[:, :-1], chunk[:, 1:]

    def state(self) -> dict[str, int]:
        return {"cursor": self.cursor}


class Block(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.ln_attention = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.attention_out = nn.Linear(width, width)
        self.ln_mlp = nn.LayerNorm(width)
        self.mlp_in = nn.Linear(width, 4 * width)
        self.mlp_out = nn.Linear(4 * width, width)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        normalized = self.ln_attention(values)
        batch, context, _ = normalized.shape
        query, key, value = self.qkv(normalized).chunk(3, dim=-1)
        query = query.view(batch, context, self.heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, context, self.heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, context, self.heads, self.head_dim).transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        attended = attended.transpose(1, 2).contiguous().view(batch, context, self.width)
        values = values + self.attention_out(attended)
        values = values + self.mlp_out(F.gelu(self.mlp_in(self.ln_mlp(values))))
        return values


class DecoderLM(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        self.position_embedding = nn.Parameter(torch.zeros(1, config.context, config.width))
        self.blocks = nn.ModuleList(
            [Block(config.width, config.heads) for _ in range(config.layers)]
        )
        self.final_norm = nn.LayerNorm(config.width)
        self.lm_head = nn.Linear(config.width, config.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        values = self.token_embedding(tokens) + self.position_embedding[:, : tokens.shape[1]]
        for block in self.blocks:
            values = block(values)
        logits = self.lm_head(self.final_norm(values))
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def _role(name: str) -> str:
    if name.startswith("token_embedding") or name.startswith("position_embedding"):
        return "embedding"
    if ".qkv" in name or ".attention_out" in name:
        return "attention"
    if ".mlp_in" in name or ".mlp_out" in name:
        return "mlp"
    if "norm" in name or ".ln_" in name:
        return "norm"
    return "head"


def _pulse_multiplier(config: Config, role: str) -> float:
    return {
        "attention": config.pulse_attention,
        "mlp": config.pulse_mlp,
        "embedding": config.pulse_embedding,
        "norm": config.pulse_norm,
        "head": config.pulse_head,
    }[role]


def train_steps(
    model: DecoderLM,
    optimizer: torch.optim.Optimizer,
    stream: TokenStream,
    config: Config,
    device: torch.device,
    steps: int,
    *,
    pulse: bool,
) -> float:
    if steps < 0:
        raise ValueError("steps must be non-negative")
    started = time.perf_counter()
    for step in range(steps):
        tokens, targets = stream.batch(
            batch_size=config.batch_size, context=config.context, device=device
        )
        optimizer.zero_grad(set_to_none=True)
        loss = model(tokens, targets)
        loss.backward()
        snapshots: dict[str, torch.Tensor] = {}
        if pulse and step < config.immediate_steps:
            snapshots = {
                name: parameter.detach().clone()
                for name, parameter in model.named_parameters()
            }
        optimizer.step()
        if snapshots:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    before = snapshots[name]
                    multiplier = _pulse_multiplier(config, _role(name))
                    parameter.add_((multiplier - 1.0) * (parameter - before))
    _sync(device)
    return time.perf_counter() - started


@torch.no_grad()
def evaluate(
    model: DecoderLM,
    values: torch.Tensor,
    *,
    config: Config,
    device: torch.device,
) -> float:
    chunk = values[: config.batch_size * (config.context + 1)].view(
        config.batch_size, config.context + 1
    )
    tokens = chunk[:, :-1].to(device=device, dtype=torch.long)
    targets = chunk[:, 1:].to(device=device, dtype=torch.long)
    return float(model(tokens, targets).item())


def _model_and_optimizer(config: Config, device: torch.device) -> tuple[DecoderLM, torch.optim.Optimizer]:
    model = DecoderLM(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    return model, optimizer


def _flops(config: Config, model: DecoderLM, tokens: int, pulse_steps: int = 0) -> float:
    parameters = sum(parameter.numel() for parameter in model.parameters())
    target = 6.0 * parameters * tokens
    pulse_overhead = 2.0 * parameters * pulse_steps
    return target + pulse_overhead


def _observation(step: int, tokens: int, loss: float, flops: float) -> Observation:
    return Observation(step=step, tokens=tokens, loss=loss, quality=-loss, compute_flops=flops)


def _branch(
    *,
    action: str,
    config: Config,
    parent: Checkpoint,
    train_values: torch.Tensor,
    validation_values: torch.Tensor,
    config_sha: str,
    data_sha: str,
    code_sha: str,
    before: Observation,
    output: Path,
    device: torch.device,
) -> tuple[Transition, dict[str, Any]]:
    model, optimizer = _model_and_optimizer(config, device)
    stream = TokenStream(train_values, cursor=0)
    started = time.perf_counter()
    metadata = load_checkpoint(
        parent,
        model=model,
        optimizer=optimizer,
        config_sha256=config_sha,
        data_sha256=data_sha,
    )
    stream = TokenStream(train_values, cursor=int(metadata["data_state"]["cursor"]))
    pulse = action == "role_pulse"
    immediate_steps = config.immediate_steps
    recovery_steps = config.recovery_steps - immediate_steps
    final_steps = config.final_steps - config.recovery_steps
    train_seconds = train_steps(
        model, optimizer, stream, config, device, immediate_steps, pulse=pulse
    )
    immediate_loss = evaluate(model, validation_values, config=config, device=device)
    train_seconds += train_steps(
        model, optimizer, stream, config, device, recovery_steps, pulse=False
    )
    recovery_loss = evaluate(model, validation_values, config=config, device=device)
    train_seconds += train_steps(
        model, optimizer, stream, config, device, final_steps, pulse=False
    )
    final_loss = evaluate(model, validation_values, config=config, device=device)
    _sync(device)
    wall_seconds = time.perf_counter() - started
    target_tokens = config.final_steps * config.batch_size * config.context
    flops = _flops(config, model, target_tokens, config.immediate_steps if pulse else 0)
    transition = Transition(
        transition_id=f"{config.landscape}-{config.seed}:{action}",
        run_id=f"{config.landscape}-{config.seed}",
        parent_id=None,
        before=before,
        action=Action(action, strength=1.0 if pulse else 0.0),
        after=_observation(
            config.final_steps,
            before.tokens + target_tokens,
            final_loss,
            flops,
        ),
        compute_flops=flops,
        wall_seconds=wall_seconds,
        reward_task=before.loss - final_loss,
        learning_progress=before.loss - final_loss,
        accepted=True,
        metadata={
            "landscape": config.landscape,
            "seed": config.seed,
            "source_checkpoint_sha256": parent.sha256,
            "config_sha256": config_sha,
            "data_sha256": data_sha,
            "code_sha": code_sha,
            "target_tokens": target_tokens,
            "train_seconds": train_seconds,
            "checkpoint_seconds": wall_seconds - train_seconds,
            "horizon_metrics": {
                "immediate": immediate_loss,
                "recovery": recovery_loss,
                "final": final_loss,
            },
            "pulse_steps": config.immediate_steps if pulse else 0,
        },
    )
    return transition, {
        "action": action,
        "immediate": immediate_loss,
        "recovery": recovery_loss,
        "final": final_loss,
        "wall_seconds": wall_seconds,
        "estimated_flops": flops,
        "target_tokens": target_tokens,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.final_steps <= args.recovery_steps or args.recovery_steps < args.immediate_steps:
        raise ValueError("horizons must satisfy immediate <= recovery < final")
    config = Config(
        landscape=args.landscape,
        seed=args.seed,
        context=args.context,
        batch_size=args.batch_size,
        width=args.width,
        layers=args.layers,
        heads=args.heads,
        prefix_steps=args.prefix_steps,
        immediate_steps=args.immediate_steps,
        recovery_steps=args.recovery_steps,
        final_steps=args.final_steps,
        learning_rate=args.learning_rate,
    )
    if config.width % config.heads:
        raise ValueError("width must be divisible by heads")
    device = torch.device(args.device)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    train_tokens = (config.prefix_steps + config.final_steps + 4) * config.batch_size * (
        config.context + 1
    )
    validation_tokens = config.batch_size * (config.context + 1)
    train_values = make_stream(
        config.landscape,
        seed=config.seed,
        length=train_tokens,
        vocab_size=config.vocab_size,
    )
    validation_values = make_stream(
        config.landscape,
        seed=config.seed + 1_000_000,
        length=validation_tokens,
        vocab_size=config.vocab_size,
    )
    data_sha = _hash_bytes(_hash_tensor(train_values).encode() + _hash_tensor(validation_values).encode())
    config_sha = _hash_config(config)
    code_sha = _code_sha()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / "transitions.jsonl"
    if archive_path.exists():
        raise FileExistsError(f"choose a fresh --output; {archive_path} already exists")
    checkpoint_path = output / "checkpoints" / "parent.pt"
    model, optimizer = _model_and_optimizer(config, device)
    stream = TokenStream(train_values)
    prefix_seconds = train_steps(
        model, optimizer, stream, config, device, config.prefix_steps, pulse=False
    )
    prefix_loss = evaluate(model, validation_values, config=config, device=device)
    prefix_tokens = stream.cursor
    prefix_flops = _flops(config, model, prefix_tokens)
    parent = save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        data_state=stream.state(),
        config_sha256=config_sha,
        data_sha256=data_sha,
        code_sha=code_sha,
        seed=config.seed,
    )
    before = _observation(config.prefix_steps, prefix_tokens, prefix_loss, prefix_flops)
    archive = Archive(archive_path)
    transitions: list[Transition] = []
    results: list[dict[str, Any]] = []
    for action in ("noop", "role_pulse"):
        transition, result = _branch(
            action=action,
            config=config,
            parent=parent,
            train_values=train_values,
            validation_values=validation_values,
            config_sha=config_sha,
            data_sha=data_sha,
            code_sha=code_sha,
            before=before,
            output=output,
            device=device,
        )
        archive.append(transition)
        transitions.append(transition)
        results.append(result)
    archive.validate()
    summary = {
        "config": asdict(config),
        "device": str(device),
        "torch": torch.__version__,
        "code_sha": code_sha,
        "config_sha256": config_sha,
        "data_sha256": data_sha,
        "parent_checkpoint_sha256": parent.sha256,
        "prefix_loss": prefix_loss,
        "prefix_seconds": prefix_seconds,
        "results": results,
        "pulse_minus_noop_final": results[1]["final"] - results[0]["final"],
    }
    (output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/decoder-smoke")
    parser.add_argument("--landscape", choices=("delayed_copy", "phase_switch", "text_shard"), default="delayed_copy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--prefix-steps", type=int, default=20)
    parser.add_argument("--immediate-steps", type=int, default=4)
    parser.add_argument("--recovery-steps", type=int, default=16)
    parser.add_argument("--final-steps", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
