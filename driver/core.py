"""Small, dependency-free contracts for landscape-driver experiments.

The target-training implementation can later provide these contracts from
PyTorch without changing the archive or selection protocol.
"""

from __future__ import annotations

import json
import math
import os
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _non_negative(value: float, name: str) -> float:
    value = _finite(value, name)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _metadata(value: Mapping[str, Any], name: str = "metadata") -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    result = dict(value)
    try:
        json.dumps(result)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be JSON-serializable") from exc
    return result


@dataclass(frozen=True)
class Action:
    """One bounded intervention proposed for a training state."""

    kind: str
    strength: float = 0.0
    parameters: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValueError("action kind must be a non-empty string")
        object.__setattr__(self, "strength", _finite(self.strength, "action strength"))
        parameters = dict(self.parameters)
        for key, value in parameters.items():
            if not isinstance(key, str) or not key:
                raise ValueError("action parameter names must be non-empty strings")
            parameters[key] = _finite(value, f"action parameter {key}")
        object.__setattr__(self, "parameters", parameters)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "strength": self.strength,
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Action":
        if not isinstance(value, Mapping):
            raise ValueError("action must be an object")
        return cls(
            kind=value["kind"],
            strength=value.get("strength", 0.0),
            parameters=value.get("parameters", {}),
        )


@dataclass(frozen=True)
class Observation:
    """Telemetry available at one decision point."""

    step: int
    tokens: int
    loss: float
    quality: float = 0.0
    compute_flops: float = 0.0
    risk: float = 0.0
    features: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.step, int) or isinstance(self.step, bool) or self.step < 0:
            raise ValueError("observation step must be a non-negative integer")
        if not isinstance(self.tokens, int) or isinstance(self.tokens, bool) or self.tokens < 0:
            raise ValueError("observation tokens must be a non-negative integer")
        for name in ("loss", "quality", "compute_flops"):
            object.__setattr__(
                self, name, _finite(getattr(self, name), f"observation {name}")
            )
        object.__setattr__(self, "risk", _non_negative(self.risk, "observation risk"))
        features = dict(self.features)
        for key, value in features.items():
            if not isinstance(key, str) or not key:
                raise ValueError("feature names must be non-empty strings")
            features[key] = _finite(value, f"feature {key}")
        object.__setattr__(self, "features", features)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "tokens": self.tokens,
            "loss": self.loss,
            "quality": self.quality,
            "compute_flops": self.compute_flops,
            "risk": self.risk,
            "features": dict(self.features),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Observation":
        if not isinstance(value, Mapping):
            raise ValueError("observation must be an object")
        return cls(
            step=value["step"],
            tokens=value["tokens"],
            loss=value["loss"],
            quality=value.get("quality", 0.0),
            compute_flops=value.get("compute_flops", 0.0),
            risk=value.get("risk", 0.0),
            features=value.get("features", {}),
        )


@dataclass(frozen=True)
class Outcome:
    """Measured result returned by the target-training loop."""

    after: Observation | None
    compute_flops: float
    wall_seconds: float
    reward_task: float
    learning_progress: float = 0.0
    accepted: bool = True
    done: bool = False
    failure: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "compute_flops", _non_negative(self.compute_flops, "outcome compute_flops")
        )
        object.__setattr__(
            self, "wall_seconds", _non_negative(self.wall_seconds, "outcome wall_seconds")
        )
        object.__setattr__(self, "reward_task", _finite(self.reward_task, "outcome reward_task"))
        object.__setattr__(
            self,
            "learning_progress",
            _finite(self.learning_progress, "outcome learning_progress"),
        )
        if not isinstance(self.accepted, bool) or not isinstance(self.done, bool):
            raise ValueError("outcome accepted and done must be booleans")
        if self.failure is not None and not isinstance(self.failure, str):
            raise ValueError("outcome failure must be a string or null")
        object.__setattr__(self, "metadata", _metadata(self.metadata, "outcome metadata"))


@dataclass(frozen=True)
class Transition:
    """One causally ordered action/result record in the experiment archive."""

    transition_id: str
    run_id: str
    parent_id: str | None
    before: Observation
    action: Action
    after: Observation | None
    compute_flops: float
    wall_seconds: float
    reward_task: float
    learning_progress: float
    accepted: bool = True
    failure: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("transition_id", "run_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.parent_id is not None and (
            not isinstance(self.parent_id, str) or not self.parent_id.strip()
        ):
            raise ValueError("parent_id must be a non-empty string or null")
        if not isinstance(self.before, Observation):
            raise TypeError("before must be an Observation")
        if not isinstance(self.action, Action):
            raise TypeError("action must be an Action")
        if self.after is not None and not isinstance(self.after, Observation):
            raise TypeError("after must be an Observation or null")
        object.__setattr__(
            self,
            "compute_flops",
            _non_negative(self.compute_flops, "transition compute_flops"),
        )
        object.__setattr__(
            self,
            "wall_seconds",
            _non_negative(self.wall_seconds, "transition wall_seconds"),
        )
        object.__setattr__(self, "reward_task", _finite(self.reward_task, "transition reward_task"))
        object.__setattr__(
            self,
            "learning_progress",
            _finite(self.learning_progress, "transition learning_progress"),
        )
        if not isinstance(self.accepted, bool):
            raise ValueError("transition accepted must be a boolean")
        if self.failure is not None and not isinstance(self.failure, str):
            raise ValueError("transition failure must be a string or null")
        object.__setattr__(self, "metadata", _metadata(self.metadata, "transition metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "run_id": self.run_id,
            "parent_id": self.parent_id,
            "before": self.before.to_dict(),
            "action": self.action.to_dict(),
            "after": None if self.after is None else self.after.to_dict(),
            "compute_flops": self.compute_flops,
            "wall_seconds": self.wall_seconds,
            "reward_task": self.reward_task,
            "learning_progress": self.learning_progress,
            "accepted": self.accepted,
            "failure": self.failure,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Transition":
        if not isinstance(value, Mapping):
            raise ValueError("transition must be an object")
        after = value.get("after")
        return cls(
            transition_id=value["transition_id"],
            run_id=value["run_id"],
            parent_id=value.get("parent_id"),
            before=Observation.from_dict(value["before"]),
            action=Action.from_dict(value["action"]),
            after=None if after is None else Observation.from_dict(after),
            compute_flops=value["compute_flops"],
            wall_seconds=value["wall_seconds"],
            reward_task=value["reward_task"],
            learning_progress=value.get("learning_progress", 0.0),
            accepted=value.get("accepted", True),
            failure=value.get("failure"),
            metadata=value.get("metadata", {}),
        )


class Archive:
    """Append-only JSONL archive with explicit end-of-campaign validation."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)

    def append(self, transition: Transition) -> None:
        if not isinstance(transition, Transition):
            raise TypeError("archive accepts Transition values")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(transition.to_dict(), sort_keys=True, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def records(self):
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    yield Transition.from_dict(payload)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"invalid archive record at line {line_number}") from exc

    def validate(self) -> int:
        """Check unique IDs and causal parent order; return record count.

        Validation is a deliberate end-of-campaign pass so appends stay O(1).
        """

        seen: dict[str, str] = {}
        count = 0
        for transition in self.records():
            if transition.transition_id in seen:
                raise ValueError(f"duplicate transition id: {transition.transition_id}")
            if transition.parent_id is not None:
                if transition.parent_id not in seen:
                    raise ValueError(
                        f"parent {transition.parent_id} is missing or appears later"
                    )
                if seen[transition.parent_id] != transition.run_id:
                    raise ValueError("parent and child must belong to the same run")
            seen[transition.transition_id] = transition.run_id
            count += 1
        return count


@dataclass(frozen=True)
class ActionEstimate:
    action: Action
    predicted_gain: float
    uncertainty: float
    risk: float = 0.0
    cost_flops: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "predicted_gain", _finite(self.predicted_gain, "predicted_gain"))
        object.__setattr__(self, "uncertainty", _non_negative(self.uncertainty, "uncertainty"))
        object.__setattr__(self, "risk", _non_negative(self.risk, "risk"))
        object.__setattr__(self, "cost_flops", _non_negative(self.cost_flops, "cost_flops"))


class Selector:
    """Small deterministic selector for the initial policy comparisons."""

    STRATEGIES = frozenset({"fixed", "random", "task", "curiosity", "balanced"})

    def __init__(
        self,
        strategy: str,
        *,
        curiosity_weight: float = 0.25,
        cost_weight: float = 0.1,
        risk_weight: float = 1.0,
        seed: int = 0,
    ):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}; use {sorted(self.STRATEGIES)}")
        self.strategy = strategy
        self.curiosity_weight = _finite(curiosity_weight, "curiosity_weight")
        self.cost_weight = _finite(cost_weight, "cost_weight")
        self.risk_weight = _finite(risk_weight, "risk_weight")
        self._random = random.Random(seed)

    def choose(self, estimates: Sequence[ActionEstimate]) -> ActionEstimate:
        estimates = tuple(estimates)
        if not estimates:
            raise ValueError("cannot select from an empty action set")
        if any(not isinstance(item, ActionEstimate) for item in estimates):
            raise TypeError("selector accepts ActionEstimate values")
        if self.strategy == "fixed":
            return estimates[0]
        if self.strategy == "random":
            return self._random.choice(estimates)

        maximum_cost = max(item.cost_flops for item in estimates)

        def score(item: ActionEstimate) -> float:
            relative_cost = item.cost_flops / maximum_cost if maximum_cost else 0.0
            penalty = self.cost_weight * relative_cost + self.risk_weight * item.risk
            if self.strategy == "task":
                return item.predicted_gain - penalty
            if self.strategy == "curiosity":
                return item.uncertainty - penalty
            return item.predicted_gain + self.curiosity_weight * item.uncertainty - penalty

        return max(estimates, key=score)


@dataclass
class _RunningStats:
    count: int = 0
    mean: float = 0.0
    squared_error: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.squared_error += delta * (value - self.mean)

    def uncertainty(self, prior: float) -> float:
        if self.count < 2:
            return prior / math.sqrt(self.count + 1)
        deviation = math.sqrt(max(0.0, self.squared_error / (self.count - 1)))
        return max(deviation, 1.0 / math.sqrt(self.count))


class OnlineActionStats:
    """Cheap history-conditioned baseline with per-run online adaptation.

    This is intentionally not the proposed deep driver. It is the control that
    must be beaten before adding a learned numerical model.
    """

    def __init__(
        self,
        *,
        prior_gain: float = 0.0,
        prior_uncertainty: float = 1.0,
        action_costs: Mapping[str, float] | None = None,
    ):
        self.prior_gain = _finite(prior_gain, "prior_gain")
        self.prior_uncertainty = _non_negative(prior_uncertainty, "prior_uncertainty")
        self.action_costs = {
            kind: _non_negative(cost, f"cost for {kind}")
            for kind, cost in (action_costs or {}).items()
        }
        self._stats: dict[str, _RunningStats] = {}

    def reset(self) -> None:
        self._stats.clear()

    def estimates(
        self, observation: Observation, actions: Sequence[Action]
    ) -> tuple[ActionEstimate, ...]:
        del observation
        result = []
        for action in actions:
            stats = self._stats.get(action.kind)
            result.append(
                ActionEstimate(
                    action=action,
                    predicted_gain=self.prior_gain if stats is None else stats.mean,
                    uncertainty=(
                        self.prior_uncertainty
                        if stats is None
                        else stats.uncertainty(self.prior_uncertainty)
                    ),
                    cost_flops=self.action_costs.get(action.kind, 0.0),
                )
            )
        return tuple(result)

    def update(self, transition: Transition) -> None:
        self._stats.setdefault(transition.action.kind, _RunningStats()).update(
            transition.reward_task
        )


class Target(Protocol):
    def observe(self) -> Observation: ...

    def actions(self, observation: Observation) -> Sequence[Action]: ...

    def execute(self, action: Action) -> Outcome: ...


class Controller(Protocol):
    def estimates(
        self, observation: Observation, actions: Sequence[Action]
    ) -> Sequence[ActionEstimate]: ...

    def update(self, transition: Transition) -> None: ...


def run_episode(
    target: Target,
    controller: Controller,
    selector: Selector,
    archive: Archive,
    *,
    run_id: str,
    max_decisions: int,
) -> list[Transition]:
    """Run one causal target episode and persist every intervention."""

    if not run_id.strip():
        raise ValueError("run_id must be non-empty")
    if not isinstance(max_decisions, int) or max_decisions < 1:
        raise ValueError("max_decisions must be a positive integer")

    transitions: list[Transition] = []
    parent_id: str | None = None
    for decision in range(max_decisions):
        before = target.observe()
        actions = tuple(target.actions(before))
        if not actions:
            raise ValueError("target returned no actions")
        estimates = tuple(controller.estimates(before, actions))
        if tuple(item.action for item in estimates) != actions:
            raise ValueError("controller must return one estimate per action, in order")
        chosen = selector.choose(estimates)
        outcome = target.execute(chosen.action)
        transition = Transition(
            transition_id=f"{run_id}:{decision}",
            run_id=run_id,
            parent_id=parent_id,
            before=before,
            action=chosen.action,
            after=outcome.after,
            compute_flops=outcome.compute_flops,
            wall_seconds=outcome.wall_seconds,
            reward_task=outcome.reward_task,
            learning_progress=outcome.learning_progress,
            accepted=outcome.accepted,
            failure=outcome.failure,
            metadata=outcome.metadata,
        )
        archive.append(transition)
        controller.update(transition)
        transitions.append(transition)
        parent_id = transition.transition_id
        if outcome.done:
            break
    return transitions
