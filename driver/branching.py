"""Small Dream-style branch seam for the landscape-driver harness.

This module deliberately stops short of a neural world model.  It provides
the part that must be correct before one is useful: matched sibling
provenance, explicit ``real``/``replay``/``imagined`` modes, and a tiny
state-conditioned empirical model that can learn online while driving.  A
future numerical encoder can implement the same ``predict``/``update``
boundary without changing the branch contract.

Imagined predictions are never evidence.  The allocator refuses to emit an
imagined branch until recent real-branch residuals meet a conservative
calibration gate; replay is allowed only for a transition already recorded in
the real archive.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from .core import Action, ActionEstimate, Observation, Transition


BranchMode = Literal["real", "replay", "imagined"]
BRANCH_MODES = frozenset({"real", "replay", "imagined"})


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


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


@dataclass(frozen=True)
class BranchSpec:
    """Provenance for one sibling branch from an immutable checkpoint."""

    group_id: str
    mode: BranchMode
    parent_checkpoint_sha256: str
    config_sha256: str
    data_sha256: str
    code_sha: str
    action: Action
    horizon: int
    source_transition_id: str | None = None
    world_model_id: str | None = None

    def __post_init__(self) -> None:
        _text(self.group_id, "branch group_id")
        if self.mode not in BRANCH_MODES:
            raise ValueError(f"unknown branch mode {self.mode!r}")
        for name in (
            "parent_checkpoint_sha256",
            "config_sha256",
            "data_sha256",
            "code_sha",
        ):
            _text(getattr(self, name), f"branch {name}")
        if not isinstance(self.action, Action):
            raise TypeError("branch action must be an Action")
        if not isinstance(self.horizon, int) or isinstance(self.horizon, bool) or self.horizon < 1:
            raise ValueError("branch horizon must be a positive integer")
        if self.source_transition_id is not None:
            _text(self.source_transition_id, "branch source_transition_id")
        if self.world_model_id is not None:
            _text(self.world_model_id, "branch world_model_id")
        if self.mode == "replay" and self.source_transition_id is None:
            raise ValueError("replay branches require source_transition_id")
        if self.mode == "imagined" and self.world_model_id is None:
            raise ValueError("imagined branches require world_model_id")
        if self.mode == "real" and self.source_transition_id is not None:
            raise ValueError("real branches cannot name a replay source")

    @property
    def sibling_key(self) -> tuple[object, ...]:
        """Fields that must match for causal sibling comparisons."""

        return (
            self.group_id,
            self.parent_checkpoint_sha256,
            self.config_sha256,
            self.data_sha256,
            self.code_sha,
            self.horizon,
        )

    @property
    def action_key(self) -> tuple[object, ...]:
        return (
            self.action.kind,
            self.action.strength,
            tuple(sorted(self.action.parameters.items())),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "mode": self.mode,
            "parent_checkpoint_sha256": self.parent_checkpoint_sha256,
            "config_sha256": self.config_sha256,
            "data_sha256": self.data_sha256,
            "code_sha": self.code_sha,
            "action": self.action.to_dict(),
            "horizon": self.horizon,
            "source_transition_id": self.source_transition_id,
            "world_model_id": self.world_model_id,
        }


class BranchLedger:
    """Validate sibling provenance and keep replay sources explicit."""

    def __init__(self) -> None:
        self._groups: dict[str, list[BranchSpec]] = {}
        self._real_specs: dict[str, BranchSpec] = {}

    def register(self, spec: BranchSpec) -> None:
        siblings = self._groups.get(spec.group_id, [])
        if any(existing.action_key == spec.action_key and existing.mode == spec.mode for existing in siblings):
            raise ValueError("duplicate action/mode in one branch group")
        if siblings and any(existing.sibling_key != spec.sibling_key for existing in siblings):
            raise ValueError("branch siblings do not share an identical parent/config/data/horizon")
        if spec.mode == "replay":
            source = self._real_specs.get(spec.source_transition_id or "")
            if source is None:
                raise ValueError("replay source must be a previously recorded real transition")
            if source.sibling_key != spec.sibling_key:
                raise ValueError("replay source does not match the requested branch group")
        self._groups.setdefault(spec.group_id, []).append(spec)

    def record_real(self, spec: BranchSpec, transition_id: str) -> None:
        if spec.mode != "real":
            raise ValueError("record_real accepts only real branch specs")
        transition_id = _text(transition_id, "transition_id")
        if transition_id in self._real_specs:
            raise ValueError("duplicate real transition id")
        self.register(spec)
        self._real_specs[transition_id] = spec

    def group(self, group_id: str) -> tuple[BranchSpec, ...]:
        return tuple(self._groups.get(_text(group_id, "group_id"), ()))

    def real_transition_ids(self) -> frozenset[str]:
        return frozenset(self._real_specs)


@dataclass(frozen=True)
class BranchPrediction:
    """Action-conditioned outcome prediction used for scoring only."""

    gain: float
    cost_flops: float
    risk: float
    uncertainty: float
    learning_progress: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "gain", _finite(self.gain, "prediction gain"))
        object.__setattr__(self, "cost_flops", _non_negative(self.cost_flops, "prediction cost_flops"))
        object.__setattr__(self, "risk", _non_negative(self.risk, "prediction risk"))
        object.__setattr__(self, "uncertainty", _non_negative(self.uncertainty, "prediction uncertainty"))
        object.__setattr__(
            self,
            "learning_progress",
            _finite(self.learning_progress, "prediction learning_progress"),
        )


@dataclass
class _Stats:
    count: int = 0
    gain_mean: float = 0.0
    gain_error: float = 0.0
    cost_mean: float = 0.0
    risk_mean: float = 0.0
    progress_mean: float = 0.0

    def update(self, gain: float, cost: float, risk: float, progress: float) -> None:
        self.count += 1
        delta = gain - self.gain_mean
        self.gain_mean += delta / self.count
        self.gain_error += delta * (gain - self.gain_mean)
        for name, value in (
            ("cost_mean", cost),
            ("risk_mean", risk),
            ("progress_mean", progress),
        ):
            old = getattr(self, name)
            setattr(self, name, old + (value - old) / self.count)

    def uncertainty(self, prior: float) -> float:
        if self.count < 2:
            return prior / math.sqrt(self.count + 1)
        deviation = math.sqrt(max(0.0, self.gain_error / (self.count - 1)))
        return max(deviation, 1.0 / math.sqrt(self.count))


class FeatureActionModel:
    """Tiny state-conditioned empirical model for the first online driver.

    It buckets a declared telemetry vector and keeps Welford statistics per
    ``(bucket, action kind)``.  This is intentionally weaker than the proposed
    deep driver, but it tests whether state information is useful at all and
    supplies a cheap control before introducing a learned representation.
    """

    def __init__(
        self,
        *,
        feature_names: Sequence[str] = (),
        bin_width: float = 0.25,
        prior_gain: float = 0.0,
        prior_cost_flops: float = 0.0,
        prior_uncertainty: float = 1.0,
    ) -> None:
        if bin_width <= 0 or not math.isfinite(float(bin_width)):
            raise ValueError("bin_width must be positive and finite")
        self.feature_names = tuple(_text(name, "feature name") for name in feature_names)
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must not contain duplicates")
        self.bin_width = float(bin_width)
        self.prior_gain = _finite(prior_gain, "prior_gain")
        self.prior_cost_flops = _non_negative(prior_cost_flops, "prior_cost_flops")
        self.prior_uncertainty = _non_negative(prior_uncertainty, "prior_uncertainty")
        self._global: dict[str, _Stats] = {}
        self._local: dict[tuple[str, tuple[int, ...]], _Stats] = {}

    def _bucket(self, observation: Observation) -> tuple[int, ...]:
        values = [observation.loss, observation.quality, observation.risk]
        values.extend(observation.features.get(name, 0.0) for name in self.feature_names)
        return tuple(int(round(value / self.bin_width)) for value in values)

    def _prediction(self, stats: _Stats | None) -> BranchPrediction:
        if stats is None:
            return BranchPrediction(
                gain=self.prior_gain,
                cost_flops=self.prior_cost_flops,
                risk=0.0,
                uncertainty=self.prior_uncertainty,
            )
        return BranchPrediction(
            gain=stats.gain_mean,
            cost_flops=max(0.0, stats.cost_mean),
            risk=min(1.0, max(0.0, stats.risk_mean)),
            uncertainty=stats.uncertainty(self.prior_uncertainty),
            learning_progress=stats.progress_mean,
        )

    def predict(self, observation: Observation, action: Action) -> BranchPrediction:
        local = self._local.get((action.kind, self._bucket(observation)))
        if local is not None:
            return self._prediction(local)
        return self._prediction(self._global.get(action.kind))

    def estimates(
        self, observation: Observation, actions: Sequence[Action]
    ) -> tuple[ActionEstimate, ...]:
        return tuple(
            ActionEstimate(
                action=action,
                predicted_gain=(prediction.gain + prediction.learning_progress),
                uncertainty=prediction.uncertainty,
                risk=prediction.risk,
                cost_flops=prediction.cost_flops,
            )
            for action in actions
            for prediction in (self.predict(observation, action),)
        )

    def update(self, transition: Transition) -> None:
        observation = transition.before
        risk = 1.0 if transition.failure else (transition.after.risk if transition.after else observation.risk)
        values = (
            transition.reward_task,
            transition.compute_flops,
            risk,
            transition.learning_progress,
        )
        self._global.setdefault(transition.action.kind, _Stats()).update(*values)
        self._local.setdefault(
            (transition.action.kind, self._bucket(observation)), _Stats()
        ).update(*values)


class DreamAllocator:
    """Online controller plus a fail-closed real/replay/imagined allocator."""

    def __init__(
        self,
        model: FeatureActionModel,
        ledger: BranchLedger | None = None,
        *,
        curiosity_weight: float = 0.1,
        learning_progress_weight: float = 0.25,
        risk_weight: float = 1.0,
        calibration_limit: float = 0.25,
        calibration_samples: int = 4,
    ) -> None:
        self.model = model
        self.ledger = ledger or BranchLedger()
        self.curiosity_weight = _finite(curiosity_weight, "curiosity_weight")
        self.learning_progress_weight = _finite(
            learning_progress_weight, "learning_progress_weight"
        )
        self.risk_weight = _finite(risk_weight, "risk_weight")
        self.calibration_limit = _non_negative(calibration_limit, "calibration_limit")
        if not isinstance(calibration_samples, int) or calibration_samples < 1:
            raise ValueError("calibration_samples must be positive")
        self.calibration_samples = calibration_samples
        self._calibration_count: dict[str, int] = {}
        self._calibration_error: dict[str, float] = {}

    def estimates(
        self, observation: Observation, actions: Sequence[Action]
    ) -> tuple[ActionEstimate, ...]:
        return self.model.estimates(observation, actions)

    def update(self, transition: Transition) -> None:
        predicted = self.model.predict(transition.before, transition.action)
        error = abs(predicted.gain - transition.reward_task)
        kind = transition.action.kind
        count = self._calibration_count.get(kind, 0) + 1
        previous = self._calibration_error.get(kind, 0.0)
        self._calibration_count[kind] = count
        self._calibration_error[kind] = previous + (error - previous) / count
        self.model.update(transition)

    def calibrated(self, action_kind: str | None = None) -> bool:
        kinds = (
            (action_kind,)
            if action_kind is not None
            else tuple(self._calibration_count)
        )
        return bool(kinds) and all(
            self._calibration_count.get(kind, 0) >= self.calibration_samples
            and self._calibration_error.get(kind, math.inf) <= self.calibration_limit
            for kind in kinds
        )

    def propose(
        self,
        observation: Observation,
        actions: Sequence[Action],
        *,
        group_id: str,
        parent_checkpoint_sha256: str,
        config_sha256: str,
        data_sha256: str,
        code_sha: str,
        horizon: int,
        mode: BranchMode = "real",
        source_transition_id: str | None = None,
        world_model_id: str | None = None,
    ) -> tuple[BranchSpec, BranchPrediction]:
        actions = tuple(actions)
        if not actions:
            raise ValueError("cannot propose from an empty action set")
        predictions = [(action, self.model.predict(observation, action)) for action in actions]
        maximum_cost = max(prediction.cost_flops for _, prediction in predictions)

        def score(item: tuple[Action, BranchPrediction]) -> float:
            _, prediction = item
            relative_cost = prediction.cost_flops / maximum_cost if maximum_cost else 0.0
            return (
                prediction.gain
                + self.learning_progress_weight * prediction.learning_progress
                + self.curiosity_weight * prediction.uncertainty
                - self.risk_weight * prediction.risk
                - 0.1 * relative_cost
            )

        action, prediction = max(predictions, key=score)
        if mode == "imagined" and not self.calibrated(action.kind):
            raise ValueError(
                "imagined planning is disabled for an action until its real predictions calibrate"
            )
        spec = BranchSpec(
            group_id=group_id,
            mode=mode,
            parent_checkpoint_sha256=parent_checkpoint_sha256,
            config_sha256=config_sha256,
            data_sha256=data_sha256,
            code_sha=code_sha,
            action=action,
            horizon=horizon,
            source_transition_id=source_transition_id,
            world_model_id=world_model_id,
        )
        return spec, prediction

    def calibration(self) -> Mapping[str, Mapping[str, float]]:
        return {
            kind: {
                "samples": float(self._calibration_count[kind]),
                "mean_absolute_gain_error": self._calibration_error[kind],
            }
            for kind in sorted(self._calibration_count)
        }
