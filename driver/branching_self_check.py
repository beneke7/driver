"""Deterministic checks for the causal Dream-style branch seam."""

from __future__ import annotations

import tempfile
from pathlib import Path

from .branching import BranchLedger, BranchSpec, DreamAllocator, FeatureActionModel
from .core import Action, Archive, Observation, Outcome, Selector, Transition, run_episode


class FailingTarget:
    def observe(self) -> Observation:
        return Observation(step=0, tokens=0, loss=1.0, quality=0.0)

    def actions(self, observation: Observation) -> tuple[Action, ...]:
        del observation
        return (Action("unstable"),)

    def execute(self, action: Action) -> Outcome:
        raise RuntimeError(f"simulated failure for {action.kind}")


def _transition(action: Action, reward: float, *, slope: float = -0.1) -> Transition:
    observation = Observation(
        step=1,
        tokens=128,
        loss=1.0,
        quality=0.0,
        features={"loss_slope": slope},
    )
    return Transition(
        transition_id=f"real:{action.kind}:{reward}",
        run_id="run",
        parent_id=None,
        before=observation,
        action=action,
        after=Observation(step=2, tokens=256, loss=0.9, quality=0.1),
        compute_flops=1.0,
        wall_seconds=0.01,
        reward_task=reward,
        learning_progress=reward,
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        archive = Archive(Path(directory) / "transitions.jsonl")
        failed = run_episode(
            FailingTarget(),
            FeatureActionModel(),
            Selector("fixed"),
            archive,
            run_id="failure",
            max_decisions=1,
        )[0]
        assert failed.after is None and not failed.accepted
        assert failed.failure and "RuntimeError" in failed.failure
        assert failed.metadata["failure_kind"] == "target_exception"

    pulse = Action("pulse", strength=0.5)
    noop = Action("noop")
    model = FeatureActionModel(feature_names=("loss_slope",), bin_width=0.5)
    ledger = BranchLedger()
    allocator = DreamAllocator(
        model,
        ledger,
        curiosity_weight=0.0,
        calibration_samples=2,
        calibration_limit=0.2,
    )
    for _ in range(2):
        allocator.update(_transition(pulse, 0.1))
    allocator.update(_transition(pulse, -0.2, slope=0.9))
    assert allocator.calibrated("pulse")
    first_state = Observation(
        step=1,
        tokens=128,
        loss=1.0,
        quality=0.0,
        features={"loss_slope": -0.1},
    )
    second_state = Observation(
        step=1,
        tokens=128,
        loss=1.0,
        quality=0.0,
        features={"loss_slope": 0.9},
    )
    assert allocator.model.predict(first_state, pulse).gain > allocator.model.predict(
        second_state, pulse
    ).gain

    real_spec = BranchSpec(
        group_id="g0",
        mode="real",
        parent_checkpoint_sha256="parent",
        config_sha256="config",
        data_sha256="data",
        code_sha="code",
        action=pulse,
        horizon=4,
    )
    ledger.record_real(real_spec, "real:pulse")
    replay_spec = BranchSpec(
        group_id="g0",
        mode="replay",
        parent_checkpoint_sha256="parent",
        config_sha256="config",
        data_sha256="data",
        code_sha="code",
        action=pulse,
        horizon=4,
        source_transition_id="real:pulse",
    )
    ledger.register(replay_spec)
    imagined, prediction = allocator.propose(
        first_state,
        (noop, pulse),
        group_id="g1",
        parent_checkpoint_sha256="parent",
        config_sha256="config",
        data_sha256="data",
        code_sha="code",
        horizon=4,
        mode="imagined",
        world_model_id="model-v0",
    )
    assert imagined.mode == "imagined" and imagined.action == pulse
    assert prediction.gain > 0

    try:
        ledger.register(
            BranchSpec(
                group_id="g0",
                mode="real",
                parent_checkpoint_sha256="different-parent",
                config_sha256="config",
                data_sha256="data",
                code_sha="code",
                action=noop,
                horizon=4,
            )
        )
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched branch siblings must be rejected")
    print("branching self-check passed: failure accounting, matched siblings, calibrated dreaming")


if __name__ == "__main__":
    main()
