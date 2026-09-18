"""Run the smallest end-to-end check for the archive and control seam."""

from __future__ import annotations

import tempfile
import math
from pathlib import Path

from .core import (
    Action,
    Archive,
    Observation,
    OnlineActionStats,
    Outcome,
    Selector,
    run_episode,
)


class ToyTarget:
    """Deterministic target used only to check causal plumbing, not research claims."""

    def __init__(self) -> None:
        self.step = 0
        self.loss = 1.0
        self.quality = 0.0

    def observe(self) -> Observation:
        return Observation(
            step=self.step,
            tokens=self.step * 32_768,
            loss=self.loss,
            quality=self.quality,
            compute_flops=self.step * 1_000_000.0,
            features={"loss_slope": -0.03},
        )

    def actions(self, observation: Observation) -> tuple[Action, ...]:
        del observation
        return (
            Action("cruise", strength=0.1),
            Action("correction", strength=0.5),
            Action("recover", strength=0.2),
        )

    def execute(self, action: Action) -> Outcome:
        gains = {"cruise": 0.03, "correction": 0.12, "recover": -0.03}
        gain = gains[action.kind]
        self.loss = max(0.1, self.loss - max(0.0, gain))
        self.quality += gain
        self.step += 1
        return Outcome(
            after=self.observe(),
            compute_flops=1_000_000.0,
            wall_seconds=0.01,
            reward_task=gain,
            learning_progress=max(0.0, gain),
            done=self.step == 6,
            metadata={"toy": True},
        )


def main() -> None:
    try:
        Action("correction", parameters={"strength": math.nan})
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite action parameters must be rejected")

    with tempfile.TemporaryDirectory() as directory:
        archive = Archive(Path(directory) / "transitions.jsonl")
        controller = OnlineActionStats(
            prior_uncertainty=1.0,
            action_costs={"cruise": 1_000_000.0, "correction": 1_000_000.0, "recover": 1_000_000.0},
        )
        transitions = run_episode(
            ToyTarget(),
            controller,
            Selector("balanced", curiosity_weight=0.5, seed=0),
            archive,
            run_id="self-check",
            max_decisions=6,
        )
        assert len(transitions) == 6
        assert any(item.action.kind == "correction" for item in transitions)
        assert archive.validate() == 6
        records = list(archive.records())
        assert records[-1].parent_id == records[-2].transition_id
        assert records[-1].metadata["toy"] is True
    print("self-check passed: causal archive, adaptive controller, and runner")


if __name__ == "__main__":
    main()
