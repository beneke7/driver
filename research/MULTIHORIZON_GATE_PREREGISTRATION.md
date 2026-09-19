# Multi-horizon cost-aware gate preregistration

Date: 2026-09-19. This is a policy-evaluation rule over an already-trained
response model. It is not a new model or a promotion claim.

## Rule

For each supported group and each non-noop candidate, select the candidate only
when all conditions hold:

1. its measured wall, FLOP, and token costs are no higher than matched no-op;
2. its final predicted loss delta plus the fitted residual radius plus three
   ensemble standard deviations is strictly below `0`;
3. its recovery predicted loss delta plus the same uncertainty margin is at
   most `1e-3`;
4. its immediate predicted loss delta plus the same uncertainty margin is at
   most `1e-3`.

Otherwise select no-op. If several candidates qualify, choose the one with the
lowest conservative final predicted loss per measured wall cost. The action
cost is known from the branch ledger; outcomes are not used at decision time.

The `1e-3` tolerance is the existing actionability tolerance. The factor of
three is fixed before reading this policy's evaluation. No threshold, horizon,
or multiplier may be tuned per holdout.

## Evaluation

Apply the unchanged model checkpoints to:

- the three complete-seed FineWeb 85M timing holdouts (seeds 24--26);
- the changed-regime leave-one-seed-out reports (seeds 9--11).

Report selected action, support, all three horizon deltas, cost ratios,
prediction/reality gaps, and comparison with the action-only prior. A selected
branch with a positive recovery delta beyond tolerance is a policy failure even
if its final loss improves. This rule does not authorize online learning,
imagined rollouts, or a larger steerer.

## Decision boundary

The rule is useful only if it removes the observed recovery false positives
without collapsing to no-op on all supported timing holdouts. It is not a
promotion gate: even a positive result remains one action family, one small
target family, and far below the 10x contract.
