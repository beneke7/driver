# Loss-ranked token-drop preregistration

Date: 2026-09-20

This is a new, bounded work-allocation action. It is a causal pilot, not a
learned driver or a 10x claim. The whole-layer depth curriculum is closed and
is not combined with this action.

## Motivation and hypothesis

Whole-layer skipping regressed because every upper-layer parameter re-entered
from a stale state. Token-level allocation tests a different hypothesis: easy
tokens may not need every upper-block update while hard tokens continue to
receive the full path. Token dropping has been studied as a pretraining
efficiency mechanism, including loss-based selection in
[Token Dropping for Efficient BERT Pretraining](https://aclanthology.org/2022.acl-long.262/),
layer dropout in [LayerSkip](https://aclanthology.org/2024.acl-long.681/), and
random layerwise token dropping in
[Random-LTD](https://arxiv.org/abs/2211.11586). Those results do not establish
decoder AdamW training speedups here; this pilot tests only a small compatible
variant.

## Frozen action

Use the existing 85M target (width 768, 12 layers, 12 heads, vocabulary 256,
context 128, batch 128, AdamW) on the versioned FineWeb-Edu byte streams. From
each immutable parent at global step 1280, fork matched `noop` and
`token_drop_curriculum` branches with immediate/recovery/final horizons
128/512/768.

For the first 384 branch steps only:

1. run the lower 6 blocks normally;
2. use the shared output head on the lower representation to compute a
   per-token next-byte loss probe;
3. keep the highest-loss 64 of 128 tokens in each sequence for upper-block
   attention-output and MLP computation;
4. leave the lower-loss token states in the upper blocks' residual stream while
   retaining all tokens as causal keys/values;
5. restore dense full-block computation for steps 384--767.

The probe is deterministic, uses no validation data, and has no trainable
selector. The parent optimizer state, data cursor, RNG, target tokens, and
branch horizons remain matched. The candidate has the declared state effect:
inactive token positions receive no upper-block attention-output or MLP
gradient during the first 384 steps. QKV and causal attention still run for
all tokens, so the action has a deliberate compute ceiling and is not described
as full token dropping.

## Cost and safety accounting

Charge the intermediate loss probe as `probe_flops`, upper-block sparse work as
the reduced `target_flops`, all validation and checkpoint work, and measured
end-to-end wall time from the common parent. Training tokens and raw data
cursor movement must be recorded and equal by branch. The manifest must retain
the action fraction, mask start, per-step mask fraction, probe work, hashes,
optimizer state, and code revision.

A root is a durable-safe pass only when both branches are finite, the candidate
has no catastrophic failure, and candidate-minus-noop validation loss is at
most `+1e-3` at immediate, recovery, and final horizons. The pilot remains
open only if at least 2 of 3 fresh seeds pass and the candidate has a positive
end-to-end wall and conservative-FLOP advantage after shared-prefix costs.
Otherwise close this token-drop action without adding a selector. A positive
pilot would still require a changed-data or width holdout before any policy.

## Reproduction command

```bash
CUDA_LAUNCH_BLOCKING=1 uv run python -m driver.decoder_campaign \
  --output runs/token-drop-fineweb85-p1280-s36-38-v1 \
  --landscape phase_switch --seed 36 --seed 37 --seed 38 \
  --strategy noop --strategy token_drop_curriculum \
  --width 768 --layers 12 --heads 12 --vocab-size 256 --context 128 \
  --batch-size 128 --prefix-steps 1280 \
  --immediate-steps 128 --recovery-steps 512 --final-steps 768 \
  --token-drop-fraction 0.5 --token-drop-start-layer 6 \
  --token-drop-steps 384 \
  --train-file data/raw/FineWeb-Edu-train-40m.txt \
  --validation-file data/raw/FineWeb-Edu-valid-8m.txt --device cuda
```
