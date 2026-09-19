# Same-corpus block-mixture oracle

Date: 2026-09-19

This experiment tests a fixed allocation action, not a learned selector. From
matched AdamW parents at step 1536, the candidate replaces 25% of the next
768-step continuation with a predeclared tail interval from the same training
file. Four of every sixteen 16,512-byte optimizer-step blocks are alternate
data; the rest preserve the ordinary continuation. Tokens, optimizer steps,
validation, and charged FLOPs are equal.

## FineWeb-Edu oracle

Run: [`manifest.json`](../runs/data-mixture-oracle-fineweb-balanced-v1/manifest.json)

Six roots completed without failures: three 85M and three 139M. The alternate
interval is `[38,772,237, 41,942,541)`, outside both the parent prefix and the
target continuation interval.

| Width | Immediate mean delta | Recovery mean delta | Final mean delta | Final wins |
| ---: | ---: | ---: | ---: | ---: |
| 85M | `-0.01481` | `-0.00470` | `-0.01180` | 3/3 |
| 139M | `-0.01691` | `-0.00516` | `-0.01323` | 3/3 |

This is a real durable equal-budget quality signal in this corpus: every root
improves at all three measured horizons. It is not a speedup. Candidate and
baseline FLOPs are exactly equal, and candidate/baseline branch wall ratios
range from `0.997x` to `1.002x`.

## Changed-data transfer check

Run: [`manifest.json`](../runs/data-mixture-oracle-tinystories85-transfer-v1/manifest.json)

The same 25% schedule was applied to TinyStories 85M seeds 3--5 using the
disjoint tail interval `[63,938,559, 67,108,863)`. The mixture improved
immediate and recovery loss on all three roots, but regressed at the final
horizon on all three:

| Immediate mean delta | Recovery mean delta | Final mean delta | Final wins |
| ---: | ---: | ---: | ---: |
| `-0.00803` | `-0.03040` | `+0.00163` | 0/3 |

This is the required long-horizon warning: transient utility does not transfer
to durable capability. The FineWeb result is therefore treated as a
corpus-specific fixed action, not evidence for a general landscape driver.

## Decision

The action has an oracle ceiling for equal-budget FineWeb quality, but no
conservative cost-to-threshold gain and no changed-data transfer. Do not train
a data-mixture selector or combine this action with a larger steerer. Keep it
as a fixed FineWeb control if future experiments need a strong same-budget
baseline; close it as a route toward the robust 10x claim.

