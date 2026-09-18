# Contract result: quadratic optimization track

The first robust gate is achieved on a deliberately narrow target: solving
ill-conditioned positive-definite quadratic landscapes from a value/gradient
oracle. It is a mechanism result, not a language-model or general-training
claim.

The evidence was generated from commit `d3f6619` on the RTX 5090 with
`torch 2.14.0+cu130`. The held-out roster contains five families, three common
held-out seed labels per family (15 cases total), and dimension 8:

- diagonal log-spectrum;
- randomly rotated log-spectrum;
- randomly rotated two-block spectrum;
- randomly rotated clustered spectrum;
- randomly rotated random spectrum.

AdamW learning rates were selected on four development seeds per family. The
driver sees only the oracle interface: it tries a diagonal secant action, up to
two conjugate-gradient probes, and a full coordinate probe as a fallback. The
generated matrix is not passed to the candidate action code. The quality
targets are relative objective thresholds `1e-4`, `3e-5`, and `1e-5`.

The fail-closed contract requires five landscapes, three seeds, complete
coverage, no branch failures, a per-landscape median of at least 10×, a
per-seed median of at least 10×, a geometric mean of at least 10×, and a
clustered 95% lower bound of at least 10×. Serial per-case wall time is the
primary metric; estimated FLOPs are secondary.

| Relative target | Wall geometric mean | Clustered 95% lower bound | Lowest landscape median |
|---|---:|---:|---:|
| `1e-4` | 26.36× | 22.82× | 21.45× |
| `3e-5` | 29.52× | 24.34× | 22.46× |
| `1e-5` | 43.07× | 28.34× | 23.54× |

The serial evaluator takes three synchronized steady-state timings per case.
Baseline development tuning and fixed driver warmup are recorded as one-time
costs and amortized over 45 threshold/case deployments. Failed branches are
not converted into speedups; the largest threshold used a 50,000-step cap so
all baseline branches reached the target instead of treating a censored cap as
an exact cost.

Reproduction uses fresh output directories:

```bash
.venv/bin/python -m driver.quadratic_benchmark \
  --output runs/contract-q-1e-4 \
  --dimension 8 --condition-floor 1e5 --threshold 1e-4 \
  --max-steps 10000 --development-seeds 4 --heldout-seeds 3 \
  --timing-replicas 64 \
  --learning-rates 0.05 0.1 0.2 0.3 0.5 0.7 1.0

.venv/bin/python -m driver.serial_quadratic \
  --source runs/contract-q-1e-4 --output runs/contract-q-1e-4-serial
```

The decoder branch harness does not pass this gate: its fixed role pulse has
not shown a reliable improvement. The quadratic result also does not establish
transfer to a transformer, a changed dimension, a changed data distribution,
or a 10× reduction in general pretraining compute. Those are the next
promotion tests, not hidden assumptions behind this result.
