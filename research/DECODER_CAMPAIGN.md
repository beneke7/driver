# Decoder development campaign

The first target-model campaign ran from commit `33044e4` on the RTX 5090.
It uses the project-local CUDA environment and a fully trainable
85,350,912-parameter decoder (`width=768`, `layers=12`, `heads=12`,
`context=128`, `vocab=128`). All branches use AdamW state restored from one
immutable parent per landscape/seed. The fixed schedule is 32 prefix steps,
then 4 immediate, 32 recovery, and 128 final steps at batch 128. Each branch
consumes 2,097,152 target tokens after the parent.

The frozen development roster is:

- `delayed_copy`, `phase_switch`, and `text_shard`;
- seeds `0`, `1`, and `2`;
- six same-budget policies: `noop`, `role_pulse`, `open_loop`,
  `shallow_controller`, `history_retrieval`, and `online_lr_control`.

The campaign wrote `runs/decoder-development-33044e4/manifest.json`,
`transitions.jsonl`, `action_ranking.json`, `capability_curves.json`, and one
977 MiB parent checkpoint per case. The run covered 9 cases and all 54 planned
branches, with zero failures. The manifest records SHA-256 code, lockfile,
objective, data, config, and parent-checkpoint hashes. The text shard rotates a
deterministic fixed shard by seed, so its three seeds are not identical byte
streams.

The branch archive contains synchronized end-to-end timings and target,
driver, recovery, evaluation, token, and FLOP accounting. Median branch wall
time was approximately 11.0 seconds and median branch token count was
2,097,152 for each policy. The common prefix is included in the capability
curves rather than treated as free.

The action-ranking report is a calibration result, not a claim of learned
control. Across its 54 predictions, prediction RMSE was `2.03` loss units and
top-1 agreement with the realized best sibling was `11.1%`; mean final regret
was `0.0093`. History retrieval excludes seed 2 outcomes from its fitting
history, leaving that seed as the first chronological transfer check.

Each policy has parent/immediate/recovery/final capability points and three
predeclared relative validation-loss thresholds (`0.995`, `0.99`, `0.98` of
the parent loss). Five of nine cases reached each threshold per policy in this
short development horizon; the remaining cases are recorded as censored
`max_steps` outcomes in the capability artifact. No imagined branches were
used.

This is not the promotion gate. The required next campaign is five held-out
landscapes × three fresh seeds × three thresholds plus a changed data or
architecture condition, with the contract’s geometric-mean, per-landscape,
per-seed, and clustered-confidence gates. The current development result
does not establish a 10× training speedup.
