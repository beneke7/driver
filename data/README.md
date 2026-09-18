# Local data layout

This directory is a manifest entry point, not a checked-in dataset. Raw data,
tokenized shards, trajectory tensors, and checkpoints are ignored by Git.

```text
data/
├── external/      # optional upstream exports or source archives
├── raw/           # downloaded text or source model files
├── pretokenized/  # deterministic, versioned token shards
└── trajectory/    # per-step target/AdamW traces and split manifests
```

Every imported asset needs a sidecar manifest with at least:

- upstream URL and revision or dataset snapshot;
- local filename and byte size;
- license or usage terms copied from the upstream card/repository;
- SHA-256 checksum;
- tokenizer, vocabulary, filtering, and train/evaluation split details when
  applicable.

Initial order:

1. keep the existing deterministic toy landscapes as calibration data;
2. collect complete AdamW traces on a fixed TinyStories-sized language loop;
3. use a fixed small FineWeb shard only after the trace and evaluator are
   stable;
4. use FineWeb-Edu or a changed architecture as a transfer holdout.

Do not download a full corpus or clone a research framework as part of setup.
The first missing asset is our own complete trajectory archive: public model
weight checkpoints generally do not include the intermediate AdamW state needed
for causal branch replay.
