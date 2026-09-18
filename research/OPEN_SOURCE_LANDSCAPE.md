# Open-source landscape for the history-jump driver

Research cutoff: 2026-09-18. This is a decision-oriented inventory for the
first RTX 5090 experiment. The links are primary papers, official repositories,
or official model/dataset cards. Reported results belong to their original
settings; none is evidence that this project can skip general language-model
pretraining.

## Selected first stack

| Asset | Use here | Status and boundary |
| --- | --- | --- |
| [Pico LM](https://github.com/pico-lm) and [Pico Decoder Medium](https://huggingface.co/pico-lm/pico-decoder-medium) | Candidate source of small decoder trajectories with weights, AdamW state, gradients, and activations | Highest-value external trajectory candidate. Verify checkpoint cadence, artifact layout, and terms before downloading; our own traces remain authoritative. |
| [OLMo](https://github.com/allenai/OLMo) [checkpoint format](https://github.com/allenai/OLMo/blob/main/docs/Checkpoints.md) | Exact optimizer-state-aware replay reference | Use only a small slice if the existing loader cannot provide enough varied trajectories. It is heavier than the current target loop. |
| [Pythia](https://github.com/EleutherAI/pythia) | A family of small causal LMs with many same-run weight checkpoints and a known data order | Use as a learning-dynamics reference and later target/control. Its intermediate optimizer states are generally not available at scale, so it cannot replace our own complete AdamW traces. |
| [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) and its [architecture config](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/main/config.json) | A concrete small modern decoder recipe for a later changed-architecture check | Reference only at first. The current campaign already has a reproducible 85M decoder; changing the target now would mix architecture effects into the driver result. |
| [TinyStories](https://arxiv.org/abs/2305.07759) | Fast language-learning loop for collecting many complete trajectories | First external corpus after the local toy calibration. Its narrow distribution limits any broad pretraining claim. |
| [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) | Fixed larger-domain shard for validation after the trace format works | Use a versioned small shard, not the full corpus. Record the dataset-card terms and checksum. |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | Changed data-distribution holdout | Later transfer test, not a first dependency. |
| [NiNo](https://github.com/SamsungSAILMontreal/nino) | Closest public reference for predicting future weights from recent weight states | Port or reproduce its graph-free WNN+ idea only after local traces exist. Its released predictors are trained for particular model families and are not a drop-in AdamW controller. |
| [Celo2](https://github.com/amoudgl/celo2) / [PyLO](https://github.com/Belilovsky-Lab/pylo) | Compact learned-optimizer baselines and implementation references | Keep outside the runtime dependency set. Port one minimal update rule or compare offline after the AdamW control is sound. |
| [learned_optimization](https://github.com/google/learned_optimization) | Reference for meta-training, truncated unrolls, and learned optimizer tasks | JAX framework with a larger dependency surface; useful for design study, not initial installation. |
| [autoresearch](https://github.com/karpathy/autoresearch) | Minimal single-GPU proposal/evaluate loop pattern | Borrow the small fixed evaluator boundary, not its task or short-run result. |

Pico and OLMo are useful additions to the earlier Pythia plan: public weight
checkpoints alone are passive evidence, while a driver needs optimizer state,
data position, and causal outcomes. Even an external optimizer-aware source
cannot replace branches collected with this repository's exact target loop.

## Useful but deferred

| Asset | Why it matters | Why it waits |
| --- | --- | --- |
| [Gensyn OPEN-1B](https://github.com/gensyn-ai/open-transformers) | A large transparent AdamW training run with frequent checkpoints and state/data provenance | Excellent external validation reference, but too large and implementation-specific for the first 5090 dependency. |
| [LLM360](https://www.llm360.ai/news/introducing-llm360-fully-transparent-open-source-llms.html) | Long-horizon checkpoint and data-history reference | The released scale is too expensive for initial local work. |
| [TaskSet](https://github.com/google-research/google-research/tree/master/task_set) | Large historical optimization-task corpus for offline learned-optimizer analysis | Mostly old TensorFlow/Sonnet infrastructure and learning curves rather than causal LM AdamW state; no download until our schema has a consumer. |
| [Open-L2O](https://github.com/VITA-Group/Open-L2O) | Public learned-optimization benchmark and code | Good control source for synthetic/vision tasks, but its environment is not the first target loop. |
| [COCO/BBOB](https://github.com/numbbo/coco) | Smoothness and conditioning suite for controlled landscape pretraining | Useful for generated/benchmark landscapes, but neural overparameterization still needs local synthetic families. |
| [OptFormer](https://github.com/google-research/optformer) | History-conditioned experiment selection | More directly relevant to the outer allocation loop than to a weight jump; revisit after real branches exist. |
| [DreamerV3](https://github.com/danijar/dreamerv3) and [TD-MPC2](https://github.com/nicklashansen/tdmpc2) | Model-based planning precedents | No imagined rollout in the first pilot. Planning begins only after action-conditioned predictions calibrate on fresh branches. |
| [Dion](https://github.com/microsoft/dion), [Muon](https://arxiv.org/abs/2502.16982), and [SOAP](https://arxiv.org/abs/2409.11321) | Stronger optimizer baselines and structured update primitives | The first causal question is whether a driver can beat AdamW. Screen these later with whole-run cost, not optimizer-step headlines. |
| [Graph HyperNetworks / DeepNets-1M](https://github.com/facebookresearch/ppuda) | Architecture-conditioned parameter generation | Useful if tensor-role conditioning fails; not needed for the first fixed decoder. |
| [Doc-to-LoRA](https://github.com/SakanaAI/doc-to-lora) | Context-to-adapter mechanism analogous to fast driver adaptation | This is an analogy, not evidence for training-trajectory control; defer until an online adapter has signal. |
| [Mamba](https://github.com/state-spaces/mamba) | Efficient long-memory sequence backbone | Use only if telemetry history, rather than prediction quality, becomes the bottleneck. |
| [Temporal Organization of Weight Corrections](https://zenodo.org/records/21843664) | Open Adam weight-correction trajectories on a small vision problem | Optional auxiliary pretraining/evaluation. It is not language-model data and must not silently become the main evidence. |
| [MLCommons Algorithmic Efficiency](https://github.com/mlcommons/algorithmic-efficiency) | Time-to-target and fixed-hardware evaluation precedent | Borrow accounting and failure handling if useful; it is a benchmark harness, not a trajectory dataset. |

## Decisions for this repository

1. Do not vendor or clone any external project and do not add JAX,
   `torch_geometric`, `transformers`, or dataset SDK dependencies yet. The
   current PyTorch environment is enough to collect the first traces.
2. Keep AdamW as the fixed target optimizer for the first history-jump test.
   Muon, SOAP, and Dion are comparison baselines after the action interface
   has a measured effect.
3. Use the current toy landscapes to validate telemetry and causal branch
   accounting, then TinyStories-sized language runs for trajectory collection,
   then a fixed FineWeb shard for a changed-data check. Pico/OLMo can be added
   only when their artifact manifests pass the same provenance checks.
4. Use a numerical causal transformer with a small per-run adapter for the
   driver. Do not start with a pretrained language backbone: it may spend
   capacity on language while importing little numerical dynamics knowledge.
5. Treat NiNo as the nearest nowcasting reference, not as a dependency. Begin
   with a graph-free recent-update basis that fits the existing checkpoint
   contract.
6. Store full model, AdamW, RNG, scheduler, scaler, and data-cursor state at
   branch boundaries. Weight-only public checkpoints are not enough for causal
   jump evaluation.

## What the sources do and do not establish

NiNo reports future-parameter prediction from a short history and applies a
prediction only occasionally in its training loop. That makes the direction
plausible, but does not show that a jump preserves long-horizon progress in our
decoder. Celo2 and PyLO show practical learned-update implementations, but
their meta-training tasks and frameworks differ from this project. Pythia is
valuable because its checkpoint series fixes data order and exposes learning
dynamics, while its missing intermediate optimizer states are exactly why this
repository must collect its own traces.

The first claim remains narrow: on held-out complete runs, can a driver choose
a bounded AdamW history jump that reduces end-to-end cost to the same quality,
including probes, rejected actions, recovery, and driver overhead?
