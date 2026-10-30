# Learned confidence models

This is an experimental detector of material meaning errors or important
omissions in a proposed plain-English rewrite. Synthetic labels are weak and
unverified. All deployed inputs contain **both the original source and candidate**.
No category, intended edit, label, rationale, ID, flag or generator provenance is
passed to the transformer or classifier.

## Fixed representation and controlled comparison

Stage A loads a separate copy of the preserved three-epoch full FLAN-T5-small
encoder. The generator decoder is never loaded or modified. Source and candidate
are jointly encoded as `source report: ... candidate rewrite: ...`; the hidden
states therefore attend to both texts. Concatenated mean source and mean candidate
states give 1,024 learned features. Prefix, separator, EOS and padding tokens are
excluded from pooling. This is contextual hidden-state pooling, not an attention
weight classifier or a manually engineered semantic feature set.

MLP, LightGBM and XGBoost use exactly the same cached vectors and grouped TRAIN
partitions. The MLP uses fit-only standardisation, a 128-unit GELU layer and 0.2
dropout. AdamW starts at 0.001; minimum validation log loss selects its checkpoint,
with at most 30 epochs and patience 5. Tree comparisons are bounded at 400 rounds
with early stopping, two CPU threads and modest depth/leaf limits. No logistic
regression, SVM, manual mismatch features or large hyperparameter search is used.

Stage B loads that selected MLP and its exact encoder, freezes the Stage A
standardisation, then updates the encoder and head end-to-end with separate
learning rates (1e-5 and 1e-4). The default bounded experiment is three epochs,
grouped-validation selection and patience 2, effective batch 32. Adam state is new
for the joint optimisation; this is an explicit new stage. Frozen heads and
generator checkpoints remain controls. Ordinary boosted trees cannot backpropagate
into the encoder; refitting trees on adapted vectors is a separate possible study.

A lower-rate extension is a separate run, not an edit of the three-epoch control.
It may use `--continue-from IMMUTABLE_PARENT/latest.pt --epochs 6
--encoder-lr 3e-6 --head-lr 3e-5` only after the TRAIN validation curve supports it.
The parent `run.json` travels with the checkpoint. The code checks its data/head/
encoder lineage, a completed epoch boundary, an improved terminal model, lower
rates and a larger finite epoch target. Adam/RNG are restored and original files
are preserved. Early stopping remains active. `--epochs` is the total target,
not an additional six epochs. The next extension decision never uses TEST.

## Data and evaluation boundaries

Assembly feature, label, metadata and mask JSONL files align by row position. The
loader verifies file hashes, exact source/candidate pair fingerprints and fixed
report/image component membership. It rejects cross-partition groups, duplicate
pairs, extra feature fields and non-TRAIN source IDs. Patient separation is not
established. All variants for a component share the existing deterministic split:
SHA256 of `confidence-split-v1:` plus the group key, modulo 10,000, gives 80% fit,
10% validation and 10% calibration. Final fitting refuses fewer than 10,000 source
components; extraction can prepare the existing partial assembly safely.

The conservative mask excludes known invalid rows, but does not turn ambiguous
or synthetic labels into verified clinical truth. Source ambiguity, generator
artifacts and domain shift must be discussed in the result readout. Class imbalance
makes accuracy alone misleading: report AUROC, both error and correct-class AUPRC,
balanced accuracy, log loss, Brier score and calibration error with denominators.
Unweighted log loss selects checkpoints. A scalar temperature is fitted only on
the reserved calibration partition; its own calibration score is an in-sample
diagnostic. Official TEST is never opened by these fitting scripts or used to
choose settings. TEST results and group-level uncertainty remain a later step.

`evaluate_confidence.py` requires all fitting/calibration results and an explicit
frozen selection receipt marking the extension decision complete. It verifies
model/temperature/TEST hashes before inference and stores raw scores in atomic
resumable chunks. Frozen MLP logits retain FP32 head arithmetic, matching its
calibration, while encoder and adapted-head autocast match their respective
training recipes. Predeclared length-dependent batching limits attention memory;
it does not change the evidence caps. All held pairs remain recorded and excluded
from scored denominators. Report unique-pair metrics, equally weighted official
report contributions, and source-component bootstrap intervals. Duplicate mapped
official reports do not become independent observations. Category slices can
overlap after identical-pair collapse. Risk/coverage thresholds are declared before
TEST; they must not be selected from TEST performance. This is a synthetic stress
test, not a probability calibration for naturally occurring generator errors.

## Running locally

Install optional classifier libraries in an isolated environment or target folder,
without upgrading the generator environment. The measured implementation used
scikit-learn 1.7.2, LightGBM 4.6.0, XGBoost 3.1.1, SciPy 1.16.3, joblib 1.5.2 and
threadpoolctl 3.6.0 in addition to the existing pinned PyTorch/Transformers runtime.
From the parent project directory (replace paths with local artifacts):

```sh
python confidence/extract_confidence.py --data local-assembly --encoder epoch3-checkpoint --output runs/critic-vectors --gpu-lease runs/critic-gpu.lock --pilot-only --stage-b-pilot
python confidence/extract_confidence.py --data local-assembly --encoder epoch3-checkpoint --output runs/critic-vectors --gpu-lease runs/critic-gpu.lock
python confidence/train_confidence.py --data complete-local-assembly --cache runs/critic-vectors --output runs/critic-frozen --gpu-lease runs/critic-gpu.lock
python confidence/adapt_confidence.py --data complete-local-assembly --stage-a runs/critic-frozen --encoder epoch3-checkpoint --output runs/critic-adapted --gpu-lease runs/critic-gpu.lock
```

Use the same GPU lease for every job and one GPU job at a time. Set CPU thread
caps and `TOKENIZERS_PARALLELISM=false`. Extraction records lengths and truncation,
benchmarks ordinary plus longest cases, pads dynamically, and writes atomic
content-addressed vector chunks. Additional assembly rows can reuse those vectors
when encoder, tokenisation, pooling and runtime are unchanged. Labels and metadata
are not in the vectors. Keep caches, generated text, data, predictions and model
weights outside Git. No external data transfer is required.

Stage A saves best and latest MLP states, optimiser/RNG and epoch history; `--resume`
requires unchanged identity. Stage B saves latest state every five minutes and
at epoch boundaries with optimiser, RNG and exact within-epoch cursor. A `STOP`
file requests a recoverable pause after an optimiser update; remove it before
`--resume`. It never deletes the frozen controls. CPU trees are bounded independent
fits and may be repeated deterministically after failure. A saved best head is
available even if a later experiment fails.

## Data-free assembly utilities

`source_index.py`, `assembly_core.py` and `schema.example.json` are reusable
indexing, repair, partition and contract helpers. Source index files contain IDs,
hashes and byte offsets, remain local and check source fingerprint/staleness.
These utilities do not reproduce machine-specific shard orchestration. Seven
invented-fixture tests accompany the handoff; model tests also check joint
conditioning, padding, masking, leakage rejection, all heads and joint training.

```sh
cd confidence
python -m unittest test_source_index test_assembly_core test_confidence
```

Implementation and resource pilots do not establish predictive quality. No
confidence-model result should be claimed until fitting and held-out evaluation
have actually completed.

References: [T5](https://jmlr.org/papers/v21/20-074.html),
[LightGBM](https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html),
[XGBoost](https://arxiv.org/abs/1603.02754),
[temperature scaling](https://proceedings.mlr.press/v70/guo17a.html).
