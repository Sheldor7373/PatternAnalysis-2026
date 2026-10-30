# Learning across existing training checkpoints

This descriptive follow-up uses existing reports, references and saved weights.
It creates no teacher examples and performs no training. The original official
validation results, report and checkpoints remain unchanged.

Compare the frozen model and the saved epoch-1/2/3 full and LoRA checkpoints on
the same previously used **dev64** manifest. The common generation recipe is
FP32, greedy, batch 32, source cap 384, output cap 256 and seed 3710. The exposure
axis is 0, 97,411, 194,822 and 292,233 training examples. Optimiser steps differ
between methods because effective batches changed during their original runs.
Full changed from 2 to 4 at 81,558 exposures; LoRA changed at 17,500. Training
sessions and hardware conditions also differed. This is not a controlled causal
comparison of learning rates or adapter capacity.

Report all four ROUGE metrics and per-report teacher-forced reference token
negative log likelihood (NLL), including token-weighted aggregate NLL. Its
exponentiated negative is the geometric mean conditional reference-token
probability. It measures agreement with this particular truncated reference;
it is **not calibrated correctness confidence**, factuality, or the probability
that a generated report is safe. Teacher forcing supplies the reference prefix,
unlike free-running generation. Both inputs and references retain the original
384/256 training caps, so likelihood does not assess text beyond those caps.

Prespecified descriptive slices are dataset source, source truncated at 384,
and reference truncated at 256. Small or empty slices must be shown as such.
Paired changes and source-grounded cases explain whether increasing likelihood
coincides with better generated reports. This small, previously monitored sample
cannot establish generalisation or independent model-selection significance.
Epoch checkpoints were originally selected on monitored dev256 loss; no new
clean confirmation group or official validation row is consumed for this study.

Generation uses the existing checksummed prediction supervisor and immutable
checkpoint identities. The diagnostic command resolves each checkpoint once,
binds weights, tokenizer, base weights, runtime, packages, dependencies, manifest
and its own code, checks source immutability after inference, and publishes one
atomic result per checkpoint. A completed result with a different identity is
refused. Short incomplete diagnostic invocations can be repeated without
overwriting any completed result. Unit tests compare the masked per-row NLL
against a real tiny T5 loss and verify padding and nonfinite/empty-label handling.
Only one GPU worker runs at a time, after the spectral study is complete.

## Measured trajectories

All seven anchors completed: 448 generated reports and 448 teacher-forced reference diagnostics, using the same 64 IDs and matched runtime/code. The sample contains 64 distinct connected report/image groups. Case IDs beginning `train:` retain their raw-dataset origin; these particular groups belong to prepared DEV and were excluded from generator gradient training. No checkpoint was retrained or reselected using this study.

| Anchor | Examples seen | Optimiser step | R1 | R2 | RL | RLsum | Reference NLL | Generation seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| frozen | 0 | 0 | 0.27466 | 0.11312 | 0.24097 | 0.25414 | 2.70844 | 8.24 |
| full-epoch1 | 97,411 | 44,743 | 0.71206 | 0.50666 | 0.64728 | 0.67455 | 0.62943 | 9.80 |
| full-epoch2 | 194,822 | 69,096 | 0.72008 | 0.51051 | 0.65185 | 0.68119 | 0.57601 | 6.72 |
| full-epoch3 | 292,233 | 93,449 | 0.72980 | 0.52880 | 0.66612 | 0.68929 | 0.55913 | 9.95 |
| lora-epoch1 | 97,411 | 28,728 | 0.58242 | 0.34359 | 0.50558 | 0.53882 | 1.30666 | 12.40 |
| lora-epoch2 | 194,822 | 53,081 | 0.59505 | 0.36384 | 0.51761 | 0.55469 | 1.21663 | 12.87 |
| lora-epoch3 | 292,233 | 77,434 | 0.60783 | 0.37732 | 0.53172 | 0.56484 | 1.18924 | 10.87 |

ROUGE values are mean F1 on [0,1]. NLL uses natural logarithms, sums token losses and divides by nonpadding reference tokens. The aggregate NLL is token-weighted; it differs from averaging per-report NLL. Generation scores use the full decoded reference text even when teacher-forced likelihood caps that reference at 256 tokens. These different scopes matter for long examples.

![Learning trajectories on existing dev64](figures/checkpoint-learning-trajectories.png)

### Paired changes

The following are ROUGE-L changes with 95% percentile intervals from 1,000 paired connected-component resamples (seed 3744). Each component has one row here. These are conditional exploratory intervals on a previously inspected small development sample, without multiple-comparison correction; they are not a fresh confirmation test.

- frozen->full-epoch1: +0.40630 [+0.36055, +0.44801].
- full-epoch1->full-epoch2: +0.00457 [-0.01327, +0.02171].
- full-epoch2->full-epoch3: +0.01427 [+0.00191, +0.02651].
- frozen->lora-epoch1: +0.26460 [+0.22514, +0.30301].
- lora-epoch1->lora-epoch2: +0.01203 [-0.00755, +0.03166].
- lora-epoch2->lora-epoch3: +0.01412 [+0.00361, +0.02608].

Most measured improvement occurs by epoch 1. Subsequent reference loss decreases consistently in aggregate, while overlap gains are smaller and per-case behaviour is not monotone. Epoch-1-to-2 overlap intervals cross zero for both methods; epoch-2-to-3 intervals are positive on this particular sample. This supports reporting diminishing returns and metric differences, rather than inferring that loss, semantic fidelity and readability are interchangeable. It does not retrospectively change the original dev256 checkpoint selection.

### Prespecified slices

| Slice | Rows | Full epoch-3 RL | Full epoch-3 NLL | LoRA epoch-3 RL | LoRA epoch-3 NLL |
|---|---:|---:|---:|---:|---:|
| all | 64 | 0.66612 | 0.55913 | 0.53172 | 1.18924 |
| source:BIMCV-COVID19 | 14 | 0.61169 | 0.59253 | 0.45065 | 1.29400 |
| source:OpenI | 2 | 0.69523 | 0.74195 | 0.57483 | 1.14007 |
| source:PadChest | 48 | 0.68078 | 0.53358 | 0.55357 | 1.15574 |
| input_truncated:False | 63 | 0.66859 | 0.55146 | 0.53578 | 1.16840 |
| reference_truncated:False | 63 | 0.66859 | 0.55146 | 0.53578 | 1.16840 |
| input_truncated:True | 1 | 0.51073 | 0.66701 | 0.27638 | 1.48227 |
| reference_truncated:True | 1 | 0.51073 | 0.66701 | 0.27638 | 1.48227 |

There are only two OpenI rows and one source/reference-truncated row. The same single example triggers both truncation slices, so their effects cannot be separated. These counts do not support broad subgroup or cutoff claims. All checkpoint slice values remain in the local aggregate comparison.

### Four fixed source-grounded cases

All seven outputs were checked against full/visible source and reference for each fixed seed-3745 case. These brief paraphrases illustrate mechanisms, not error prevalence or clinician adjudication.

| Case | What the trajectory reveals |
|---|---|
| train:97605 | Full outputs remain identical while reference likelihood changes. LoRA loss improves slightly but generated text increasingly misbinds the location/nature of the finding; ROUGE-L falls from 0.4255 to 0.3077. Lower reference loss does not guarantee a better greedy output. |
| train:93726 | Full preserves key negative lung findings and line placement while lexical/detail changes lower overlap. LoRA epochs 2 and 3 produce identical text despite lower loss, retaining vessel jargon. |
| train:120236 | Full preserves bilateral peripheral cloudiness and its small increase. LoRA epoch 2 attributes size/density to the lungs themselves, then epoch 3 partly recovers cloudiness. Anatomic attachment errors can persist despite improving aggregate loss. |
| train:73063 | Full learns to retain clips and location, later adding an explanation. LoRA outputs stay identical while reference NLL falls from 0.7748 to 0.6424, leaving location vague. |

Reference likelihood requires the labelled reference and its prefixes; it is not available as this same quantity on an unlabelled deployment report. Its exponent is neither an empirical correctness rate nor a calibrated probability that a generated report is factual. This study fits no confidence classifier and makes no clinical safety claim.

### Resources, recovery and provenance

Successful generation-supervisor sessions total 185.9 seconds, including loading/scoring; timed reference diagnostic passes total 11.1 seconds, excluding model loading. Peak sampled generation device use is 1369 MiB. Single ordered timings include differing output lengths and are not isolated checkpoint speed benchmarks. Frozen/full/LoRA produce 3/1/2 outputs without EOS at their final anchor, respectively; no EOS count is a semantic completion judgement.

The initial actual-model loss consistency assertion stopped safely before publishing a diagnostic. The per-row calculation was changed to use the same flattened-vocabulary cross-entropy path as Transformers T5, and all 64 frozen rows then passed. A standalone verification used a different tokenizer-runtime setting; strict identity binding correctly refused mixing it into the workflow. It was retained separately, and diagnostics were rerun consistently. Completed generation batches were reused; original logs and failure receipts remain. Final integrity checks verify all seven diagnostic identities/row hashes, 448 paired IDs, aggregate loss reconstruction, immutable checkpoints, 13 spectral generation jobs, seven checkpoint generation jobs, and unchanged original 30,000 predictions/README/PDF.

Manifest SHA-256: `db03f331690a106a2504f4ac269de2054d1db9c87e272d55a7e34a9cf4f610dd`. Exact immutable checkpoint completion hashes are:

- full-epoch1: `50ed569fe213e621e94a712e0e93f7a1cd616669bb35275e56b0df0d30360221`.
- full-epoch2: `9157642380e2b585af137f2d50cb1778b87752314a2d3b1add91bb47194854ea`.
- full-epoch3: `fda47b10d94ba377c4ed1279c16b7b4d994a21b3d4019f20dfbf2298f863d358`.
- lora-epoch1: `67806c79d97677ae562fde481756794346dc8f568f14272ecf4ee058cd137474`.
- lora-epoch2: `ddbda3c2b0338d1b1b846894c5583939c8843507eedc57216a61d0f9106028c6`.
- lora-epoch3: `2f5800e64b514c0b2ac4c35ea04422cf20069d9516b3e7adff408f92f1b02053`.

The frozen base remains pinned to FLAN-T5-small revision `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab`; local manifests also bind actual base/tokenizer bytes. Public files contain aggregate results and minimal case paraphrases, not raw data or weights. Codex assisted code, testing, analysis and writing; the author remains responsible. The original official-validation report stays unchanged, and no upstream PR, merge or submission is performed by this follow-up.
