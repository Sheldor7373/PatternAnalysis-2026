# One additional full-model epoch

This separately authorised experiment continues the completed full-model
three-epoch checkpoint. The original report, official-validation results,
selected model and resumable final checkpoint remain unchanged.

The new run starts at step 93,449, exposure 292,233 and epoch cursor (3, 0).
It trains exactly 97,411 further examples: final exposure 389,644, step 117,802.
Batch 4, accumulation 1, BF16, source/target caps 384/256, AdamW moments and
update counters, weight decay 0.01, clipping, RNG and data lineage are preserved.
The next shuffled epoch uses the existing seed-plus-epoch loader mechanism.

The completed original scheduler has zero next LR. An explicit
`--restart-schedule` transition sets the first additional update to **3e-5**
(original peak 1e-4), then decays linearly by additional examples to zero over
one epoch. There is no new warmup. This intentionally changes the schedule;
it is not a bit-identical continuation of the original schedule. Normal resume
still refuses silent epoch/LR changes. The extension path requires an exactly
completed source budget, consistent zero-LR scheduler/optimizer state, unchanged
batch/accumulation and exactly one additional epoch. Subsequent recovery restores
the new schedule counters, optimizer moments, RNG and cursor from its own latest.

An immutable imported checkpoint is published before the first update. Existing
atomic/checksummed saves, finite loss/gradient checks, thirty-minute checkpoint
cadence and fifteen-minute dev256 diagnostics remain. Read-only health observations
run at least every ten minutes until termination. No arbitrary elapsed-time cutoff
is used; the finite exposure target remains enforced. Thermal safeguards remain.

The original best reference is copied into this separate run, so a non-improving
extra epoch cannot silently replace the baseline selection. After completion,
compare **the extra epoch's final/latest checkpoint** with epoch 3 on the same
already-used dev64 generation/reference-diagnostic set. Keep original official
validation results unchanged; do not use official validation to tune this extension.

CPU regressions verify handoff weights, Adam moments/counters, RNG and cursor,
new LR and global scheduler counters, exact short-final-batch exposure, unchanged
source hashes, resumed versus uninterrupted final states, and rejection of
incomplete source budgets, silent changes, extra epochs or batch changes.
The targeted 28-test training/recovery/checkpoint suite passed before launch.
The real source completion SHA-256 is
`7b6e6f2b8f2b0b27a1bce9bcdf6bad1d1dd94dd0e35a281f0f1208f5ad018c7c`.

## Completed experiment

The run finished exactly **389,644 exposures / 117,802 global optimiser steps**: one extra epoch, not an open-ended continuation. Successful worker exit and final model/Adam/scheduler/cursor/RNG checkpoint checks preceded all inference. The final dev256 BF16 loss was **0.569939**, versus **0.578559** at epoch 3. The extra-run best improved under the existing dev selection rule; the original three-epoch baseline remains immutable.

The matched FP32 generation comparison reuses the same previously inspected **64 development report/image groups**. Seed 3710, greedy decoding, batch 32 and 384/256 source/output caps are identical. Baseline cached weights match the resumed training parent byte hashes. Runtime, package, source-code and manifest identities match. No new official-validation evaluation or tuning was performed.

| Metric | Epoch 3 | Epoch 4 | Paired change | 95% group-bootstrap interval |
|---|---:|---:|---:|---:|
| ROUGE-1 | 0.72980 | 0.73554 | +0.00574 | [-0.00209, +0.01468] |
| ROUGE-2 | 0.52880 | 0.53685 | +0.00805 | [-0.00277, +0.02311] |
| ROUGE-L | 0.66612 | 0.66827 | +0.00215 | [-0.00749, +0.01309] |
| ROUGE-Lsum | 0.68929 | 0.69596 | +0.00666 | [-0.00115, +0.01708] |
| Token-weighted reference NLL | 0.55913 | 0.55154 | -0.00758 | [-0.01036, -0.00466] |

Intervals use 1,000 shared paired draws over the 64 sampled groups, seed 3746. Token-weighted NLL resamples each group's token-loss sum and token count, then recomputes the ratio; mean per-report NLL is a different estimand. ROUGE scores are mean F1 on [0,1]. All four overlap intervals cross zero: the extra epoch lowers reference loss, but this small reused sample does **not establish a generation-quality improvement**. Teacher forcing supplies reference prefixes and is distinct from free-running generation. Reference NLL requires labels and is not calibrated factual-correctness confidence. No multiple-comparison correction or independent confirmation is claimed.

![Extra epoch monitoring and matched generation](figures/extra-epoch-comparison.png)

### Length, cutoff and repetition

| Diagnostic | Epoch 3 | Epoch 4 |
|---|---:|---:|
| Empty outputs | 0.00 | 0.00 |
| Outputs without EOS | 1.00 | 0.00 |
| Inputs clipped at384 | 1.00 | 1.00 |
| Repeated-4gram candidates | 0.00 | 0.00 |
| Mean generated tokens | 58.55 | 58.12 |
| Median generated tokens | 54.50 | 54.00 |
| 95th percentile tokens | 112.70 | 117.55 |
| Maximum generated tokens | 256.00 | 239.00 |

Generated lengths include the first EOS and exclude decoder start/padding. Thirty-five of 64 output strings are exactly unchanged; ROUGE-L improves for 14, declines for 13 and ties for 37. The repetition flag requires at least 24 words, a four-gram repeated at least 3 times and duplicate-fourgram fraction at least .3. It is a review heuristic, not an error rate. Removing the sole noEOS output does not prove semantic completeness; the unchanged input cap still hides part of one source.

### Source-grounded changes and regressions

Four largest positive/negative RL changes were selected for diagnosis, alongside the previous four fixed seed-3745 cases and the former noEOS case. They are not a prevalence sample. IDs beginning `train:` retain raw-dataset origin; these groups belong to prepared DEV, outside generator gradient training. Public paraphrases are minimal; full local comparisons remain private. No clinician adjudication is claimed.

| Case | What changed |
|---|---|
| train:58584; RL +0.26561 | The biggest gain mostly comes from closer reference wording and technical tube names. Both preserve placement/negative findings; epoch 3 provides more plain-language explanation. Higher overlap is not automatically better readability. |
| train:13879; RL +0.09800 | Both preserve the main finding/stability but keep a date interpretation inconsistent with the reference/source format. The metric gain does not repair that error. |
| train:86709; RL -0.09878 | Both retain the principal findings and uncertainty. The large overlap drop largely reflects a different spinal-curvature paraphrase, with no clear major new factual failure established. |
| train:17567; RL -0.09365 | Epoch4 drops a plain-language COPD explanation and risks placing a fat pad inside the heart rather than at its adjacent angle. This is a substantive source-visible regression concern. |
| train:145841; termination diagnostic | Epoch4 corrects an unsupported history detail and reaches EOS earlier, but ends with unsupported negative density wording. The source is449tokens clipped to384, hiding later cyst details. EOS still does not certify complete or faithful coverage. |

The fixed cases also contain unchanged greedy outputs whose reference likelihood moves in either direction. Loss, greedy decisions, overlap, readability and factual preservation must be evaluated separately. **Decision:** retain the epoch 4 candidate and its lower-loss checkpoint, while keeping the original official baseline/report unchanged; do not claim robust semantic superiority from dev64.

### Cost, recovery and provenance

The extra training loop took **9864.0 seconds (2h44m24s)**, processing 6,273,565 target tokens, with peak allocated GPU memory 2681.7MiB. This session timer includes in-loop monitoring/saves and excludes initial model/optimizer loading. Generation-only timings were 9.95s for cached epoch 3 and 8.67s for epoch 4; peak allocated generation memory was 823.1/809.2 MiB. These single ordered measurements have different session/thermal/background conditions, so they do not establish a model speedup.

Two observation processes exited during the run; read-only replacements attached to the same surviving worker without restarting training. Native success/failure exit detection was checked using CPU-only processes (exit0 and7). Training loss/gradient checks and original checkpoints remained intact. Ordinary-window throughput varied substantially with host load. Collector indexing was rolled out during training, but uncontrolled workload/adoption changes prevent attributing rate changes to it. No thermal/security safeguards were altered.

Final latest checkpoint completion SHA-256: `9a5934dc7ef2d6eabf05318f93b9d57e4ec73a0954a558238637724fb17938dc`. Training parent: `7b6e6f2b8f2b0b27a1bce9bcdf6bad1d1dd94dd0e35a281f0f1208f5ad018c7c`. Evaluation manifest: `db03f331690a106a2504f4ac269de2054d1db9c87e272d55a7e34a9cf4f610dd`. Full source/dependency, base/tokenizer, output-batch and diagnostic identities are retained locally. Public files contain aggregate metrics/figures and these reviewed paraphrases only.
 Codex assisted implementation and
testing; the author remains responsible for the experiment and interpretation.
