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

Results are pending. This file records the protocol before generation.
