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

Training and comparison results are pending. Codex assisted implementation and
testing; the author remains responsible for the experiment and interpretation.
