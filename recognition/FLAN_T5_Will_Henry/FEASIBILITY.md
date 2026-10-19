# Feasibility review — radiology reports in plain English

**William Henry · COMP3710 · Project 2.6 · tutor consultation: 20 October 2026**

**User and purpose.** Help a patient understand an existing radiology report.
The prototype rewrites text for review; it does not interpret images or offer
diagnoses. The engineering question is whether fine-tuning improves readability
without dropping findings, reversing negation or overstating uncertainty.

**Acceptance criteria.**
1. Frozen, full and LoRA models run on the same held-out examples. Report
   ROUGE-1, ROUGE-2, ROUGE-L and ROUGE-Lsum, including sample counts and settings.
2. No exact normalised input report or image path overlaps training, development
   and final evaluation. Audit missing pairs, duplicates and conflicting targets;
   disclose that patient independence cannot be verified without patient IDs.
3. Fine-tuning gives a measurable development improvement over the frozen
   baseline. Reserve final evaluation for the selected settings. If improvement
   is absent, report it and investigate rather than claim success.
4. Audit 3–5 actual failures for hallucination, omitted findings, changed negation,
   uncertainty and readability. Give a concrete recommendation on human review.
5. Run a local smoke test within 4 GiB VRAM; measure peak allocated/reserved memory,
   parameter counts and generated tokens/s. If turnaround is impractical, assess
   Colab before spending; no paid compute has yet been authorised.

**Model, data and course concepts.** FLAN-T5-small (about 77M parameters), listed
T5/FLAN-T5 Hard difficulty. BioLaySumm2025-LaymanRRG open-source track: 150,454
training and 10,000 validation rows. Compare the frozen pretrained baseline with
full fine-tuning and rank-8 LoRA attention adapters. The method uses conditional
sequence modelling, teacher-forced cross-entropy, padding masks, regularisation
and held-out evaluation. Remove 42,249 validation-connected training rows, then
derive 97,411 training and 10,794 development examples from the remaining training
components. Reserve all 10,000 official validation rows for locked final evaluation,
retaining the original row weighting and reference rewrites.

**Initial evidence.** Pinned data/model assets are downloaded and hash-verified.
All 51 tests pass, including CPU model and resume tests. The raw data has no
missing report/target pairs. All 26 independent split checks pass, with no
cross-split ID, canonical report or available image overlap; patient independence
remains unverified. On the RTX 3050 Ti Laptop (4 GiB, working CUDA/BF16), full and
LoRA completed two optimiser steps each at batch 1, accumulation 2 and 384/256
source/target caps without OOM. Sampled whole-device peaks were 1,845/975 MiB;
all three modes generated on the same eight development IDs. These are bounded
feasibility checks, not final-quality results. Evidence: tested commit `28869a0`,
`runs/empirical-20261019/empirical-summary.json` and `data/prepared/` audits.

**Risks, budget and next step.** Main risks are repeated/overlapping reports,
limited metadata, truncating important findings, and fluent but incorrect outputs.
The token audit found 0.80% of training inputs and 1.83% of targets exceed the
provisional caps. Choose lengths and training scope using development data;
ROUGE scoring retains complete reference text. The 64-example timing pilots
naively scale to about 14 training-loop hours per model for three full-data epochs,
excluding validation, checkpointing, loading and sustained thermal effects.
Next: obtain tutor checkoff, agree settings and assess the training budget.
Fallback: retain the small checkpoint, reduce microbatch size and use gradient
accumulation/checkpointing; if its output remains inadequate, reassess model size
before further tuning. A negative result remains reportable but is not a promise
of full marks. Complete the required modular code, README, PR and review response.
