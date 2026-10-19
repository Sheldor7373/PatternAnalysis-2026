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
5. Run a local smoke test within 4 GB VRAM; measure peak allocated/reserved memory,
   parameter counts and generated tokens/s. If impractical, benchmark Colab and
   reassess before more than 100 CU; no paid compute has yet been authorised.

**Model, data and course concepts.** FLAN-T5-small (about 77M parameters), listed
T5/FLAN-T5 Hard difficulty. BioLaySumm2025-LaymanRRG open-source track: 150,454
training and 10,000 validation rows. Compare the frozen pretrained baseline with
full fine-tuning and rank-8 LoRA attention adapters. The method uses conditional
sequence modelling, teacher-forced cross-entropy, padding masks, regularisation
and held-out evaluation. Split official validation into development and reserved
evaluation by report group; deduplicate boilerplate and report the changed weighting.

**Initial evidence.** The dataset is publicly downloadable (about 29 MB total).
Five offline tests pass for duplicate/image leakage, reproducible sampling and
missing targets. XPS GPU identified: RTX 3050 Ti Laptop, 4 GB VRAM. Real-data audit,
working CUDA environment and training throughput are pending. There are no
measured model scores or training-cost claims yet.

**Risks, budget and next step.** Main risks are repeated/overlapping reports,
limited metadata, truncating important findings, and fluent but incorrect outputs.
Next: download the pinned data, complete the audit, inspect token lengths, then
benchmark frozen inference and a short full/LoRA training run locally. A provisional
Colab planning range is 50–150 CU, to be replaced by measurements before spending.
Fallback: retain the small checkpoint, reduce microbatch size and use gradient
accumulation/checkpointing; if its output remains inadequate, reassess model size
before further tuning. A negative result remains reportable but is not a promise
of full marks. Complete the required modular code, README, PR and review response.
