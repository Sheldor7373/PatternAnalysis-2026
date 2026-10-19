# Radiology reports in plain English

COMP3710 Project 2.6 — William Henry

This project compares frozen FLAN-T5-small, full fine-tuning and LoRA on
BioLaySumm LaymanRRG. The question is whether adaptation improves plain-English
reports without losing findings, negation or diagnostic uncertainty, and whether
the extra training cost is justified.

## Status

Preparation checks pass: 51 tests, the real-data audit and bounded local GPU
pilots. These establish pipeline and memory feasibility, not final model quality.
The feasibility review is in [FEASIBILITY.md](FEASIBILITY.md).

## Data

Dataset: [BioLaySumm2025-LaymanRRG-opensource-track](https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track).
The download is pinned to revision `6783ab38f9de28c4372799346f51d3927ae7eb9a`.
It contains 150,454 training rows and 10,000 validation rows. We do not use the
challenge test set: the assignment specifies held-out validation evaluation.

The raw audit found no missing report/target pairs. `dataset.py` builds report/image
connected components and removes 42,249 validation-connected training rows.
The remaining rows form 97,411 training and 10,794 training-derived development
examples. All 10,000 official validation rows are reserved for locked final
evaluation, retaining the original row weighting and reference rewrites.
The independent verifier passes all 26 checks, with no cross-split ID, canonical
report or available image overlap. Conflicting reference rewrites are counted
in the audit. These rules cannot
establish patient independence because patient IDs are not supplied.

```bash
python dataset.py --output data/prepared
python -m unittest -v test_dataset.py
```

No reports, downloaded datasets, checkpoints or model weights belong in Git.

## Planned comparison

All three models will use the same reserved examples, prompt, truncation limits
and generation settings. The report will include ROUGE-1/2/L/Lsum, total and
trainable parameters, peak GPU memory, generation throughput, and 3–5 actual
failure cases. Development data selects settings; final evaluation does not.
We will distinguish measured results from estimates and flag small smoke tests.

The RTX 3050 Ti Laptop GPU has 4 GiB VRAM and working CUDA/BF16. Both full and
LoRA pilots completed two optimiser steps at batch 1, accumulation 2 and 384/256
source/target caps without OOM. Sampled whole-device peaks were 1,845/975 MiB
respectively; all three modes generated on the same eight development IDs.
These short pilots do not establish sustained throughput or model quality.
Evidence is stored locally under `runs/empirical-20261019/`, with data audits in
`data/prepared/`; neither directory is committed. No Colab units have been used.

## References

- Raffel et al. (2020), [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://jmlr.org/papers/v21/20-074.html).
- Chung et al. (2022), [Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416).
- Hu et al. (2022), [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).
- [Google FLAN-T5-small model card](https://huggingface.co/google/flan-t5-small).
- [BioLaySumm 2025 task](https://biolaysumm.org/).
