# Radiology reports in plain English

COMP3710 Project 2.6 — William Henry

This project compares frozen FLAN-T5-small, full fine-tuning and LoRA on
BioLaySumm LaymanRRG. The question is whether adaptation improves plain-English
reports without losing findings, negation or diagnostic uncertainty, and whether
the extra training cost is justified.

## Status

Implementation in progress. There are no trained-model results yet.
The initial split tests pass; the real-data audit and GPU benchmark are pending.
The feasibility review is in [FEASIBILITY.md](FEASIBILITY.md).

## Data

Dataset: [BioLaySumm2025-LaymanRRG-opensource-track](https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track).
The download is pinned to revision `6783ab38f9de28c4372799346f51d3927ae7eb9a`.
It contains 150,454 training rows and 10,000 validation rows. We do not use the
challenge test set: the assignment specifies held-out validation evaluation.

`dataset.py` removes missing pairs, groups identical normalised reports, removes
training reports/images shared with validation, and reserves part of validation
for final evaluation. Development and final evaluation share no report keys or
image paths. One example per distinct report prevents repeated boilerplate from
dominating the primary score. This changes the weighting of the supplied data;
we report it explicitly rather than presenting the result as a challenge score.
Conflicting reference rewrites are counted in the audit. These rules cannot
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

The local machine has an RTX 3050 Ti Laptop GPU with 4 GB VRAM. Local feasibility
is unverified. A short measured benchmark will decide whether to train locally
or move the same scripts to Colab. No Colab units have been used.

## References

- Raffel et al. (2020), [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://jmlr.org/papers/v21/20-074.html).
- Chung et al. (2022), [Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416).
- Hu et al. (2022), [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).
- [Google FLAN-T5-small model card](https://huggingface.co/google/flan-t5-small).
- [BioLaySumm 2025 task](https://biolaysumm.org/).
