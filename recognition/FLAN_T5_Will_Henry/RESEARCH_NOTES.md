# Follow-up experiments

These studies extend the completed baseline without replacing its README,
official-validation predictions, checkpoints or report PDF. They use development
data for research; the inspected official validation set is never reused to
select extension settings. IDs starting `train:` describe original dataset origin;
the documented research samples belong to prepared DEV, outside gradient training.

| Study | Result and evidence |
|---|---|
| [Beam search and context-aware decoding](DECODING_EXPERIMENTS.md) | Beam 2 has a small uncertain overlap gain at extra cost; CAD gives no consistent improvement. Greedy remains default. |
| [Input/output cutoff diagnosis](DECODING_EXPERIMENTS.md) | A longer output cap recovers some continuations, but neither EOS nor longer text proves factual completion. Increasing input length can worsen repetition and omissions. |
| [Post-training LoRA spectral shrinkage](SPECTRAL_STUDY.md) | Half rank preserves 99.08% of update energy but loses representative overlap and adds source-grounded errors. Original adapter retained; confirmation reserve unused. |
| [Existing checkpoint learning curves](CHECKPOINT_LEARNING_STUDY.md) | Both methods improve mainly by epoch 1, with smaller subsequent gains. Reference likelihood, greedy outputs, readability and fidelity can diverge per case. |

The spectral study implements the nuclear-norm soft-thresholding connection to
LASSO, distinguishes post-training edits from AdaLoRA, and explains why
activation-aware error differs from unweighted matrix reconstruction. Negative
results are retained. All notes include scope, sample sizes, uncertainty,
resource costs, implementation checks and limitations. No new teacher examples,
classifier training or additional fine-tuning is part of these studies.
