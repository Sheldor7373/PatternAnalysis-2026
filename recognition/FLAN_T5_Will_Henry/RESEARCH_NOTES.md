# Follow-up research results

These extensions preserve the completed assignment baseline: its README,
official-validation predictions, selected checkpoints and report PDF remain
immutable comparisons. Development research results must not be presented as
fresh official-test confirmation. IDs beginning `train:` retain raw-dataset
origin; the research examples documented here belong to prepared DEV and were
excluded from generator gradient training.

| Question | Prespecified method and split | Measured finding and uncertainty | Runtime tradeoff | Decision |
|---|---|---|---|---|
| Does stronger decoding help? | [Greedy, beam 2/4, CAD alpha .25/.5](DECODING_EXPERIMENTS.md); full model, 192 representative + 48 enriched DEV groups. | Beam 2 gains .00186 representative RL, paired interval [-.00722,.01060]. CAD has no consistent gain; reviewed semantic changes are mixed. | Beam 2 uses 1.35x representative generation time; CAD roughly twice greedy. | Keep greedy; confirmation128 unused. |
| Are omissions caused by caps? | [384/768 input x 256/512 output factorial](DECODING_EXPERIMENTS.md); 128 representative + 32 enriched DEV groups, fixed greedy full model. | Output-only extension gains .01108 challenge RL [.00564,.01674] and recovers continuations, but factual errors persist. Input-only change loses .01824 [-.03573,-.00373], often with copying/repetition. | Output-only challenge generation takes 1.49x baseline. | Longer output is a reviewed recovery candidate, not a factuality fix; no blanket input-cap increase. |
| Can weak adapter directions be removed? | [Original/zero, hard half/quarter rank, soft half-rank, uniform .5 strength](SPECTRAL_STUDY.md); 192 representative + 32 fully visible enriched DEV groups; matched FP32. | **Half rank preserves 99.08% of update energy yet loses .03861 representative RL [-.05282,-.02494].** Fixed source-grounded cases show anatomy errors and omissions. Every rank-reducing candidate fails the prespecified screen. | All 13 supervised sessions, including pilot/loading/scoring, total 420.1s. Smaller adapter rank does not halve base-model cost. | Keep original adapter; confirmation128 unused. This negative result belongs in the final write-up. |
| How did performance develop during training? | [Frozen plus full/LoRA epoch 1/2/3](CHECKPOINT_LEARNING_STUDY.md); same existing dev64, matched FP32, likelihood and generation scored separately. | Full RL .64728/.65185/.66612; LoRA .50558/.51761/.53172. Later gains are smaller; per-case likelihood and semantic quality can diverge. Conditional paired intervals and fixed cases are documented. | Successful generation sessions 185.9s; timed reference passes 11.1s, excluding their loading. Recovery overhead is separately documented. | Keep original checkpoint selection; reference likelihood is not calibrated correctness confidence. |
| Does a modest extra epoch help? | [One separate full-model epoch](EXTRA_EPOCH.md), from verified final epoch3; peak 3e-5 then linear decay, batch 4/BF16, preserved Adam/RNG/cursor. Exactly 97,411 further examples. | **Running; results pending.** Compare final/latest with epoch3 on the same previously-used dev64; official validation remains outside tuning. | Sustained throughput and thermal state tracked in recoverable run logs. No wall-clock cutoff; finite exposure target enforced. | Await measured loss, generation quality and source-grounded cases before choosing a model. Baseline unchanged. |

The first four studies perform inference/CPU analysis without new training;
the fifth is the separately authorised training extension. No new teacher
examples or confidence-classifier fit is part of these studies. Negative results,
conditional uncertainty, source/reference limitations and runtime costs are
retained. The spectral method connects nuclear-norm soft thresholding to LASSO,
while distinguishing post-training edits from AdaLoRA and activation-aware error.

After the extra-epoch comparison finishes, consolidate all findings into the
eventual final write-up. Do not repeatedly regenerate the final PDF during an
active study or silently replace the immutable original assignment baseline.
