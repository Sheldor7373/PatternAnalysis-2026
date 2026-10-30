# Post-training spectral study of the learned LoRA update

This bounded study preserves the original adapter, full-model baseline, report
and saved evaluations. It does not train or fine-tune a model. It compares the
original adapter, zero update, hard half/quarter rank, one soft half-rank update
and uniform 0.5 update strength. Full-rank reconstruction is a numerical control,
not a seventh tuned decoding method. Activation-aware whitening is deferred.

## Objective and literature

For every adapted query/value projection, decompose the actual scaled update
`delta=(alpha/r)*B@A`, not its basis-dependent factors individually. Hard top-k
SVD is optimal for unweighted Frobenius matrix reconstruction at rank k; it is
not automatically optimal for activations, sequence likelihood or factuality.

Soft singular-value thresholding solves

`min_Z 0.5*||Z-delta||_F^2 + lambda*||Z||_*`,

where the nuclear norm is the sum of singular values. The solution replaces
each singular value with `max(sigma-lambda,0)`. This parallels scalar L1 soft
thresholding/LASSO, but induces low matrix rank rather than entrywise sparsity
in a fixed coefficient basis. It is not supervised LASSO on radiology labels.
See [Cai, Candes and Shen, Theorem 2.1](https://arxiv.org/abs/0810.3286).

[AdaLoRA](https://arxiv.org/abs/2303.10512) allocates rank during training;
that is different from this post-training edit. [SVD-LLM](https://arxiv.org/abs/2403.07378)
motivates activation-aware error/whitening and compensation, which this simpler
matrix-only study does not implement. Spectral energy alone is not evidence of
clinical quality, and the zero/uniform controls help contextualise shrinkage.
The uniform 0.5 control is not a precisely energy-matched control for every layer.

## Prespecified transformations and safeguards

Actual source: 48 q/v projections, A 8x512, B 384x8, rank 8, alpha 16, scale 2,
ordinary LoRA, bias none, no saved modules, no DoRA/rsLoRA. Each scaled dense
update is formed/SVD'd in float32 on CPU, once per layer. Only small derived
factors are retained. The source completion hash and every payload hash are
checked before and after derivation and equivalence tests.

- Hard half and quarter: retain top 4/top 2 singular directions per layer.
- Soft half: lambda is the midpoint of sigma4 and sigma5 independently per
  layer. Exact ties retain fewer than 4 positive directions; no norm rescaling.
- Uniform strength: multiply B by 0.5, retaining rank 8 and A.
- Zero: zero B, retaining the same ordinary rank 8 adapter structure.
- Full reconstruction control: retain all 8 directions and compare logits.

Derived rank k uses alpha=2k, so PEFT still applies scale 2. Factors represent
`delta_variant/2`; multiplying them by PEFT scaling once reconstructs the intended
update without double scaling. Each variant has a fresh immutable checkpoint,
complete provenance, weights/config checksums and a separate prediction output.
Existing outputs and source checkpoints are never overwritten. Unexpected bias,
saved modules, rank patterns, DoRA/rsLoRA or tensor schemas are refused.

Tests check known hard/soft optima, non-unit scaling, factor-gauge invariance,
threshold ties, real tiny-PEFT reload, zero-versus-disabled equivalence and source
immutability. Actual trained-adapter CPU logits are finite. Full reconstruction
has maximum absolute logit difference 0.00006104 on four previously used dev cases
and identical argmax at the checked prefixes; zero versus disabled differs 0.
This is a numerical sample check, not a global bitwise-equivalence guarantee.

All spectral comparison methods use **FP32**, greedy decoding, 384/256 caps,
seed 3710 and a common batch 32 after a known-development memory pilot. FP32 reduces
factor-refactor BF16 rounding confounds. These scores are compared within this
matched study, not directly with the old BF16 official-validation results. No
sampling or clinical judge/teacher model is used. Ten-minute health observations
and checksummed batch recovery remain enabled.

## Development and confirmation

New seed 3741 reservations exclude complete report/image components connected
to monitoring, previous pilots and every prior study/reservation. Draw 192 random
group-balanced representative rows and 128 untouched confirmation rows before
input-only challenge selection. Challenge 32 comprises 16 cue-rich and 16 long
fully visible inputs, all within 384 prompted tokens. Patient independence and
unrecorded earlier human inspection cannot be certified.

Compare all 6 methods on 192 representative and 32 enriched challenge rows using
all four ROUGE scores, paired sampled-group intervals, runtime/memory, spectra
and source-grounded semantic cases. Review 8 representative and 4 challenge cases
selected with seed 3742. Retained spectral energy is descriptive, not a confidence
score or factual-error label. Challenge enrichment does not estimate prevalence.

A rank-reducing candidate may be selected only if representative ROUGE-L is
within 0.01 of original and the paired case review shows no clear added substantive
source-grounded regression. If several qualify, prefer lower physical rank, then
the stronger representative overlap. Freeze at most one candidate before using
the 128 confirmation rows; compare only original and that candidate there.
Otherwise leave confirmation unused. This is an engineering screening rule,
not a clinical noninferiority claim or calibrated safety threshold.

Raw data, original/derived weights and bulk outputs remain outside Git. Public
updates contain code, aggregate metrics/figures and minimal reviewed paraphrases.
ChatGPT/Codex assists implementation, literature reading, testing and analysis;
the author remains responsible for the work. No clinician adjudication is claimed.

## Measured results

All 1,344 comparison outputs (6 methods x 224 groups) and the 64-row pilot completed with matched IDs, checked batch hashes, no empty outputs and unchanged source hashes. The 128 confirmation groups were never generated. All four ROUGE values below are mean F1 on [0,1], with stemming and explicit sentence splitting for Lsum. Each sampled row represents a different report/image component.

### Representative: 192 groups

| Method | R1 | R2 | RL | RLsum | Generation seconds | No EOS | Repetition flag |
|---|---:|---:|---:|---:|---:|---:|---:|
| original | 0.58955 | 0.35414 | 0.51643 | 0.55489 | 29.38 | 8 | 7 |
| zero | 0.28254 | 0.11260 | 0.24630 | 0.26020 | 20.00 | 2 | 2 |
| hard_half | 0.56160 | 0.31167 | 0.47782 | 0.52060 | 29.18 | 3 | 2 |
| hard_quarter | 0.46749 | 0.22193 | 0.39080 | 0.43239 | 30.89 | 9 | 15 |
| soft_half | 0.46143 | 0.21046 | 0.38949 | 0.42369 | 24.39 | 8 | 12 |
| scale_half | 0.42757 | 0.18542 | 0.37144 | 0.39608 | 23.02 | 2 | 2 |

### Enriched challenge: 32 fully visible source groups

| Method | R1 | R2 | RL | RLsum | Generation seconds | No EOS | Repetition flag |
|---|---:|---:|---:|---:|---:|---:|---:|
| original | 0.42963 | 0.19259 | 0.32774 | 0.41005 | 9.48 | 26 | 15 |
| zero | 0.09306 | 0.04257 | 0.08153 | 0.08978 | 7.45 | 1 | 1 |
| hard_half | 0.42867 | 0.19438 | 0.32000 | 0.41047 | 6.62 | 23 | 11 |
| hard_quarter | 0.40858 | 0.18444 | 0.33091 | 0.39552 | 5.95 | 27 | 13 |
| soft_half | 0.38281 | 0.17717 | 0.31537 | 0.36989 | 6.46 | 23 | 6 |
| scale_half | 0.41545 | 0.19546 | 0.35058 | 0.40423 | 5.60 | 24 | 1 |

### Paired changes and the decision

Representative ROUGE-L changes versus original, with 95% percentile intervals from 1,000 shared paired sampled-group bootstrap draws (seed 3743):

- zero: -0.27014 [-0.29356, -0.24645].
- hard_half: -0.03861 [-0.05282, -0.02494].
- hard_quarter: -0.12563 [-0.14621, -0.10671].
- soft_half: -0.12695 [-0.14586, -0.10911].
- scale_half: -0.14499 [-0.16483, -0.12550].

Hard half retains **99.083%** of scaled-update energy yet loses 0.03861 representative ROUGE-L. Hard quarter retains 95.248%; soft half retains 84.517%; uniform half strength retains 25%. These are globally weighted squared-Frobenius fractions, not percentages of factual information. No rank-reducing candidate is within the prespecified 0.01 tolerance, so **retain the original adapter and leave confirmation unused**. This negative result does not imply every rank reduction must fail, only that these untrained matrix-only edits fail this screen.

On enriched challenges, uniform weakening has the highest overlap but its paired change +0.02285 has interval [-0.02197, +0.07305]. It often copies source terminology and is not a representative-set winner. Reduced repetition flags can result from copying or shorter outputs; the flags are not clinical error rates. Challenge selection enriches long/cue-rich inputs, so its rates are not population prevalence.

![Spectral energy and matched development overlap](figures/spectral-development-comparison.png)

### Source-grounded fixed-case review

All six outputs were reviewed on the preselected eight representative and four challenge cases. The local private audit records all 12; these minimal paraphrases illustrate findings without publishing bulk reports.

| Case | Grounded comparison |
|---|---|
| train:132218 | Original explains a raised diaphragm; half-rank changes the structure to chest wall, and quarter-rank changes direction/anatomy. Soft/uniform preserve technical phrasing but lose explanatory simplification. |
| train:18226 | Half-rank and soft shrinkage omit visible calcified chest lymph nodes. Quarter-rank adds unsupported collapse/fluid. Uniform weakening mainly copies the source. |
| train:140866 | Both original and half-rank change an atypical pattern to typical. Compression does not repair this baseline error; quarter-rank also loses laterality/stability. |
| train:141336 | Original and half-rank loop. Quarter-rank retains more late findings and terminates, but soft shrinkage also terminates while contradicting source stability. EOS does not certify correctness. |
| train:136083 | Original loops on a fabricated sampling phrase. Other variants sometimes recover later text but introduce anatomy errors, leave jargon, or exhaust the cap before key lesion details. |

All challenge inputs fit 384 prompted tokens, so their omissions cannot automatically be blamed on hidden source text. Output cap 256 remains a confound for long references. One reference expands an acronym beyond explicit source wording; agreement with that reference is not independent clinical adjudication.

### Costs, reproducibility and limits

Observed supervised sessions total 420.1 seconds; peak sampled device usage is 1369 MiB and minimum sampled headroom 2727 MiB. Each session includes load/scoring overhead. Generation timings above are a single ordered run with different output lengths, so they are not a precise speed benchmark. Rank 8 has 344,064 adapter parameters, rank 4 has 172,032 and rank 2 has 86,016, against 76,961,152 base parameters. Halving the adapter does not halve total memory or latency. No OOM or reported observer fault occurred.

Local receipts bind the original adapter completion hash `2f5800e64b514c0b2ac4c35ea04422cf20069d9516b3e7adff408f92f1b02053`, builder source hash, derived immutable checkpoints, manifest hashes, runtime and prediction producer. Selection manifest hashes: representative `7124e96127228a6a24c87342e645379229de02dbb0c727cf2c575ea3d2f9fa49`; challenge `f9e836249161e66778bee7029536dd9f86080159f0e5725eb303c0c1c340052a`. Exact prediction token IDs were not saved; generated-token/EOS counters use the tested first-EOS rule. Development informed this study and its checkpoint selection; foundation pretraining exposure and patient independence are unknown. The FP32 study is internally matched and does not replace the old BF16 official results.
