# Post-training spectral study of the learned LoRA update

This bounded study preserves the original adapter, full-model baseline, report
and saved evaluations. It does not train or fine-tune a model. It compares the
original adapter, zero update, hard half/quarter rank, one soft half-rank update
and uniform0.5 update strength. Full-rank reconstruction is a numerical control,
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
See [Cai, Candes and Shen, Theorem2.1](https://arxiv.org/abs/0810.3286).

[AdaLoRA](https://arxiv.org/abs/2303.10512) allocates rank during training;
that is different from this post-training edit. [SVD-LLM](https://arxiv.org/abs/2403.07378)
motivates activation-aware error/whitening and compensation, which this simpler
matrix-only study does not implement. Spectral energy alone is not evidence of
clinical quality, and the zero/uniform controls help contextualise shrinkage.
The uniform0.5 control is not a precisely energy-matched control for every layer.

## Prespecified transformations and safeguards

Actual source:48 q/v projections, A8x512, B384x8, rank8, alpha16, scale2,
ordinary LoRA, bias none, no saved modules, no DoRA/rsLoRA. Each scaled dense
update is formed/SVD'd in float32 on CPU, once per layer. Only small derived
factors are retained. The source completion hash and every payload hash are
checked before and after derivation and equivalence tests.

- Hard half and quarter: retain top4/top2 singular directions per layer.
- Soft half: lambda is the midpoint of sigma4 and sigma5 independently per
  layer. Exact ties retain fewer than4 positive directions; no norm rescaling.
- Uniform strength: multiply B by0.5, retaining rank8 and A.
- Zero: zero B, retaining the same ordinary rank8 adapter structure.
- Full reconstruction control: retain all8 directions and compare logits.

Derived rank k uses alpha=2k, so PEFT still applies scale2. Factors represent
`delta_variant/2`; multiplying them by PEFT scaling once reconstructs the intended
update without double scaling. Each variant has a fresh immutable checkpoint,
complete provenance, weights/config checksums and a separate prediction output.
Existing outputs and source checkpoints are never overwritten. Unexpected bias,
saved modules, rank patterns, DoRA/rsLoRA or tensor schemas are refused.

Tests check known hard/soft optima, non-unit scaling, factor-gauge invariance,
threshold ties, real tiny-PEFT reload, zero-versus-disabled equivalence and source
immutability. Actual trained-adapter CPU logits are finite. Full reconstruction
has maximum absolute logit difference0.00006104 on four previously used dev cases
and identical argmax at the checked prefixes; zero versus disabled differs0.
This is a numerical sample check, not a global bitwise-equivalence guarantee.

All spectral comparison methods use **FP32**, greedy decoding,384/256 caps,
seed3710 and a common batch32 after a known-development memory pilot. FP32 reduces
factor-refactor BF16 rounding confounds. These scores are compared within this
matched study, not directly with the old BF16 official-validation results. No
sampling or clinical judge/teacher model is used. Ten-minute health observations
and checksummed batch recovery remain enabled.

## Development and confirmation

New seed3741 reservations exclude complete report/image components connected
to monitoring, previous pilots and every prior study/reservation. Draw192 random
group-balanced representative rows and128 untouched confirmation rows before
input-only challenge selection. Challenge32 comprises16 cue-rich and16 long
fully visible inputs, all within384 prompted tokens. Patient independence and
unrecorded earlier human inspection cannot be certified.

Compare all6 methods on192 representative and32 enriched challenge rows using
all four ROUGE scores, paired sampled-group intervals, runtime/memory, spectra
and source-grounded semantic cases. Review8 representative and4 challenge cases
selected with seed3742. Retained spectral energy is descriptive, not a confidence
score or factual-error label. Challenge enrichment does not estimate prevalence.

A rank-reducing candidate may be selected only if representative ROUGE-L is
within0.01 of original and the paired case review shows no clear added substantive
source-grounded regression. If several qualify, prefer lower physical rank, then
the stronger representative overlap. Freeze at most one candidate before using
the128 confirmation rows; compare only original and that candidate there.
Otherwise leave confirmation unused. This is an engineering screening rule,
not a clinical noninferiority claim or calibrated safety threshold.

Raw data, original/derived weights and bulk outputs remain outside Git. Public
updates contain code, aggregate metrics/figures and minimal reviewed paraphrases.
ChatGPT/Codex assists implementation, literature reading, testing and analysis;
the author remains responsible for the work. No clinician adjudication is claimed.

Results pending.
