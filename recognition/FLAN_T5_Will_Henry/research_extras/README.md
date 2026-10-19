# CPU research extras

## Synthetic behavioural families

```powershell
python -B behavioural_templates.py --output ../runs/behavioural-suite-v1
python -B -m unittest test_behavioural_templates -v
```

The suite follows the capability/test-type approach of Ribeiro et al. (2020),
[Beyond Accuracy: Behavioral Testing of NLP Models with CheckList](https://aclanthology.org/2020.acl-main.442/).
This is a small local adaptation, not the CheckList package or a reproduced result.
No model output has been generated or evaluated for these fixtures.

There are 40 authored template families: ten each for negation, uncertainty,
laterality and temporal change. Each family expands to three finding slots,
two semantic contrasts and two whitespace variants (480 cases total). Proposed
propositions support minimum-functionality review; whitespace pairs support
invariance review; semantic contrasts support directional meaning-change review.
Compare meanings rather than exact output strings. These relations do not claim
that a particular token must become more or less probable.

The explicit family split is 28 development / 12 heldout, balanced 7/3 within each
capability (336/144 cases). All finding slots, contrasts and invariance variants
stay with their parent family. No canonical full source text crosses partitions.
Related constructions and the same concepts still occur across partitions;
withholding families does not certify independence or absence from pretraining.
The heldout partition is a synthetic research holdout, separate from the official
10,000-row final validation set. Do not tune prompts or thresholds on heldout
cases, then report them as untouched. Lock the suite hash before later inference.

`families.jsonl` exposes both templates and proposed labels; `development.jsonl`
and `heldout.jsonl` contain concrete examples and relation IDs. `human-review.csv`
contains each source, proposed proposition and blank approval/reviewer/notes.
`manifest.json` records counts, limitations and source fingerprints. Files are
deterministic, and an existing suite directory cannot be overwritten.

Labels describe assertions in the synthetic text, not clinical ground truth:
`absent` means the report negates a finding, `uncertain` retains diagnostic
uncertainty, and temporal labels distinguish previous/current statements.
Scope controls include secondary propositions. All labels are author-proposed
and require human review for wording, clinical plausibility and correctness.
This is a suite preparation result, not a measured model failure rate. The
project's limited regex flags cannot validate all these capabilities.

## Saved-array linear CKA (no inference)

`representation_analysis.py` compares already pooled numeric representations.
It loads no model and performs no tokenisation, capture, fitting or PCA. Based on
[Kornblith et al. (2019)](https://proceedings.mlr.press/v97/kornblith19a.html), it
centres feature columns across the same ordered reports and computes biased
linear CKA: `||Xc.T Yc||_F^2 / (||Xc.T Xc||_F * ||Yc.T Yc||_F)`, using the
equivalent Gram form. Positive isotropic rescaling keeps extreme finite magnitudes
numerically manageable. CKA describes similarity, not quality or causation.

```powershell
python -B -m unittest test_representation_analysis -v
python -B representation_analysis.py --left left.npz --right right.npz --output ../runs/cka-review.json
```

Each small NPZ must contain exactly `activations` (real `[reports, features]`),
`report_ids` (unique nonempty Unicode strings in identical order), and `metadata`
(scalar Unicode JSON text). Objects/pickle are refused. Maximums: 256 reports,
2,048 features per model, 8 MB on disk and expanded. Feature widths may differ.
IDs are never silently reordered, intersected or deduplicated. At least three
reports are needed; fewer reports or zero centred variance produce null CKA and
an explicit unavailable reason. Nonfinite/malformed arrays raise errors. No
sample interval is estimated; small panels can inflate descriptive similarity.

Required matching metadata: `data_split` (`synthetic`, `train` or `dev` only),
`source_manifest_sha256`, `tokenizer_revision`, `prompt_sha256`, `source_length`,
`pooling`, `site`, and ordered `visible_tokens_sha256` (one lowercase SHA256 per
report, over exact visible token IDs and mask). Required version fields are
`model_label`, `model_revision`, `checkpoint_sha256`; model versions may differ
and are retained. `pooling` must describe content/prompt/special/padding treatment.
The module verifies consistency of recorded metadata, not upstream extraction or
weight provenance. Tests provide the complete synthetic schema. Source files
are fingerprinted/read-only and existing output files cannot be replaced.
No official validation arrays are accepted. Shared PCA is deliberately deferred.

## Projection-swap toy only

```powershell
python -B -m unittest test_toy_interventions -v
python -B toy_interventions.py --output ../runs/toy-intervention.json
```

`toy_interventions.py` accepts token-by-feature arrays and an explicitly supplied
rank-k orthonormal column basis `Q`; it computes `base + (source - base) @ Q @ Q.T`
on declared equal-length half-open token spans. It preserves the orthogonal
complement and all positions outside the base span. Named base/source sites must
match exactly; width/rank, bounds, Boolean token masks and orthonormality are
validated. Inputs are copied. The module never fits a basis or infers alignment.
It has no model/version information and is **not** an adapter for real activations.

The deterministic toy has a rotated causal coordinate, an orthogonal nuisance
coordinate and a known linear score. Patching the causal coordinate changes the
score by -9; the nuisance control changes it by zero within floating-point error.
These are constructed arithmetic checks, not model effects. `ToySite` is a local
callback registry with cleanup on normal/exceptional exit; it cannot register
real framework hooks. Tests cover rank-k algebra, identity patches, known effect,
span/site/mask errors, finite arithmetic and callback cleanup. Pretrained-model
execution, fitting and claims of causal abstraction remain deferred to the
documented experiment protocol. The separate adapter below is exercised
only with a tiny randomly initialized T5. The conceptual interchange-intervention
background is [Geiger et al. (2021)](https://proceedings.neurips.cc/paper/2021/hash/4f5c422f4d49a5a807eda27434231040-Abstract.html);
this toy does not reproduce that paper's experiments.

## Fixed-basis CPU T5 span-mean adapter

`t5_intervention_adapter.py` takes an existing plain `T5ForConditionalGeneration`
already frozen and fully in eval mode on CPU. It never loads weights, moves
devices, calls `generate`, changes model state, or fits a basis. Initially it
supports one donor/recipient example per call, float32/float64 hidden states,
and supplied orthonormal `Q[d,k]` with matching dtype. PEFT wrappers and other
architectures are outside this initial interface.

```powershell
python -B -m unittest test_t5_intervention_adapter -v
```

Construct `SourceTokens(input_ids, attention_mask, eligible_mask)` using CPU
`int64` IDs shaped `[1,T]` and explicit Boolean masks of the same shape. Spans
are nonempty half-open `(start, stop)` pairs; all span tokens must be eligible
and attended. The caller defines eligibility for content/prompt/special tokens;
the adapter never infers it from token IDs or character offsets. Source and
candidate lengths are limited to 64 tokens for restrained CPU tests.

`T5SpanMeanAdapter(model, "encoder.block.0", Q)` accepts only existing named
encoder-block output sites. `encode_pair(recipient, donor, recipient_span=Sr,
donor_span=Sd)` captures the donor and returns baseline/patched recipient encoder
outputs. At the specified block it applies, in row-vector notation:

`delta = (mean(H_donor[Sd]) - mean(H_recipient[Sr])) @ Q @ Q.T`

`H_patched[t] = H_recipient[t] + delta` for every `t` in `Sr`.

Donor/recipient spans may have different lengths. Within-span token differences,
the orthogonal complement and positions outside the recipient span are preserved
up to floating-point tolerance. Activations are copied; T5 block tuple fields
after the hidden-state tensor are retained. Each site must execute exactly once.
Only adapter-owned hook handles are removed in `finally`; unrelated hooks remain.
Encoder hooks are removed before teacher-forced decoder scoring.

`Candidate(input_ids, target_mask, eos_included)` has an explicit nonempty valid
prefix/right-padding mask independent of token ID. `eos_included` must match
one final valid EOS or no valid EOS; nothing is appended or dropped silently.
Masked target IDs do not contribute to the score. Decoder masks explicitly keep
the start position visible even when T5's decoder-start ID equals its pad ID.

`adapter.compare(recipient, donor, candidate_a, candidate_b, recipient_span=Sr,
donor_span=Sd)` returns baseline/patched A/B **summed** log probabilities, token
counts (including EOS iff declared), EOS inclusion, A-minus-B margins and the
patched-minus-baseline margin effect. There is no length normalisation. Each
recipient encoder output is reused across candidates. Log-softmax/gather/sums
are evaluated in float64; these are model scores, not calibrated confidence.

Tests use synthetic arrays and a seeded random T5 (vocabulary 32, hidden width
16, two encoder blocks, one decoder block), with no pretrained assets. Float32
self-patch logits use `atol=rtol=1e-6`; margin effects use absolute tolerance
`1e-6`. Parameters are checked unchanged for self and non-self interventions.
Tests also cover unequal-span algebra, masked padding with arbitrary token IDs,
manual log-probability sums, explicit EOS, validation and hook cleanup after
capture/patch/decoder exceptions while preserving unrelated observers.

`torch.inference_mode()` is appropriate for this **fixed-basis inference** adapter
only. Future subspace fitting requires a separately reviewed differentiable
path; this module does not train `Q`. Pretrained execution, GPU use, basis
fitting, clinical labels and final evaluation remain outside this module's test
evidence and await the documented experiment protocol.
