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
span/site/mask errors, finite arithmetic and callback cleanup. Real-model hooks,
fitting, inference and claims of causal abstraction remain deferred to the
parent's approved experiment protocol. The conceptual interchange-intervention
background is [Geiger et al. (2021)](https://proceedings.neurips.cc/paper/2021/hash/4f5c422f4d49a5a807eda27434231040-Abstract.html);
this toy does not reproduce that paper's experiments.
