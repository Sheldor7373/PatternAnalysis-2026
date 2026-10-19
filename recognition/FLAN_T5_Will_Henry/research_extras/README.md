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
