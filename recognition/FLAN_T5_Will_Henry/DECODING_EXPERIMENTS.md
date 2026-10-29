# Development decoding and cutoff investigations

This is a separate extension to the completed 29 October baseline. Its original
README, PDF, 30,000 predictions and selected checkpoints remain unchanged.
No retraining, teacher model, API or paid compute is involved.

## Design fixed before the new outputs

First compare the selected full-tuned model with greedy, beam 2, beam 4 and greedy
context-aware decoding (CAD alpha 0.25/0.5). Then investigate source/output caps.
The implementation follows [Shi et al., 2024, Section 2.2](https://aclanthology.org/2024.naacl-short.69/)
and was checked against their [official implementation](https://github.com/xhan77/context-aware-decoding).
For every decoder prefix, CAD uses the same model twice:

`adjusted_logits = (1 + alpha) * logits(instruction + report, prefix) - alpha * logits(instruction, prefix)`.

Both streams share the generated prefix and keep separate encoder/KV state.
The instruction-only stream contains no report. Contrast arithmetic is float32
under BF16 model autocast. Alpha 0 bypasses the extra pass and exactly follows
the ordinary generation path. CAD is restricted to beam 1; ordinary beam search
uses Transformers generation with length penalty 1 and early_stopping false.
This greedy CAD adaptation is not a reproduction of the paper's nucleus-sampling
summarisation protocol and does not inherit its factuality results.

`predict.py` binds the decoder, alpha, instruction-only prior, beams, length
penalty, stopping defaults, source/output caps and decoder module hash into its
durable resume identity. Changing these requires a fresh output directory.
The original producer remains available at commit `860d28f`; original artifacts
are preserved rather than relabelled as produced by the extended code.

## New development reservations

`prepare_decoding_study.py` excludes complete transitive report/image components
connected to the fixed dev 256 monitor or any supplied historical prediction/
review manifest, including dev 64. The actual inventory excluded 316 components,
leaving 8,820 of 9,136 development components. It reserves disjoint groups before
any new generation: 192 representative decoding rows, 48 input-enriched challenge
rows, 128 unused confirmation rows, and separate 128 representative/32 challenge
rows for cutoff work. Representative rows are sampled before enrichment: one
random row per shuffled component, seed 3729. This is group-balanced rather than
the original row-weighted distribution. The 48 challenge rows all come from
BIMCV-COVID19; domain and difficulty are therefore confounded.

Historical exclusions are as complete as the recorded manifests. The broader
development split already influenced checkpoint selection through its monitor.
The official 10,000 validation rows have been inspected and are not a fresh test.
No patient IDs exist. Reservations and all raw clinical text remain local.

## Interpretation and checks

Read ROUGE-1/2/L/Lsum alongside paired source-grounded factuality, omissions,
uncertainty and qualitative readability. Regex flags are review candidates,
not measured clinical-error prevalence. Record generation time, output lengths,
EOS/repetition, memory and thermal conditions. Beam search optimises sequence
likelihood under its scoring rule; it does not enforce factual truth. A promising
candidate may be compared with greedy on the reserved 128 only after selection;
a negative result need not consume that reserve.

Cutoff work follows the decoding report, using its own reserved groups. Compare
384/256 with 768/256,384/512 and 768/512. Separate information outside the visible
input from omitted visible content, and a missing EOS token from demonstrated
semantic/sentence truncation. Longer caps can add cost or unsupported material;
they are not automatically a recovery. Any inspection of old official-validation
cutoff cases is exploratory, not independent confirmation.

Offline tests cover the normalized CAD probability ratio, invalid settings,
alpha 0/native equivalence, real beam 2/4 paths, cached versus full-prefix logits,
controlled mixed early-EOS batches, resume equality and changed-knob refusal.
An independent read-only implementation review found no blocking issue. All 151 CPU regressions passed after the token-counter correction. The independent results audit reproduced all 40 ROUGE means, all 32 paired intervals and source-grounded case claims. New
strategy memory pilots must pass before the bounded development runs. Ten-minute
process/GPU health observations and checksum-published batches remain enabled.

A memory pilot exposed a Transformers 4.57 beam-tail accounting issue: zero-pad
finished outputs can be filled with repeated EOS. The extended counter stops at
the first EOS (inclusive), rather than counting filled tails. Original pilot
artifacts are retained as superseded; corrected experiments use fresh output
folders and identities. Their decoded content was unaffected. Reported tokens/s
counts the returned sequence's content/EOS, not every explored beam candidate.

AI assistance: ChatGPT/Codex helped read the paper, implement the extension,
prepare group reservations, write tests and inspect saved outputs. Independent
AI-assisted read-only review checked the formula, caches, EOS, identities and
sampling design. The author remains responsible for verifying and defending
these choices. No clinical reviewer or patient-comprehension study is claimed.

## Option 3 measured results (development only)

All 5 strategies completed 192 representative and 48 enriched challenge rows at
batch 16/BF16,384/256 caps. All 1,200 outputs and published batches passed identity,
order and checksum checks. A corrected pilot exercised output 256 and retained
at least 2,869 MiB sampled headroom. The largest representative beam 4 sample used
1,457 MiB. Counts below are returned-output EOS flags, not semantic completion. Raw generated token IDs were not retained: the first-EOS counter was checked with controlled tests and exported totals were reconciled, but exact EOS positions cannot be independently reconstructed from decoded text.


### Representative sample

| Decoder | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum | Generation seconds | No EOS |
|---|---:|---:|---:|---:|---:|---:|
| greedy | 0.71723 | 0.52367 | 0.65879 | 0.68772 | 42.85 | 3 |
| beam 2 | 0.71760 | 0.52281 | 0.66065 | 0.68777 | 57.80 | 2 |
| beam 4 | 0.71323 | 0.51466 | 0.65552 | 0.68243 | 58.67 | 2 |
| CAD 0.25 | 0.72104 | 0.51691 | 0.65631 | 0.68619 | 83.56 | 1 |
| CAD 0.5 | 0.70753 | 0.50070 | 0.64451 | 0.67165 | 85.97 | 3 |

| Difference from greedy | ROUGE-L difference | 95% paired interval | Observed generation time / greedy |
|---|---:|---|---:|
| beam 2 | +0.00186 | [-0.00722, +0.01060] | 1.35 x |
| beam 4 | -0.00326 | [-0.01421, +0.00717] | 1.37 x |
| CAD 0.25 | -0.00247 | [-0.01347, +0.00839] | 1.95 x |
| CAD 0.5 | -0.01427 | [-0.02751, -0.00122] | 2.01 x |

### Challenge sample

| Decoder | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum | Generation seconds | No EOS |
|---|---:|---:|---:|---:|---:|---:|
| greedy | 0.58860 | 0.35725 | 0.46376 | 0.57287 | 16.96 | 39 |
| beam 2 | 0.57488 | 0.34902 | 0.45550 | 0.55889 | 21.08 | 41 |
| beam 4 | 0.55629 | 0.33059 | 0.44027 | 0.54235 | 21.11 | 42 |
| CAD 0.25 | 0.55102 | 0.32729 | 0.44009 | 0.53762 | 27.30 | 41 |
| CAD 0.5 | 0.54680 | 0.31873 | 0.43612 | 0.53270 | 27.50 | 36 |

| Difference from greedy | ROUGE-L difference | 95% paired interval | Observed generation time / greedy |
|---|---:|---|---:|
| beam 2 | -0.00826 | [-0.02319, +0.00534] | 1.24 x |
| beam 4 | -0.02349 | [-0.04390, -0.00555] | 1.24 x |
| CAD 0.25 | -0.02367 | [-0.04382, -0.00566] | 1.61 x |
| CAD 0.5 | -0.02764 | [-0.04807, -0.00616] | 1.62 x |

Intervals use 1,000 paired sampled-group bootstrap draws, seed 3731. Because
there is one sampled row per source/image component, resampling units are those
sampled groups. They describe this development comparison, not independent
clinical performance, patient confidence or training-seed uncertainty.35/48
challenge inputs truncate at 384;0/192 representative inputs truncate. Source
length, domain and output budget jointly affect challenge results. Empty outputs
are retained. Timing includes first-batch generation but excludes model loading,
scoring and observer overhead. Sequential order, output lengths and thermals
limit controlled speedup claims. BF16 beam shapes also change numerical paths.

![Paired development differences and observed cost](figures/decoding-dev-comparison.png)

### Source-grounded findings

The predeclared textual review used 8 representative and 4 challenge cases, with
method names hidden during the first pass. This AI-assisted review is not
radiologist adjudication and does not estimate clinical-error prevalence.
Original/visible sources, references, all 5 outputs, initial notes and label keys
are retained privately. Two further overlap-divergence candidates were generated
for possible review; they are not counted as reviewed cases here.

- `train:136784`: CAD retained the source's suggestive diagnostic wording;
  greedy added severity and stronger assertion. This is a local repair, not a
  general factuality guarantee.
- `train:140274`: beam 4 preserved the upper-lung qualifier. Other outputs added
  lower-lung involvement or grammatically attached findings to the reporting
  clinician. More fluent text can still misbind attributes and entities.
- `train:121892`: greedy retained growth measurements but reversed their
  direction. Beam 2 introduced contradictory and misbound findings; CAD0.25
  retained the major visible growth/uncertainty information but remained
  technical and unfinished. Important facts were visible, so these failures
  cannot all be attributed to input truncation.
- `train:131366`: greedy and beam 2 reversed a visible negative organ finding;
  other variants retained it. Some outputs contained EOS while ending with an
  unfinished phrase. Literal EOS is not proof of semantic completion.
- `train:141676`: every decoder repeated a finding and omitted later visible
  content. Beam 2/4 additionally reversed a technique qualifier. The full input
  was visible: this is a repetition-mediated budget failure, not lost input.
- `train:30622`: every decoder echoed the reference's more specific wording
  although the source supplied a broader pattern description. A reference is
  an imperfect proxy for source-grounded factuality; no clinical adjudication
  established equivalence.

Other reviewed cases showed grammar-only differences, retained jargon and
malformed dates rather than a stable readability winner. On aggregate, beam 2's
small representative overlap gain is uncertain and costs 1.35 x observed generation
time; CAD and larger beams did not dominate on the source-grounded review.
Keep greedy as the default for this prototype. Leave the 128-row confirmation
reserve unused rather than declaring a winner from ROUGE alone. CAD alpha is an
inference contrast weight, not a calibrated confidence score. No new inference
method makes the baseline suitable for unreviewed patient use.

Beam search approximately searches high-probability sequences; wider search
costs more and does not optimise clinical fidelity. See Freitag and Al-Onaizan
(2017), [Beam Search Strategies for Neural Machine Translation](https://aclanthology.org/W17-3207/).
CAD's theory motivates this experiment, but its paper's general summarisation
results do not establish benefits for a domain-tuned small radiology model.

### Reproducing the extension

Use the dependencies and pinned data/model preparation in the main README.
Generate the historical dev 64 with `prepare_generation_pilot.py`, then pass it
and any additional recorded development prediction manifests via repeated
`--seen` arguments to `prepare_decoding_study.py`. Its default monitor exclusion,
seed and sample sizes reproduce this study's group reservation when the same
historical records are supplied. Do not substitute official validation for dev.
For example, after setting `$Checkpoint` to the immutable selected full model:

```powershell
.\.venv\Scripts\python.exe prepare_decoding_study.py --development data/prepared/dev.jsonl --seen runs/known/dev64.jsonl --output runs/extension/reserved
.\.venv\Scripts\python.exe predict.py --manifest runs/extension/reserved/decoding-representative.jsonl --checkpoint $Checkpoint --output runs/extension/cad025 --decoder cad --cad-alpha 0.25 --batch-size 16 --beams 1 --source-length 384 --target-length 256 --cpu-threads 4 --precision bf16 --device cuda --evaluation-role development-pilot
```

For beams use `--decoder standard --cad-alpha 0 --beams 2` or 4 and fresh output
paths. Resume only with identical identity settings. The extension producer corresponds to commit `f95b3fb`. Original baseline
artifacts are preserved and correspond to their original producer commit, not
this newer code. The original README/PDF are not rewritten by this extension.

## Option 1 cutoff investigation

Prepared separately; not yet complete. Its four settings and reserved samples
are fixed above. Findings will be added after generation and source review.
