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

## Option 1: input/output cutoff investigation

After the option 3 readout, the same selected full model completed the fixed
2-by-2 cap comparison on separate 128 representative and 32 challenge groups:
640 additional outputs, greedy decoding, batch 16 and BF16. The source cap
includes the instruction. These remain development investigations, not a fresh
confirmation on the already-inspected official validation set. No further GPU
work or training followed this study; the 128-row confirmation reserve remains
unused. Raw outputs and diagnostic packets remain local.

### Representative cap comparison

| Source/output cap | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum | Truncated inputs | No EOS | Repetition candidates |
|---|---:|---:|---:|---:|---:|---:|---:|
| 384/256 | 0.72059 | 0.53151 | 0.66608 | 0.69404 | 1 | 3 | 0 |
| 768/256 | 0.72099 | 0.53139 | 0.66591 | 0.69429 | 0 | 3 | 0 |
| 384/512 | 0.72083 | 0.53182 | 0.66628 | 0.69424 | 1 | 0 | 0 |
| 768/512 | 0.72121 | 0.53176 | 0.66629 | 0.69455 | 0 | 0 | 0 |

| Difference from 384/256 | ROUGE-L difference | 95% paired interval | Generation seconds | Returned tokens/s | Allocated / sampled-device MiB |
|---|---:|---|---:|---:|---:|
| 384/256 | Baseline | - | 26.56 | 314.10 | 572.97 / 1019 |
| 768/256 | -0.00018 | [-0.00120, +0.00068] | 34.70 | 240.31 | 575.21 / 1025 |
| 384/512 | +0.00019 | [+0.00000, +0.00050] | 30.97 | 269.83 | 572.97 / 1019 |
| 768/512 | +0.00020 | [-0.00020, +0.00079] | 31.19 | 268.43 | 575.21 / 1025 |

### Challenge cap comparison

| Source/output cap | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum | Truncated inputs | No EOS | Repetition candidates |
|---|---:|---:|---:|---:|---:|---:|---:|
| 384/256 | 0.50269 | 0.29168 | 0.36398 | 0.48984 | 32 | 29 | 1 |
| 768/256 | 0.48597 | 0.27536 | 0.34574 | 0.47262 | 4 | 29 | 4 |
| 384/512 | 0.51525 | 0.29870 | 0.37506 | 0.50215 | 32 | 0 | 2 |
| 768/512 | 0.49501 | 0.28050 | 0.35416 | 0.48172 | 4 | 1 | 6 |

| Difference from 384/256 | ROUGE-L difference | 95% paired interval | Generation seconds | Returned tokens/s | Allocated / sampled-device MiB |
|---|---:|---|---:|---:|---:|
| 384/256 | Baseline | - | 9.56 | 852.39 | 573.99 / 783 |
| 768/256 | -0.01824 | [-0.03573, -0.00373] | 12.52 | 653.27 | 1189.58 / 1627 |
| 384/512 | +0.01108 | [+0.00563, +0.01674] | 14.20 | 612.14 | 573.99 / 783 |
| 768/512 | -0.00981 | [-0.02860, +0.00598] | 15.11 | 595.81 | 1189.58 / 1627 |

Intervals use 1,000 shared paired draws over sampled groups, seed 3732. They
are exploratory evaluation-sample uncertainty, not factuality confidence. The
challenge comprises the 32 longest remaining reserved groups, all BIMCV-COVID19;
length and domain are confounded. Its sources reach 1,111 tokens: four still
exceed 768. Full reference lengths are not minimum required output lengths, but
nine challenge references exceed 512 tokens. References remain complete for ROUGE.

All 640 outputs were retained; none was empty. Repetition candidates use the same
limited four-gram heuristic as the original report and are not verified clinical
error rates. A missing terminal full stop is also only a textual review flag.
In the challenge, output-only expansion changes no-EOS from 29 to zero while
outputs lacking terminal punctuation change from 29 to 31: a termination flag
improves without proving sentence/content completion. The selected cases below
check the actual text rather than inferring failure from punctuation alone.

![Cap comparison and termination flags](figures/cutoff-dev-comparison.png)

Timing includes the first generation batch and excludes loading/persistence/
scoring. These short sequential jobs have noisy timing and different output
lengths; do not infer a controlled causal speedup from nonmonotonic times.
Peak allocated memory reached 1,189.58 MiB and sampled device use 1,627 MiB,
leaving at least 2,469 MiB sampled headroom. Thermal safeguards stayed enabled.
The complete eight-job workflow lasted about 5.2 minutes including overhead;
per-setting generation costs are reported above. No OOM or process error occurred.

### Four diagnostic case comparisons

These cases were selected using long inputs, gain/repetition patterns and paired
contrasts. They are AI-assisted source-grounded text reviews, not a random
clinical-error estimate or radiologist adjudication. Original source, full
reference, visible inputs and all four outputs were checked. Only minimal
paraphrases are published.

- `train:145240`: holding the source at 384 and extending output to 512 preserves
  the original prefix, then adds the already-visible residual-fluid and newly
  appeared left-lung findings. This is a local completeness recovery. Earlier
  fistula/vein wording errors remain. With input 768, the model instead repeats
  procedural statements; extra output extends that repetition.
- `train:150111`: even 768 still truncates this long source. Some additional
  clinical content becomes visible, yet the generator loops on placeholder
  metadata. At 384/512 it emits EOS after contradictory statements and an
  unfinished clause. Enlarging budgets is neither input-completeness proof nor
  a remedy for a repetition loop.
- `train:148510`: source 768 exposes the late remission interpretation and
  fracture findings, but outputs omit them and repeat organ descriptions. The
  combined-cap ROUGE-L increases from about 0.196 to 0.282 despite that failure.
  EOS appears at token 251 with an unfinished phrase; a 512 cap gives exactly
  the same output. This is early termination/content selection, not a binding
  maximum-output limit.
- `train:132241`: a new splenic finding is outside the 384 input but inside
  768. Neither wider-input output mentions it; both repeat chest findings.
  More input repairs a temporal measurement association locally while other
  anatomical errors persist. At source 384, both output caps yield the same
  214-token EOS output: raising a limit cannot force useful continuation after
  the model already chose EOS.

### What the controlled comparisons imply

For fixed source/batch/greedy settings, every baseline no-EOS output in both
samples is an exact decoded-text prefix of its 512-token-cap counterpart:
3/3 representative and 29/29 challenge rows at each source cap. This is text
prefix agreement, not independently reconstructed raw-token equivalence. Extra
budget appends content and cannot retroactively repair an erroneous prefix.
Changing the source cap instead changes the encoder representation and therefore
can change the entire generated sequence, including early claims.

A tokenizer-only probe of the existing `ReportDataset` rule on four selected long
references produced 256 labels ending in EOS. Training used capped targets,
whereas scoring uses full references. A learned termination/content-selection
bias is consistent with premature EOS and short coverage on long inputs; no
training ablation established that it caused these particular failures. Raising
an inference maximum does not retrain that behaviour. Likewise, longer encoder
input alone supplies no explicit coverage or factual-consistency objective.

Output-only expansion is a candidate for a **reviewed continuation experiment**:
it recovered a visible finding and raised challenge ROUGE-L by 0.01108
[0.00563, 0.01674] at 1.49 times observed generation cost. It is not an automatic
safety fix. Wider input removed 28 of 32 input-truncation flags but worsened
mean overlap and raised repetition candidates; expanding both did not establish
an aggregate improvement. Keep the completed baseline defaults/results intact.
Route long, repetitive, unfinished or source-inconsistent reports for source
review; section-aware handling and training with suitable full/section targets
are future work, not implemented or clinically validated controls here.

### Verification and retained evidence

The implementation passed 151 CPU regressions and independent code review.
Independent results audits reproduced option 3's 40 ROUGE means/32 intervals and
option 1's 32 means/24 intervals, checked group separation, all 1,840 comparison
outputs, producer/checkpoint identities, checksummed journals, resources and
reported case claims. The original baseline README/PDF, weights and 30,000
predictions were preserved. Superseded accounting pilots remain separate.
No raw token IDs were saved: first-EOS logic was tested and exported totals
reconciled, but exact EOS positions cannot be reconstructed from decoded text.
No additional teacher model, clinical judge, retraining, upstream PR or Turnitin
submission was performed as part of these extensions.
