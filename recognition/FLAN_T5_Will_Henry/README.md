# Radiology reports in plain English

COMP3710 Project 2.6 — William Henry

This prototype rewrites technical radiology reports into language a patient can
understand. It compares untuned FLAN-T5-small, full fine-tuning and rank-8 LoRA
on BioLaySumm LaymanRRG. The engineering question is whether adaptation
preserves findings, negation and diagnostic uncertainty while improving the
rewrite, and whether its computational cost is justified. The intended output
is a draft for review; the clinical safety audit below determines what use the
evidence can support.

## Current experimental status

**Completed comparison, 29 October 2026 (UTC).** Full fine-tuning and LoRA
each completed three epochs and 292,233 training exposures. All three models
were evaluated on the same 10,000 held-out reports. Checkpoint payloads, paired
IDs, visible inputs and final exports were verified. The results support a
prototype comparison, not clinical deployment or calibrated confidence in an
individual rewrite. The frozen model requires no task-specific training.

The practical feasibility check-off has been reported complete by the author.
The one-page review presented on 29 October is reproduced below.

## How the model works

FLAN-T5 is an instruction-tuned encoder–decoder Transformer, building on T5's
text-to-text formulation [1, 2]. The pinned small checkpoint has eight encoder
and eight decoder blocks, model width 512, six attention heads, gated-GELU
feed-forward blocks and relative position attention. The encoder reads the task
prompt and report. During generation, the decoder attends to that representation
and its own previously generated tokens, stopping at EOS or the output cap.
These dimensions come from the pinned checkpoint configuration rather than an
assumption that every model named “T5-small” has identical settings.

![Model and adaptation paths](figures/architecture.svg)

Training uses teacher forcing: the decoder receives the reference prefix and
predicts the next reference token. Padding labels are `-100` and contribute no
loss. For valid target-token positions M, the objective is

$$
\mathcal{L}(\theta)=-\frac{1}{|M|}\sum_{(i,t)\in M}
\log p_\theta(y_{i,t}\mid x_i,y_{i,<t}).
$$

The training loop weights accumulated gradients by target-token count, including
short final accumulation groups. A long report therefore contributes more
target tokens than a short report. This is different from giving each report's
mean loss equal weight. A CPU test compares the masked model loss with a direct
cross-entropy calculation.

Full fine-tuning updates all model parameters. LoRA freezes the base model and
adds low-rank updates to attention query and value projections [3]:

$$W'=W+\frac{\alpha}{r}BA,\qquad r=8,\quad\alpha=16.$$

The saved adapter has 48 A matrices of shape 8 x 512 and 48 B matrices of
shape 384 x 8: `48 x 8 x (512 + 384) = 344,064` parameters. Query/value
adaptation spans eight encoder self-attention, eight decoder self-attention
and eight decoder cross-attention modules. Only A and B are trained;
adapter dropout is 0.05. This reduces trainable
parameters and optimizer state, but the frozen base still occupies memory and
must participate in forward/backward computation. Fewer trainable parameters
alone do not guarantee faster generation or better factual faithfulness.

## Data, tokenisation and leakage controls

The dataset is [BioLaySumm2025-LaymanRRG open-source track][dataset], pinned to
revision `6783ab38f9de28c4372799346f51d3927ae7eb9a`. Its raw splits contain
150,454 training pairs and 10,000 validation pairs. The audit found no missing
report/target pairs. Report/image connected components prevent indirect overlap:
42,249 training rows connected to validation are removed, and the remaining
components are split into **97,411 training** and **10,794 development** rows.
All **10,000 official validation rows** retain their original weighting and full
reference rewrites for the final comparison. The challenge test set is unused.

Normalised text is used for overlap checks; the model receives the report text
with its negation and uncertainty wording preserved. Available report/image
identities can be checked, but the data supplies no patient IDs, so these checks
do not establish patient independence. Repeated inputs and differing reference wordings
are counted in the local audit and are a limitation of reference-based scoring.
Data hygiene and length audits are distinct from model selection: validation is
not used to select checkpoints, decoding or inference batch size.

Every input is prefixed with:

```text
Rewrite this radiology report in plain English. Preserve all findings, negation and uncertainty:
```

The source cap is 384 tokens and the target-training/generation cap is 256.
Training references are truncated for the loss; ROUGE uses the complete saved
reference text. The preliminary training audit found 0.80% of sources and 1.83%
of targets beyond these caps. Final predictions record source truncation,
tokenizer-decoded visible input, token lengths and EOS/cap status so an omission
can be investigated against what was available to the model.

## Training and checkpoint selection

All runs use seed 3710, AdamW with base LR `1e-4`, weight decay 0.01, 5% warm-up,
gradient norm clipping at 1.0, and BF16 autocast on the RTX 3050 Ti Laptop GPU
(4 GiB). Autocast does not imply that every stored weight or optimizer tensor is
BF16. The fixed 256-report monitoring set comes from the training-derived
development split. Checkpoint selection uses eligible epoch-end loss with
`min_delta=0.001` and patience 2. Periodic diagnostics and interrupted-session
evaluations are not new selection decisions.

Both adaptation runs began with microbatch 1 and accumulation 2. LoRA moved to
batch 4 / accumulation 1 after 17,500 exposures; full fine-tuning moved after
81,558 exposures. Continuations restore optimizer/scheduler state, RNG and the
exact shuffled-data cursor. The next LR is preserved, then decay is anchored to
remaining example exposure. These batch transitions change gradient noise and
update frequency; the comparison must disclose that confound even if both runs
finish with the same number of exposures. Three epochs are this experiment's
planned budget, not a universal requirement for effective adaptation. Full used
93,449 optimizer updates versus 77,434 for LoRA despite equal total exposure.

| Recorded development evidence | Exposures / scope | Token loss |
|---|---|---:|
| Untuned checkpoint | Fixed dev256, before adaptation | 2.662897 |
| LoRA | End of epoch 3; 292,233 training exposures | 1.221684 |
| Full fine-tuning | Selected epoch-one checkpoint; 97,411 exposures | 0.648476 |
| Full fine-tuning | Selected epoch-two checkpoint; 194,822 exposures | 0.594326 |
| Full fine-tuning | Selected epoch-three checkpoint; 292,233 exposures | 0.578559 |

These measurements establish training progress, not final summary quality or
clinical safety. The loss curve uses cumulative examples seen, divided by
97,411, because optimizer steps differ between batch schedules. Training-window
and evaluation-mode development losses are different quantities. Session
segments and replayed uncheckpointed tails must not be presented as continuous
new exposure.

![Recorded training and development losses](figures/training-loss-snapshot.png)

*Completed three-epoch histories. Raw logs, immutable checkpoint checksums and
sampled device telemetry are retained locally; training data and weights are
excluded from Git. Lines remain separate across process/session boundaries.*

## Final held-out comparison

The recipe was locked before held-out generation: selected immutable
checkpoints, batch 64, BF16, greedy decoding (`beams=1`, no sampling), source
cap 384, output cap 256 and seed 3710. Prompt, validation IDs, full references,
effective runtime and all eight producer-file hashes match across models.
The source cap includes the prompt. A separate 64-report development timing
pilot tested batches 4/8/16/32/64 without quality-based selection. Batch 64 was
fastest tested, with at least 2,391 MiB sampled device headroom. BF16 greedy
outputs can change with batch shape, including meaningful wording changes;
the final batch and runtime were therefore fixed, including on resume.

| Model | Validation reports | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | ROUGE-Lsum F1 |
|---|---:|---:|---:|---:|---:|
| Frozen FLAN-T5-small | 10,000 | 0.26450 | 0.09638 | 0.23524 | 0.24556 |
| Full fine-tuning | 10,000 | 0.73844 | 0.55926 | 0.68952 | 0.70882 |
| LoRA | 10,000 | 0.61454 | 0.39429 | 0.55197 | 0.57977 |

![Held-out ROUGE comparison](figures/heldout-rouge.png)

ROUGE is the arithmetic mean of per-report F1 values on scale 0-1. Stemming is
enabled; explicit sentence boundaries support union-LCS ROUGE-Lsum. Full
references are scored even where training targets or visible sources exceeded
their caps. Matching words do not establish factual faithfulness [4].

### Paired uncertainty and where performance weakens

All 10,000 original rows retain their weighting. They form 7,599 observable
report/image components; the largest has 289 rows. There are 283 normalised
input groups with multiple reference wordings, which need not be contradictory.
The paired percentile bootstrap resamples whole components with shared draws,
then recomputes row-weighted differences (1,000 draws; seed 3710).

| ROUGE-L difference | Point difference | 95% percentile interval |
|---|---:|---:|
| Full minus frozen | 0.45427 | [0.43075, 0.48046] |
| LoRA minus frozen | 0.31673 | [0.29240, 0.34313] |
| Full minus LoRA | 0.13755 | [0.12045, 0.16223] |

Intervals describe evaluation-sample variation conditional on these trained
models, not training-seed uncertainty, patient independence or per-rewrite
confidence. No repeated-seed experiment was run. Different batch transitions
also prevent attributing the entire full-versus-LoRA gap to adaptation capacity.

| Source length (whitespace words) | Reports | Frozen ROUGE-L | Full ROUGE-L | LoRA ROUGE-L |
|---|---:|---:|---:|---:|
| 0-50 words | 9,188 | 0.24077 | 0.69806 | 0.56207 |
| 51-100 words | 621 | 0.19950 | 0.61836 | 0.46571 |
| >100 words | 191 | 0.08564 | 0.51010 | 0.34655 |

The aggregate is dominated by short reports: 9,188 have at most 50 words and
only 191 exceed 100. Full tuning leads in every reported length bin, but its
ROUGE-L falls from 0.69806 to 0.51010 between those extremes. Length, source
domain and case difficulty are entangled; this is descriptive stratification,
not a causal length effect. Source counts are PadChest 7,824, BIMCV-COVID19
2,042 and OpenI 134, limiting broad cross-domain claims.

### Resources and practical trade-offs

| Model | Total parameters | Trainable parameters | Training lineage time | Peak training allocated / reserved MiB | Sampled training device MiB |
|---|---:|---:|---:|---:|---:|
| Frozen | 76,961,152 | 0 | No task adaptation | Not applicable | Not applicable |
| Full | 76,961,152 | 76,961,152 | 13.60 h | 2687.49 / 3080 | 3266 |
| LoRA | 77,305,216 | 344,064 | 12.16 h | 1893.19 / 2226 | 2337 |

Training lineage time includes development evaluation and checkpointing, while
excluding paused calendar time. It is not total project compute: logged full
process spans including replay total at least 13.87 h, versus 13.60 h along the
retained checkpoint lineage. LoRA's retained lineage is 12.16 h; a superseded
trial consumed additional time. Whole-device samples can miss peaks and include
other processes. Foundation-model pretraining cost is outside this comparison.

| Model | Batch wall / observer session seconds | Median batch generation seconds | Amortised batch wall ms/report | Generated tokens/s | Inference allocated / reserved / sampled-device MiB |
|---|---:|---:|---:|---:|---:|
| Frozen | 587.63 / 620.19 | 4.580 | 58.76 | 663.72 | 1256.31 / 2624 / 2727 |
| Full | 697.12 / 740.42 | 4.528 | 69.71 | 709.04 | 1288.93 / 2530 / 2633 |
| LoRA | 966.33 / 1011.05 | 6.200 | 96.63 | 503.94 | 1257.84 / 2462 / 2565 |

Batch wall sums tokenisation, synchronised generation and decoding, including
each session's first batch; it excludes loading, persistence, scoring and paused
gaps. Observer session time includes those session costs. The median is per
batch, not individual-request latency; amortised ms/report is a throughput
quantity. Generated-token counts exclude padding/decoder start and include EOS.
Outputs average 38.68/49.01/48.41 tokens for frozen/full/LoRA, so reports/s alone
does not isolate model speed. Thermal conditions varied (sampled maxima
75/69/69 degrees C); these are observed costs, not a controlled speedup claim.

LoRA trains about 0.45% of its parameters and reduced recorded allocated
training memory, but the base weights/activations remain. Its adapters were
unmerged during inference, adding operations; lower trainable count did not
produce faster generation here. Both adapted models fitted this 4 GiB GPU.

### Output diagnostics

| Model | Truncated inputs | Outputs without EOS | Empty outputs | Within-output repetition candidates |
|---|---:|---:|---:|---:|
| Frozen | 52 | 119 | 27 | 90 |
| Full | 52 | 115 | 0 | 29 |
| LoRA | 52 | 195 | 0 | 157 |

Empty and unfinished outputs were retained and scored; no held-out setting was
changed to repair them. Repetition candidates require at least 24 words, a
four-word phrase repeated at least three times and duplicate four-gram fraction
at least 0.30. This is a review heuristic, not verified error prevalence.
Repeated complete outputs can also reflect repeated or normal input reports.

The final manifest SHA-256 is
`f3707225785622f24c4b60b412f364a26453c3d4cf92c5d9381e106a5e0d01a0`.
Producer code corresponds to commit `860d28f`; later report-only commits do
not change those eight source hashes. Local checksum batches, metrics, bootstrap
results and immutable checkpoint receipts retain the full numerical provenance.

## Source-grounded failure autopsies

These five failure-selected cases cover different mechanisms, rather than
estimating an error rate. Original source, tokenizer-visible source, full
reference and all three saved outputs were checked. The input/output summaries
below are minimal paraphrases, not substitute inputs used for a new experiment.
Row IDs refer to the pinned official validation split. The pinned dataset card
does not declare a licence; source datasets retain research/sharing conditions
[7]. Original reports, images and bulk predictions remain local.

### validation:69: Entity-attribute misbinding

**Input summary (PadChest; paraphrase):** The source separately describes an elongated aorta, an enlarged heart and a right basal infiltrate compatible with infection.

**Observed outputs:** LoRA retains heart enlargement but additionally turns the right lung finding into lung enlargement. Full tuning preserves the relevant infiltrate/infection relationship; frozen largely copies the technical source and task instruction.

**Trigger and response:** The error fits entity-attribute misbinding across adjacent findings: a fluent sentence attaches an enlargement property to the wrong organ. Capacity, attention and reference-noise contributions were not isolated. Check each finding against its anatomical subject; fluency and shared disease words do not establish the correct relation.

### validation:164: Spatial negation and modifier scope

**Input summary (OpenI; paraphrase):** The source describes normal central airways, patchy lung densities, suspected acute pneumonia and no pneumothorax.

**Observed outputs:** LoRA changes the normal-airway statement into normal lungs and expresses the absent extra-pulmonary air finding as absence of air inside the lungs. Full preserves the relevant spatial negation. Frozen returns mainly the concluding impression, omitting other supplied findings.

**Trigger and response:** This is a relation/scope error, not merely a missing negation token. A normality modifier migrates between structures, and a preposition changes what is being negated. The entire source was visible. Review the negated concept together with its anatomical location; a bag-of-concepts match can miss this failure.

### validation:206: Internal contradiction and anatomical substitution

**Input summary (OpenI; paraphrase):** The source lists a possible right-middle-lobe process, explicitly excludes pleural fluid and notes a raised right hemidiaphragm.

**Observed outputs:** LoRA first adds a small fluid collection and then denies fluid in the next sentence; it also substitutes the right middle lung region for the elevated diaphragm. Full preserves the relevant uncertainty and negation. Frozen keeps only the possible lung-process sentence.

**Trigger and response:** Autoregressive token prediction does not enforce cross-sentence consistency or correct anatomical binding. All relevant statements were visible; the reference also preserves them. No causal decoder/adapter ablation was run. Check for contradictory positive/negative claims and align each anatomical finding to the source.

### validation:3207: Unsupported reassurance and certainty inflation

**Input summary (BIMCV-COVID19; paraphrase):** The source states that acquisition technique reduces diagnostic sensitivity and describes a possible small upper-right opacity.

**Observed outputs:** Full tuning dismisses the acquisition limitation and turns the possible opacity into an unqualified finding. LoRA also drops the opacity qualifier and omits part of the acquisition caveat. Frozen enters a repetitive loop rather than providing a useful rewrite.

**Trigger and response:** This is unfaithful simplification despite a fully visible input. Full achieves ROUGE-L 0.678 on this case, showing that many matching words can coexist with a changed clinical assertion. A learned reassuring style is a hypothesis, not a proven cause. Preserve modality and uncertainty qualifiers explicitly; lexical overlap and completed text are insufficient safety checks.

### validation:195: Separate source and generation budget failures

**Input summary (BIMCV-COVID19; paraphrase):** A long multi-region CT report contains thoracic, abdominal and bone findings; some later findings lie beyond the visible input budget.

**Observed outputs:** The prompted source has 520 tokens but only 384 are visible. Full and LoRA both reach 256 generated tokens without EOS and stop unfinished. Frozen instead links the cut-off organ-name fragment to a lung-tumour label and ends with EOS.

**Trigger and response:** Unseen late findings are an input-visibility limitation, not evidence that the decoder ignored available text. Both adapted outputs also omit available later content because the generation budget is exhausted. The frozen association is an additional unsupported content failure; truncation as its trigger remains a hypothesis. Route truncated or unfinished reports for review; sentence-aware segmentation and larger budgets require development testing. EOS alone cannot certify fidelity.

Across these cases, frozen copies technical text or loops; full usually explains
terms more clearly, but still retains jargon such as the specialised nodule and
lung-region terminology in case 195. LoRA also retains jargon and awkward
phrasing; its contradictory sentences in case 206 undermine clarity. Full's
fluent reassurance in case 3207 illustrates that easier wording can be less
faithful. These are qualitative observations, not measured patient comprehension
or a collected numerical readability score.

### What the flags and scores cannot tell us

The full model's regex flags include false positives: `validation:171` preserves
the source negation, but the limited matcher misses a negative construction in
the output. `validation:127` preserves the source's questionable fracture, but
the matcher does not recognise the source uncertainty cue. Therefore the count
of flagged concepts is not a clinical-error rate. The rules have incomplete
synonyms, negation scope and anatomical relations. Source-grounded textual
review is also not independent radiologist adjudication.

**Manager recommendation:** prefer full tuning for further prototype work:
it has the strongest overlap scores and fitted the local GPU. LoRA offers a
large trainable-parameter reduction and lower recorded training memory, but
lost quality and was slower in this unmerged inference setup. None should
produce unreviewed patient-facing text: even full tuning altered uncertainty
and acquisition limitations. Empty/non-EOS outputs and truncated sources need
explicit review routing; all critical facts, negation and uncertainty need
source comparison before release. EOS is a completion signal, not a fidelity
certificate, as `validation:195` illustrates.

Next development priorities are sentence-aware long-report handling and a
labelled source-grounded error audit, followed by calibrated review thresholds
on development data. A future confidence score would need validation against
those factual-error labels; ROUGE is not such a score. These controls have not
been clinically validated or implemented as a deployment safety system here.

## Running the project

The code uses Python 3.11.2. Recorded versions are PyTorch 2.8.0+cu126,
Transformers 4.57.1, PEFT 0.17.1, Accelerate 1.10.1, Hugging Face Hub 0.35.3,
SentencePiece 0.2.1, safetensors 0.6.2, PyArrow 21.0.0, NumPy 2.2.6,
rouge-score 0.1.2, NLTK 3.9.2 and Matplotlib 3.10.6. Requirements are pinned in
[requirements.txt](requirements.txt); PyTorch's CUDA wheel is installed
separately following the [official previous-version instructions][torch-install].

Fresh-checkout PowerShell setup, from this project directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe dataset.py --output data/prepared
.\.venv\Scripts\python.exe prepare_assets.py --revision 0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab
$ModelDir = (Get-Content data/prepared/token_audit.json -Raw | ConvertFrom-Json).local_path
```

For a bounded **smoke run**, not the final training comparison:

```powershell
.\.venv\Scripts\python.exe train.py --mode full --model $ModelDir --revision 0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab --output runs/smoke/full --device cuda --precision bf16 --batch-size 1 --accumulate 2 --train-limit 64 --dev-limit 8 --epochs 1 --max-steps 2
```

Historical full/LoRA launch arguments, immutable checkpoint identities and
transition anchors are stored with each local run. They are necessary to
reconstruct the reported trajectories; the smoke command above is not a claim
to reproduce those trajectories. `modules.py`, `dataset.py`, `train.py` and
`predict.py` contain the model, preprocessing, optimization and saved-model test
driver respectively. Helper modules implement checkpoint integrity, run
supervision, durable prediction batches and CPU analysis.

### Generation after training

See [EVALUATION_PLAN.md](EVALUATION_PLAN.md) for the development-only batching
check. Once settings are frozen, set `$FullCheckpoint`, `$LoRACheckpoint` to
selected immutable best-checkpoint directories and `$InferenceBatch` to the
measured common batch size. Example final invocation:

```powershell
.\.venv\Scripts\python.exe predict.py --manifest data/prepared/evaluation.jsonl --checkpoint $FullCheckpoint --output runs/final/full --device cuda --precision bf16 --batch-size $InferenceBatch --source-length 384 --target-length 256 --beams 1 --evaluation-role final-validation
```

Use the same invocation/output folder with `--resume` after an interruption.
`--max-minutes` can bound a session and pauses after a saved batch; it may change
on resume. Model, manifest, producer-code identity, decoder, batch and precision
must match. The identity also binds actual local binary/safetensors shards and
tokenizer/config files, or the resolved immutable Hub revision. LoRA's external
base is bound separately from its adapter. Effective BF16/FP32, device
capability and CPU/tokenizer thread settings must remain consistent so resumed
resource measurements are not silently labelled with a different runtime.
Long evaluation jobs can use `supervise_prediction.py` for process/GPU logs and
ten-minute health observations after the completed-training audit. Its shared
`--gpu-lock` is held by the prediction worker, preventing another supervised
evaluation from entering while that worker survives an observer failure.
Observation-write failures are reported and retried; outputs remain recoverable.
Completed batches are checksummed and skipped; a failed unpublished
batch may be repeated. Final `predictions.jsonl`, `metrics.json` and review
candidates are produced only after the selected manifest completes. Official
validation runs omit `--limit`. All three final result sets must match in IDs,
row counts and generation settings before comparison.

CPU diagnostics and tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -p "test_*.py"
.\.venv\Scripts\python.exe research_diagnostics.py --result runs/final/frozen/metrics.json --result runs/final/full/metrics.json --result runs/final/lora/metrics.json --analyse-predictions --output runs/final-review
.\.venv\Scripts\python.exe plot_research.py --review runs/final-review/review.json --output runs/final-figures
.\.venv\Scripts\python.exe plot_training_history.py --runs-root runs --continuation $FullRunRoot --output runs/training-figure
```

Tests include padding-mask correctness, trainable adapter gradients, saved-model
reload, checkpoint/RNG/cursor continuation, Windows status-sharing behavior and
resumable inference with tiny offline models. CPU process tests also cover
observer termination with a live worker lease, real status-write failures and
checksum-preserving generation recovery. Exact floating-point agreement
across different GPUs or library versions is not promised. Data, weights and
generated report exports remain in ignored local directories.

## Feasibility review presented on 29 October 2026

The following reproduces the one-page review presented for the feasibility
check on 29 October 2026. The author reported the check-off successful that
morning. Its preliminary measurements and proposed next steps describe the
state at presentation; subsequent training progress is reported above.

**William Henry | COMP3710 Project 2.6 | Radiology report summarisation | 29 October 2026**

### User Need, Scope & Acceptance Criteria

Patients need readable explanations of radiology reports without invented
findings, missing acute findings or altered diagnostic certainty. This project
studies text-only simplification, aiming for plain, approximately sixth-grade
language; it does not interpret images or provide clinical advice. The central
question is whether task adaptation improves the usefulness of FLAN-T5 summaries
without sacrificing factual fidelity. Acceptance requires:

1. Frozen, full-tuned and LoRA outputs on the same held-out rows, with zero
   audited report/image-identity overlap across splits.
2. ROUGE-1/2/L/Lsum and total/trainable parameters, peak VRAM, runtime and
   inference tokens/s for all applicable runs.
3. A documented audit of 3–5 representative cases, with no invented pathology,
   negation reversal or critical omission as the target, and every failure reported.
4. The same cases scored for jargon, sentence clarity and preserved uncertainty,
   with every unclear output reported.
5. Operation within the local 4 GiB GPU limit without out-of-memory failure.

A small successful case audit would not establish clinical safety.

### Model Choice & Course Concepts

The HARD T5/FLAN-T5 option uses BioLaySumm2025-LaymanRRG-opensource-track
technical-report/lay-rewrite pairs. The audited split contains 97,411 training
and 10,794 development rows; all 10,000 official validation rows are reserved
for final comparison. FLAN-T5-small has 76,961,152 base parameters and an
encoder–decoder Transformer architecture. Full fine-tuning and rank-8 LoRA
start independently from the same pretrained checkpoint. The comparison
applies transfer learning, attention, supervised sequence prediction, masked
cross-entropy, gradient optimisation and low-rank adaptation. Source/target
limits are 384/256 tokens. Transformers/PEFT dependency approval must be
confirmed against the course rules.

### Preliminary Feasibility Evidence

The data audit removed 42,249 validation-connected training rows before
splitting, with no remaining cross-split overlap on the audited identities;
patient independence is unverified. Historical preparation checks recorded
51 passing tests and frozen/full/LoRA inference on eight development examples.
A representative RTX 3050 Ti 4 GB optimiser-step pilot peaked at 1,845 MiB for
full tuning and 975 MiB for LoRA, without an out-of-memory error. These
measurements establish capacity, not final throughput. In logs verified on
20 October, LoRA had completed three epochs; fixed development-subset loss
fell from 2.662897 to 1.221684. Full tuning was paused on 20 October at step
40,779 before its first epoch finished. No final three-way held-out quality
result is available.

### Risks, Budget & Fallback

The main risks are clinically misleading rewrites, truncation,
duplicate/uneven references, overfitting and a batch-size confound: LoRA changed
effective batch 2 to 4, while full tuning stayed at 2. The budget is one local
GPU, no paid compute, and up to three epochs for each adapted model, with
checkpoints and bounded retention. Once continuation is authorised, the next
experiment is to complete full tuning, select checkpoints using development
data, then run the locked three-way evaluation and case audit. If time is
insufficient, a clearly labelled shorter full-tuning run preserves the required
comparison; LoRA-only results do not satisfy it. Unsafe outputs remain research
examples, with source review and clinical sign-off required before any patient use.

## Artificial Intelligence Usage Disclosure

OpenAI ChatGPT/Codex assisted with literature/context lookup, implementation and
refactoring, data/metric audits, tests, training supervision, plotting and report
drafting. Suggestions were checked through split/hash verification, masked-loss
and model-reload tests, optimizer/RNG/cursor continuation tests, bounded GPU
feasibility measurements and retained training logs. Resumable inference was
tested on tiny offline CPU frozen/full/LoRA models, including failures and
unchanged output/order after resume. The author remains responsible for reviewing
the code and report. The final 30,000 outputs were checksum/identity audited;
per-example scoring was recomputed, report/image-group bootstrap intervals were
checked, and five cases were compared to original/visible source and reference.
The implementation passed 131 CPU regression tests before final generation;
producer source hashes remained unchanged. AI-assisted textual review and tests
do not substitute for human responsibility or establish clinical safety.

## Submission completion

Training, all three 10,000-report evaluations, resource summaries and the five
case autopsies are complete. A PDF export of this saved README is prepared
separately for Turnitin. Public Git contains code/documentation/figures, not
datasets or model weights. Final upstream PR creation and Turnitin submission
remain approval steps: the required working PR must target
`shakes76/PatternAnalysis-2026` branch `topic-recognition`. The later teaching-team
code-review response is separately marked; requested changes and replies must
be completed in its announced response window. No upstream merge or submission
is claimed here.

## References

1. Raffel et al. (2020), [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://jmlr.org/papers/v21/20-074.html).
2. Chung et al. (2024), [Scaling Instruction-Finetuned Language Models](https://www.jmlr.org/papers/v25/23-0870.html).
3. Hu et al. (2022), [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).
4. Maynez et al. (2020), [On Faithfulness and Factuality in Abstractive Summarization](https://aclanthology.org/2020.acl-main.173/).
5. [Google FLAN-T5-small model card](https://huggingface.co/google/flan-t5-small), pinned revision `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab`.
6. [BioLaySumm 2025 task](https://biolaysumm.org/) and [LaymanRRG dataset][dataset].

7. [Pinned LaymanRRG dataset card](https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track/blob/6783ab38f9de28c4372799346f51d3927ae7eb9a/README.md), [PadChest data-use information](https://bimcv.cipf.es/bimcv-projects/padchest/) and [BIMCV-COVID19 data-use information](https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/), checked 29 October 2026.

[dataset]: https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track
[torch-install]: https://pytorch.org/get-started/previous-versions/#v280
