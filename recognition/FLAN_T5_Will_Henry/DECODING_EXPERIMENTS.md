# Development decoding and cutoff investigations

This is a separate extension to the completed 29 October baseline. Its original
README, PDF, 30,000 predictions and selected checkpoints remain unchanged.
No retraining, teacher model, API or paid compute is involved.

## Design fixed before the new outputs

First compare the selected full-tuned model with greedy, beam2, beam4 and greedy
context-aware decoding (CAD alpha0.25/0.5). Then investigate source/output caps.
The implementation follows [Shi et al., 2024, Section 2.2](https://aclanthology.org/2024.naacl-short.69/)
and was checked against their [official implementation](https://github.com/xhan77/context-aware-decoding).
For every decoder prefix, CAD uses the same model twice:

`adjusted_logits = (1 + alpha) * logits(instruction + report, prefix) - alpha * logits(instruction, prefix)`.

Both streams share the generated prefix and keep separate encoder/KV state.
The instruction-only stream contains no report. Contrast arithmetic is float32
under BF16 model autocast. Alpha0 bypasses the extra pass and exactly follows
the ordinary generation path. CAD is restricted to beam1; ordinary beam search
uses Transformers generation with length penalty1 and early_stopping false.
This greedy CAD adaptation is not a reproduction of the paper's nucleus-sampling
summarisation protocol and does not inherit its factuality results.

`predict.py` binds the decoder, alpha, instruction-only prior, beams, length
penalty, stopping defaults, source/output caps and decoder module hash into its
durable resume identity. Changing these requires a fresh output directory.
The original producer remains available at commit860d28f; original artifacts
are preserved rather than relabelled as produced by the extended code.

## New development reservations

`prepare_decoding_study.py` excludes complete transitive report/image components
connected to the fixed dev256 monitor or any supplied historical prediction/
review manifest, including dev64. The actual inventory excluded316 components,
leaving8,820 of9,136 development components. It reserves disjoint groups before
any new generation:192 representative decoding rows,48 input-enriched challenge
rows,128 unused confirmation rows, and separate128 representative/32 challenge
rows for cutoff work. Representative rows are sampled before enrichment: one
random row per shuffled component, seed3729. This is group-balanced rather than
the original row-weighted distribution. The48 challenge rows all come from
BIMCV-COVID19; domain and difficulty are therefore confounded.

Historical exclusions are as complete as the recorded manifests. The broader
development split already influenced checkpoint selection through its monitor.
The official10,000 validation rows have been inspected and are not a fresh test.
No patient IDs exist. Reservations and all raw clinical text remain local.

## Interpretation and checks

Read ROUGE1/2/L/Lsum alongside paired source-grounded factuality, omissions,
uncertainty and qualitative readability. Regex flags are review candidates,
not measured clinical-error prevalence. Record generation time, output lengths,
EOS/repetition, memory and thermal conditions. Beam search optimises sequence
likelihood under its scoring rule; it does not enforce factual truth. A promising
candidate may be compared with greedy on the reserved128 only after selection;
a negative result need not consume that reserve.

Cutoff work follows the decoding report, using its own reserved groups. Compare
384/256 with768/256,384/512 and768/512. Separate information outside the visible
input from omitted visible content, and a missing EOS token from demonstrated
semantic/sentence truncation. Longer caps can add cost or unsupported material;
they are not automatically a recovery. Any inspection of old official-validation
cutoff cases is exploratory, not independent confirmation.

Offline tests cover the normalized CAD probability ratio, invalid settings,
alpha0/native equivalence, real beam2/4 paths, cached versus full-prefix logits,
controlled mixed early-EOS batches, resume equality and changed-knob refusal.
An independent read-only implementation review found no blocking issue. New
strategy memory pilots must pass before the bounded development runs. Ten-minute
process/GPU health observations and checksum-published batches remain enabled.

A memory pilot exposed a Transformers4.57 beam-tail accounting issue: zero-pad
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

Results are pending; no improvement is claimed here.
