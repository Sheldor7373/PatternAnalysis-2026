# Required generation and evaluation plan

This plan prepares the mandatory frozen/full/LoRA comparison. **No GPU pilot or
official-validation generation has been run by this preparation.** Start GPU
inference only after the training worker and its supervisor have exited.

## 1. Freeze model selection

Verify the completed run's immutable `best` checkpoint and recorded development
selection decision. Record actual epoch/exposure/step, LR/batch history and
training resources. Resolve `$FullCheckpoint` and `$LoRACheckpoint` to immutable
directories. The frozen model uses the same pinned base revision. Do not use
official validation to choose a checkpoint, decoder or batch size.

## 2. Prepare a small timing set on CPU

```powershell
.\.venv\Scripts\python.exe prepare_generation_pilot.py --data data/prepared --output runs/generation-pilot/dev64.jsonl --limit 64
```

The helper reads only `dev.jsonl`, excludes the fixed 256 monitoring IDs, and
selects 64 remaining reports deterministically with seed 3711. It saves source
and selection hashes and refuses to overwrite the prepared-data directory or
an existing pilot. Neither official validation nor training weights are read.

## 3. Measure feasible batching after training

Keep prompt, source cap 384, output cap 256, greedy decoding, precision and
selected IDs fixed. Begin with batch 4 for all three models; compare batch 8 only
if batch 4 is stable with useful headroom. Use separate fresh output directories
for each model/batch. For example:

```powershell
.\.venv\Scripts\python.exe predict.py --manifest runs/generation-pilot/dev64.jsonl --checkpoint $FullCheckpoint --output runs/generation-pilot/full-b4 --device cuda --precision bf16 --batch-size 4 --source-length 384 --target-length 256 --beams 1 --evaluation-role development-pilot
.\.venv\Scripts\python.exe predict.py --manifest runs/generation-pilot/dev64.jsonl --checkpoint $FullCheckpoint --output runs/generation-pilot/full-b8 --device cuda --precision bf16 --batch-size 8 --source-length 384 --target-length 256 --beams 1 --evaluation-role development-pilot
```

Repeat for LoRA; use `--model $ModelDir --revision
0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab` for frozen. These pilot files carry a
development-only scope even though every row of that small manifest is used.

Record generated token counts, wall/generation time, tokens/s and allocated,
reserved and sampled device memory. Allow at least 512 MiB whole-device memory
headroom and recognise that a small sample does not prove a worst-case memory
bound. If a candidate OOMs, preserve its outputs/logs and choose the smaller
stable batch; no training restart is required. Select a common batch for the
primary comparison based on measured runtime and memory, not pilot ROUGE.
Do not add beam search, confidence scoring or new model experiments here.

Estimate all three 10,000-report generation costs from their measured rates,
including startup, persistence and scoring allowance. The old eight-report
batch-one pilot alone extrapolates to about 7.32 hours, so assuming that schedule
fits the remaining project budget would be unsafe. No batching speedup is
claimed before measurement. Reassess the mandatory schedule if it will not fit.

## 4. Lock settings and run final validation

Use `data/prepared/evaluation.jsonl`, `--evaluation-role final-validation`,
all 10,000 rows, no `--limit`, the selected
immutable model identities and the same settings for all models. Run the three
jobs sequentially under the existing durable supervision pattern. Set output
directories `runs/final/frozen`, `runs/final/full`, `runs/final/lora`.

Generation now publishes checksum-protected batches. On failure, repeat the
same command/output with `--resume`; completed batches are verified and skipped.
Do not change the model, source file, producer code, decoder or batch during a
final run. Keep the same effective precision, device and CPU/tokenizer thread
settings. Base weight/config/tokenizer artifacts are bound separately from a
LoRA adapter; local binary shards are included. Set
`$env:TOKENIZERS_PARALLELISM = "false"` consistently for supervised commands.
A bounded `--max-minutes` session can be resumed with a different
time allowance. Inspect `generation_status.json`; partial runs are not final
metrics. Preserve all successful batches and the only recoverable state.

## 5. Complete the required report

Check all result IDs/hashes/counts/settings match before comparing ROUGE. Collate
training and inference resources with consistent units and scope. Use CPU
diagnostics for the aggregate/per-example comparison and inspect 3–5 actual
failures for hallucinations, omitted acute findings and readability trade-offs.
Explain source evidence and plausible triggers; do not treat regex flags as
ground truth or selected failures as a representative error rate. Update the
manager recommendation, completed loss figure, README and PDF with actual
results. Keep all unmeasured fields visibly pending until that work is done.
