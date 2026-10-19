"""Generate comparable predictions and profile latency on a fixed manifest."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import torch
from transformers import AutoTokenizer

from dataset import PROMPT, read_jsonl, select_rows, write_json, write_jsonl
from metrics import safety_flags, score_predictions
from modules import MODEL, RadiologySimplifier, choose_device, load_checkpoint
from runtime import environment, memory_stats, precision_context, reset_memory, set_seed, sha256, synchronise


def main(args):
    set_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = choose_device(args.device)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; choose a fresh path")
    output.mkdir(parents=True, exist_ok=True)
    if args.checkpoint:
        model, tokenizer, settings = load_checkpoint(args.checkpoint, device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
        model = RadiologySimplifier(args.model, args.revision, mode="frozen").to(device)
        settings = {"model": args.model, "mode": "frozen", "revision": model.model.config._commit_hash or args.revision}
    model.eval()
    model.model.config.use_cache = True
    rows = select_rows(read_jsonl(args.manifest), args.limit, args.seed)
    if not rows:
        raise ValueError("Empty input manifest")
    predictions, latencies = [], []
    generated_tokens, truncated_inputs, capped_outputs = 0, 0, 0
    reset_memory(device)
    start = time.perf_counter()
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset:offset + args.batch_size]
        texts = [PROMPT + row["input"] for row in batch]
        lengths = [len(ids) for ids in tokenizer(texts, truncation=False)["input_ids"]]
        truncated_inputs += sum(n > args.source_length for n in lengths)
        tokens = tokenizer(texts, truncation=True, max_length=args.source_length, padding=True,
                           return_tensors="pt").to(device)
        synchronise(device)
        tick = time.perf_counter()
        with torch.inference_mode(), precision_context(device, args.precision):
            generated = model.model.generate(**tokens, max_new_tokens=args.target_length,
                                             num_beams=args.beams, do_sample=False)
        synchronise(device)
        latencies.append(time.perf_counter() - tick)
        # Count nonpadding tokens after the decoder start, including EOS.
        generated_tokens += int(generated[:, 1:].ne(tokenizer.pad_token_id).sum())
        capped_outputs += sum(tokenizer.eos_token_id not in seq[1:] for seq in generated.tolist())
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        predictions.extend({**row, "prediction": text} for row, text in zip(batch, decoded))
        print(f"{len(predictions)}/{len(rows)} reports", flush=True)
    elapsed = time.perf_counter() - start
    write_jsonl(output / "predictions.jsonl", predictions)
    metrics = {"rouge": score_predictions(predictions), "examples": len(rows), "manifest": str(args.manifest),
               "manifest_sha256": sha256(args.manifest),
               "selected_ids_sha256": hashlib.sha256("\n".join(r["id"] for r in rows).encode()).hexdigest(),
               "evaluation_scope": "complete manifest" if args.limit is None else "sampled pilot; not final evaluation",
               "model": settings, "generation": {"prompt": PROMPT, "source_length": args.source_length,
                                                  "max_new_tokens": args.target_length, "beams": args.beams,
                                                  "sampling": False, "seed": args.seed},
               "truncated_inputs": truncated_inputs, "outputs_without_eos": capped_outputs,
               "wall_seconds": elapsed, "generation_seconds": sum(latencies),
               "generated_tokens": generated_tokens, "tokens_per_second": generated_tokens / sum(latencies),
               "reports_per_second": len(rows) / elapsed, "batch_size": args.batch_size,
               "median_batch_latency_seconds": statistics.median(latencies),
               "latency_note": "Model generation only, includes first batch; wall time includes tokenisation, excludes loading/scoring.",
               "precision": args.precision, **model.parameter_counts(), **memory_stats(device),
               "environment": environment(device)}
    write_json(output / "metrics.json", metrics)
    flagged = []
    for row in predictions:
        flags = safety_flags(row["input"], row["prediction"])
        if any(flags.values()):
            flagged.append({**row, "review_flags": flags})
    write_jsonl(output / "review_candidates.jsonl", flagged)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="data/prepared/dev.jsonl")
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--revision", default="main")
    p.add_argument("--limit", type=int)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--source-length", type=int, default=384)
    p.add_argument("--target-length", type=int, default=256)
    p.add_argument("--beams", type=int, default=1)
    p.add_argument("--seed", type=int, default=3710)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--precision", choices=["auto", "bf16", "fp32"], default="auto")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = p.parse_args()
    if args.batch_size < 1 or args.beams < 1:
        p.error("batch-size and beams must be positive")
    main(args)
