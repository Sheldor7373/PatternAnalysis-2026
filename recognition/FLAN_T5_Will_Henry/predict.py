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
from checkpoint_storage import resolve_checkpoint
from prediction_storage import PredictionRun
from gpu_lease import gpu_lease
from inference_identity import bind_source, verify_source, effective_runtime


def main(args):
    # The actual worker owns this lease, so an observer crash cannot release it.
    with gpu_lease(args.gpu_lock):
        with PredictionRun(args.output, args.resume) as run:
            return generate(args, run)


def generate(args, run):
    set_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = choose_device(args.device)
    runtime = effective_runtime(args.precision, device)
    output = Path(args.output)
    base = None
    if args.checkpoint:
        # Bind a mutable best/latest alias once. Loading and recorded identity
        # must refer to this same immutable directory if the alias later moves.
        checkpoint = resolve_checkpoint(args.checkpoint)
        checkpoint_settings = json.loads((checkpoint / "experiment.json").read_text(encoding="utf-8"))
        if checkpoint_settings["mode"] == "lora":
            base = bind_source(checkpoint_settings["model"], checkpoint_settings["revision"])
            model, tokenizer, settings = load_checkpoint(checkpoint, device,
                                                        base_source=(base["source"], base["revision"]))
        else:
            model, tokenizer, settings = load_checkpoint(checkpoint, device)
    else:
        base = bind_source(args.model, args.revision)
        tokenizer = AutoTokenizer.from_pretrained(base["source"], revision=base["revision"])
        model = RadiologySimplifier(base["source"], base["revision"], mode="frozen").to(device)
        settings = {"model": args.model, "mode": "frozen", "revision": base["revision"]}
    if base is not None:
        verify_source(base)
    model.eval()
    model.model.config.use_cache = True
    rows = select_rows(read_jsonl(args.manifest), args.limit, args.seed)
    if not rows:
        raise ValueError("Empty input manifest")
    if args.evaluation_role == "final-validation" and (args.limit is not None or len(rows) != 10000):
        raise ValueError("Final-validation role requires all 10000 reports and no --limit")
    producer = environment(device)
    if args.checkpoint:
        model_identity = {"checkpoint_completion_sha256": sha256(checkpoint / "COMPLETE.json")}
        if base is not None:
            model_identity["adapter_base"] = base["identity"]
    else:
        model_identity = base["identity"]
    selected_ids_sha256 = hashlib.sha256("\n".join(row["id"] for row in rows).encode()).hexdigest()
    generation = {"prompt": PROMPT, "source_length": args.source_length,
                  "max_new_tokens": args.target_length, "beams": args.beams,
                  "sampling": False, "seed": args.seed, "visible_input_includes_prompt": True}
    code = Path(__file__).parent
    identity = {"model": model_identity, "manifest_sha256": sha256(args.manifest),
                "evaluation_role": args.evaluation_role,
                "selected_ids_sha256": selected_ids_sha256, "generation": generation,
                "batch_size": args.batch_size, "precision": args.precision, "device": str(device),
                "effective_runtime": runtime,
                "packages": producer["packages"],
                "code_sha256": {name: sha256(code / name) for name in
                                ("predict.py", "prediction_storage.py", "inference_identity.py", "gpu_lease.py", "modules.py", "dataset.py", "metrics.py", "runtime.py")}}
    run.configure(identity, {"model": settings, "environment": producer,
                             "effective_runtime": runtime,
                             "parameters": model.parameter_counts()}, rows)
    reset_memory(device)
    start = time.perf_counter()
    for offset in range(len(run.predictions), len(rows), args.batch_size):
        batch_started = time.perf_counter()
        batch = rows[offset:offset + args.batch_size]
        texts = [PROMPT + row["input"] for row in batch]
        lengths = [len(ids) for ids in tokenizer(texts, truncation=False)["input_ids"]]
        tokens = tokenizer(texts, truncation=True, max_length=args.source_length, padding=True,
                           return_tensors="pt").to(device)
        synchronise(device)
        tick = time.perf_counter()
        with torch.inference_mode(), precision_context(device, runtime["effective_precision"]):
            generated = model.model.generate(**tokens, max_new_tokens=args.target_length,
                                             num_beams=args.beams, do_sample=False)
        synchronise(device)
        latency = time.perf_counter() - tick
        # Count nonpadding tokens after the decoder start, including EOS.
        token_counts = generated[:, 1:].ne(tokenizer.pad_token_id).sum(dim=1).tolist()
        ended = [tokenizer.eos_token_id in seq[1:] for seq in generated.tolist()]
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        visible = tokenizer.batch_decode(tokens["input_ids"], skip_special_tokens=True)
        visible_counts = tokens["attention_mask"].sum(dim=1).tolist()
        produced = [{**row, "prediction": decoded[i], "visible_input": visible[i],
                     "source_tokens_before_truncation": lengths[i], "visible_source_tokens": visible_counts[i],
                     "input_truncated": lengths[i] > args.source_length,
                     "generated_tokens": token_counts[i], "output_has_eos": ended[i]}
                    for i, row in enumerate(batch)]
        run.commit({"start": offset, "rows": produced, "generation_seconds": latency,
                    "wall_seconds": time.perf_counter() - batch_started,
                    "generated_tokens": sum(token_counts),
                    "truncated_inputs": sum(n > args.source_length for n in lengths),
                    "outputs_without_eos": sum(not value for value in ended), **memory_stats(device)})
        print(f"{len(run.predictions)}/{len(rows)} reports (durably saved)", flush=True)
        if args.max_minutes is not None and time.perf_counter() - start >= args.max_minutes * 60 and len(run.predictions) < len(rows):
            run.progress("paused_time_budget")
            return {"state": "paused_time_budget", "committed_examples": len(run.predictions), "total_examples": len(rows)}
    predictions = run.predictions
    run.export_predictions()
    latencies = [batch["generation_seconds"] for batch in run.batches]
    elapsed = sum(batch["wall_seconds"] for batch in run.batches)
    generated_tokens = sum(batch["generated_tokens"] for batch in run.batches)
    truncated_inputs = sum(batch["truncated_inputs"] for batch in run.batches)
    capped_outputs = sum(batch["outputs_without_eos"] for batch in run.batches)
    peaks = {key: max((batch[key] for batch in run.batches if batch.get(key) is not None), default=None)
             for key in ("peak_allocated_mib", "peak_reserved_mib")}
    scopes = {"development-pilot": "development-only pilot; not final validation",
              "final-validation": "official held-out validation; all 10000 reports",
              "custom": "complete manifest" if args.limit is None else "sampled pilot; not final evaluation"}
    metrics = {"rouge": score_predictions(predictions), "examples": len(rows), "manifest": str(args.manifest),
               "manifest_sha256": sha256(args.manifest),
               "selected_ids_sha256": selected_ids_sha256,
               "evaluation_scope": scopes[args.evaluation_role], "evaluation_role": args.evaluation_role,
               "model": run.metadata["model"], "generation": generation,
               "truncated_inputs": truncated_inputs, "outputs_without_eos": capped_outputs,
               "wall_seconds": elapsed, "generation_seconds": sum(latencies),
               "generated_tokens": generated_tokens, "tokens_per_second": generated_tokens / sum(latencies),
               "reports_per_second": len(rows) / elapsed, "batch_size": args.batch_size,
               "median_batch_latency_seconds": statistics.median(latencies),
               "latency_note": "Generation time includes each session's first batch. Wall time sums batch tokenisation/generation/decoding; excludes loading, persistence, scoring and gaps between sessions.",
               "precision": args.precision, **run.metadata["parameters"], **peaks,
               "environment": run.metadata["environment"], "effective_runtime": run.metadata["effective_runtime"],
               "model_identity": run.identity["model"], "resumable_batch_count": len(run.batches)}
    write_json(output / "metrics.json", metrics)
    flagged = []
    for row in predictions:
        flags = safety_flags(row["input"], row["prediction"])
        if any(flags.values()):
            flagged.append({**row, "review_flags": flags})
    write_jsonl(output / "review_candidates.jsonl", flagged)
    run.progress("completed")
    print(json.dumps(metrics, indent=2))
    return metrics


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="data/prepared/dev.jsonl")
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--gpu-lock", help="Optional shared local GPU lease; acquired before model loading")
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
    p.add_argument("--resume", action="store_true", help="Verify and continue completed batches in this output directory")
    p.add_argument("--max-minutes", type=float, help="Pause after a committed batch; loading/scoring excluded; may be changed on resume")
    p.add_argument("--evaluation-role", choices=["custom", "development-pilot", "final-validation"], default="custom",
                   help="Explicit result scope; final-validation requires the complete 10000-report manifest")
    return p


if __name__ == "__main__":
    p = parser()
    args = p.parse_args()
    if args.batch_size < 1 or args.beams < 1:
        p.error("batch-size and beams must be positive")
    if args.max_minutes is not None and args.max_minutes < 0:
        p.error("max-minutes must be nonnegative")
    main(args)
