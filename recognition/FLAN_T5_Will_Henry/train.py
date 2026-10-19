"""Fine-tune the same checkpoint with full weights or LoRA; record actual cost."""

import argparse
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, get_linear_schedule_with_warmup

from dataset import ReportDataset, read_jsonl, select_rows, write_json
from modules import MODEL, RadiologySimplifier, choose_device, load_checkpoint
from runtime import append_jsonl, environment, memory_stats, precision_context, reset_memory, set_seed, sha256, synchronise


def evaluate_loss(model, loader, device, precision):
    model.eval()
    total, tokens = 0.0, 0
    with torch.inference_mode():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            count = int(batch["labels"].ne(-100).sum())
            with precision_context(device, precision):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite development loss")
            total += float(loss) * count
            tokens += count
    return total / max(1, tokens)


def main(args):
    if args.batch_size < 1 or args.accumulate < 1 or args.epochs < 1:
        raise ValueError("batch_size, accumulate and epochs must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("max_steps must be positive")
    set_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = choose_device(args.device)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError("Output directory is not empty; use --resume or a fresh directory")
    output.mkdir(parents=True, exist_ok=True)
    train_path, dev_path = Path(args.data) / "train.jsonl", Path(args.data) / "dev.jsonl"
    train_rows = select_rows(read_jsonl(train_path), args.train_limit, args.seed)
    dev_rows = select_rows(read_jsonl(dev_path), args.dev_limit, args.seed)
    if args.resume:
        model, tokenizer, saved_settings = load_checkpoint(output / "latest", device)
        original = saved_settings["arguments"]
        excluded = {"resume", "device", "max_minutes", "cpu_threads"}
        changed = [key for key, value in vars(args).items() if key not in excluded and original.get(key) != value]
        if changed:
            raise ValueError(f"Resume configuration changed: {changed}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
        model = RadiologySimplifier(args.model, args.revision, args.mode, args.rank).to(device)
    revision = model.model.config._commit_hash or args.revision
    if args.gradient_checkpointing:
        model.model.gradient_checkpointing_enable()
        model.model.enable_input_require_grads()
    model.model.config.use_cache = False
    collator = DataCollatorForSeq2Seq(tokenizer, model=model.model, label_pad_token_id=-100)
    training = ReportDataset(train_rows, tokenizer, args.source_length, args.target_length)
    development = ReportDataset(dev_rows, tokenizer, args.source_length, args.target_length)
    dev_loader = DataLoader(development, batch_size=args.batch_size, collate_fn=collator,
                            generator=torch.Generator().manual_seed(args.seed))
    steps_per_epoch = math.ceil(math.ceil(len(training) / args.batch_size) / args.accumulate)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.05), total_steps)
    settings = {"model": args.model, "revision": revision, "mode": args.mode, "rank": args.rank,
                "arguments": vars(args), "train_manifest_sha256": sha256(train_path),
                "dev_manifest_sha256": sha256(dev_path), "train_examples": len(training),
                "dev_examples": len(development), **model.parameter_counts(), "environment": environment(device),
                "precision_used": "bf16" if device.type == "cuda" and args.precision != "fp32"
                and torch.cuda.is_bf16_supported() else "fp32"}
    start_epoch, start_batch, step, best, stale, elapsed_before = 0, 0, 0, math.inf, 0, 0.0
    if args.resume:
        # This state is produced locally by this script; never load an untrusted .pt file.
        state = torch.load(output / "latest" / "trainer.pt", map_location="cpu", weights_only=False)
        if saved_settings.get("checkpoint_step") != state["step"]:
            raise ValueError("Checkpoint save was interrupted: model and optimizer steps differ")
        if state["train_sha"] != sha256(train_path) or state["dev_sha"] != sha256(dev_path):
            raise ValueError("Prepared data changed since checkpoint")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, start_batch, step = state["epoch"], state["next_batch"], state["step"]
        best, stale, elapsed_before = state["best"], state["stale"], state["elapsed_seconds"]
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    write_json(output / "run.json", settings)
    reset_memory(device)
    started = time.perf_counter()
    trained_tokens = 0

    def save(epoch, next_batch):
        model.save(output / "latest", tokenizer, {**settings, "checkpoint_step": step})
        state = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "epoch": epoch, "next_batch": next_batch, "step": step, "best": best, "stale": stale,
                 "elapsed_seconds": elapsed_before + time.perf_counter() - started,
                 "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                 "train_sha": sha256(train_path), "dev_sha": sha256(dev_path)}
        temporary = output / "latest" / "trainer.tmp"
        torch.save(state, temporary)
        temporary.replace(output / "latest" / "trainer.pt")

    stopped = "epochs_complete"
    for epoch in range(start_epoch, args.epochs):
        if step >= total_steps:
            break
        generator = torch.Generator().manual_seed(args.seed + epoch)
        loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, generator=generator,
                            collate_fn=collator, num_workers=0)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_sum, epoch_tokens, group_tokens = 0.0, 0, 0
        batch_index = -1
        for batch_index, batch in enumerate(loader):
            if epoch == start_epoch and batch_index < start_batch:
                continue
            batch = {k: v.to(device) for k, v in batch.items()}
            count = int(batch["labels"].ne(-100).sum())
            with precision_context(device, args.precision):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; do not spend more compute until investigated")
            # Token-weighted accumulation, including short final batches. A plain
            # mean of microbatch losses overweights the shortest reports.
            (loss * count).backward()
            epoch_sum += float(loss.detach()) * count
            epoch_tokens += count
            group_tokens += count
            trained_tokens += count
            if (batch_index + 1) % args.accumulate == 0 or batch_index + 1 == len(loader):
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(group_tokens)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                group_tokens = 0
                step += 1
                if step % args.log_every == 0:
                    entry = {"epoch": epoch + 1, "step": step, "train_loss": epoch_sum / epoch_tokens,
                             "elapsed_seconds": elapsed_before + time.perf_counter() - started}
                    print(json.dumps(entry), flush=True)
                    append_jsonl(output / "history.jsonl", entry)
                time_up = args.max_minutes is not None and time.perf_counter() - started >= args.max_minutes * 60
                if step >= total_steps or time_up:
                    stopped = "time_budget" if time_up else "step_budget"
                    break
        dev_loss = evaluate_loss(model, dev_loader, device, args.precision)
        entry = {"epoch": epoch + 1, "step": step, "train_loss": epoch_sum / max(1, epoch_tokens),
                 "dev_loss": dev_loss, "elapsed_seconds": elapsed_before + time.perf_counter() - started}
        print(json.dumps(entry), flush=True)
        append_jsonl(output / "history.jsonl", entry)
        if dev_loss < best - args.min_delta:
            best, stale = dev_loss, 0
            model.save(output / "best", tokenizer, {**settings, "checkpoint_step": step})
        else:
            stale += 1
        finished_epoch = batch_index + 1 == len(loader)
        save(epoch + 1 if finished_epoch else epoch, 0 if finished_epoch else batch_index + 1)
        start_batch = 0
        if stale >= args.patience:
            stopped = "early_stopping"
            break
        if stopped != "epochs_complete":
            break
    synchronise(device)
    elapsed = time.perf_counter() - started
    result = {"optimizer_steps": step, "best_dev_loss": best, "stop_reason": stopped,
              "session_seconds": elapsed, "total_elapsed_seconds": elapsed_before + elapsed,
              "session_target_tokens": trained_tokens, "target_tokens_per_second": trained_tokens / max(elapsed, 1e-9),
              **memory_stats(device), **settings}
    write_json(output / "training_metrics.json", result)
    print(json.dumps({k: result[k] for k in ("optimizer_steps", "best_dev_loss", "session_seconds", "peak_allocated_mib")}))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/prepared")
    p.add_argument("--output", required=True)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--revision", default="main")
    p.add_argument("--mode", choices=["full", "lora"], required=True)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accumulate", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--source-length", type=int, default=384)
    p.add_argument("--target-length", type=int, default=256)
    p.add_argument("--dev-limit", type=int, default=256)
    p.add_argument("--train-limit", type=int)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--max-minutes", type=float)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--min-delta", type=float, default=0.001)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=3710)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--precision", choices=["auto", "bf16", "fp32"], default="auto")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p


if __name__ == "__main__":
    main(parser().parse_args())
