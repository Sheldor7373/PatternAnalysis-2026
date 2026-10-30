"""Fine-tune the same checkpoint with full weights or LoRA; record actual cost."""

import argparse
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, get_linear_schedule_with_warmup

from dataset import ReportDataset, read_jsonl, select_rows, write_json
from modules import MODEL, RadiologySimplifier, choose_device, load_checkpoint
from runtime import append_jsonl, environment, memory_stats, precision_context, reset_memory, set_seed, sha256, synchronise
from checkpoint_storage import publish_checkpoint, resolve_checkpoint
from status_io import atomic_json
from training_control import selection_point, update_selection, validate_best_reference
from training_transition import (RemainingBatches, checkpoint_cursor, transition_policy, lr_factor,
                                 validate_arguments, epoch_extension_policy, restore_optimizer_schedule)


def atomic_status(output, value):
    atomic_json(output / "status.json", value)


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
    if args.log_every < 1 or args.checkpoint_minutes < 0 or args.eval_minutes < 0:
        raise ValueError("log_every must be positive; monitoring intervals must be nonnegative")
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
    restoring = args.resume or args.transition_from is not None
    restarting_schedule = args.restart_schedule and not args.resume
    if args.restart_schedule and (args.transition_from is None or args.mode != 'full'):
        raise ValueError('Explicit schedule restart requires a full-model transition checkpoint')
    if args.resume and args.transition_from is not None:
        # The recorded parent path is a provenance argument on later resumes;
        # the checkpoint loaded here is always this segment's own latest.
        source_checkpoint = resolve_checkpoint(output / "latest")
    elif args.transition_from is not None:
        if not Path(args.transition_from).is_dir():
            raise ValueError("transition-from must name an immutable checkpoint directory")
        source_checkpoint = resolve_checkpoint(args.transition_from)
        if source_checkpoint.is_relative_to(output.resolve()):
            raise ValueError("Transition source must be outside the new output directory")
    elif args.resume:
        source_checkpoint = resolve_checkpoint(output / "latest")
    if restoring:
        model, tokenizer, saved_settings = load_checkpoint(source_checkpoint, device)
        original = saved_settings["arguments"]
        validate_arguments(vars(args), original, transition=not args.resume, schedule_restart=restarting_schedule)
        state = torch.load(source_checkpoint / "trainer.pt", map_location="cpu", weights_only=False)
        if saved_settings.get("checkpoint_step") != state["step"]:
            raise ValueError("Checkpoint save was interrupted: model and optimizer steps differ")
        if state["train_sha"] != sha256(train_path) or state["dev_sha"] != sha256(dev_path):
            raise ValueError("Prepared data changed since checkpoint")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
        model = RadiologySimplifier(args.model, args.revision, args.mode, args.rank).to(device)
    revision = saved_settings["revision"] if restoring else (model.model.config._commit_hash or args.revision)
    if args.gradient_checkpointing:
        model.model.gradient_checkpointing_enable()
        model.model.enable_input_require_grads()
    model.model.config.use_cache = False
    collator = DataCollatorForSeq2Seq(tokenizer, model=model.model, label_pad_token_id=-100)
    training = ReportDataset(train_rows, tokenizer, args.source_length, args.target_length)
    development = ReportDataset(dev_rows, tokenizer, args.source_length, args.target_length)
    dev_batch_size = saved_settings.get("dev_batch_size", original["batch_size"]) if restoring else args.batch_size
    dev_loader = DataLoader(development, batch_size=dev_batch_size, collate_fn=collator,
                            generator=torch.Generator().manual_seed(args.seed))
    steps_per_epoch = math.ceil(math.ceil(len(training) / args.batch_size) / args.accumulate)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                                 weight_decay=args.weight_decay)
    policy = state.get("scheduler_policy") if restoring else None
    transition = saved_settings.get("transition") if restoring else None
    if restoring and not args.resume:
        policy = (epoch_extension_policy if restarting_schedule else transition_policy)(state, original, vars(args), len(training))
        transition = {"source_checkpoint": str(source_checkpoint.resolve()),
                      "source_completion_sha256": sha256(source_checkpoint / "COMPLETE.json"),
                      "source_trainer_sha256": sha256(source_checkpoint / "trainer.pt"),
                      "source_step": state["step"], "source_arguments": original,
                      "source_next_batch": state["next_batch"],
                      "cursor": {"epoch": policy["anchor_epoch"], "next_sample": policy["anchor_offset"],
                                 "consumed_examples": policy["anchor_examples"]},
                      "previous_monitor": state.get("monitor", {}),
                      "selection_note": "Checkpoint continuation; same fixed dev rows and epoch-end selection policy"}
        if restarting_schedule:
            transition['schedule_restart'] = {'source_scheduler':state['scheduler'], 'new_policy':policy,
                'preserved':'model, Adam moments/step, Python/Torch/CUDA RNG, data cursor and original best reference',
                'changed':'one additional epoch; LR restarted then linearly decayed, no warmup; not bit-identical original schedule continuation'}
    if policy:
        total_steps = policy["planned_steps"]
        # LambdaLR.load_state_dict restores counters, not the lambda function.
        # Reconstruct the recorded callable before loading saved scheduler state.
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                    [partial(lr_factor, policy, group=i) for i in range(len(optimizer.param_groups))])
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.05), total_steps)
    settings = {"model": args.model, "revision": revision, "mode": args.mode, "rank": args.rank,
                "arguments": vars(args), "train_manifest_sha256": sha256(train_path),
                "dev_manifest_sha256": sha256(dev_path), "train_examples": len(training),
                "dev_examples": len(development), "dev_batch_size": dev_batch_size,
                **model.parameter_counts(), "environment": environment(device),
                "dev_selected_ids_sha256": hashlib.sha256("\n".join(r["id"] for r in dev_rows).encode()).hexdigest(),
                "selection_scope": "epoch-end loss on the fixed selected training-derived development rows",
                "precision_used": "bf16" if device.type == "cuda" and args.precision != "fp32"
                and torch.cuda.is_bf16_supported() else "fp32"}
    if transition:
        settings.update({"transition": transition, "scheduler_policy": policy})
    start_epoch, start_sample, step, best, stale, elapsed_before = 0, 0, 0, math.inf, 0, 0.0
    best_step = None
    monitor = {}
    if restoring:
        restore_optimizer_schedule(optimizer, scheduler, state, policy, restart=restarting_schedule)
        if policy and (optimizer.param_groups[0]["lr"] != scheduler.get_last_lr()[0]
                       or not math.isclose(optimizer.param_groups[0]["lr"],
                                          policy["base_lrs"][0] * lr_factor(policy, state["step"]),
                                          rel_tol=1e-12, abs_tol=1e-15)):
            raise ValueError("Restored next LR does not match the recorded scheduler policy")
        start_epoch, start_sample = checkpoint_cursor(state, original, len(training))
        step = state["step"]
        best, stale, elapsed_before = state["best"], state["stale"], state["elapsed_seconds"]
        published_best_step = None
        best_root = output if args.resume else Path(original["output"])
        if (best_root / "best.json").exists() or (best_root / "best").is_dir():
            best_directory = resolve_checkpoint(best_root / "best")
            published_best_step = json.loads((best_directory / "experiment.json").read_text(encoding="utf-8"))["checkpoint_step"]
        # Older checkpoints have no recorded best step. Their published best
        # must still exist when a finite best loss is recorded, and cannot be
        # ahead of the resumable state.
        best_step = state.get("best_step", published_best_step if math.isfinite(best) else None)
        validate_best_reference(best, best_step, published_best_step, step)
        if not args.resume and best_step is not None:
            publish_checkpoint(output / "best", lambda stage: shutil.copytree(best_directory, stage, dirs_exist_ok=True))
        monitor = dict(state.get("monitor", {}))
        if not args.resume:
            for key in ("window_sum", "window_tokens", "window_steps", "window_max_grad_norm"):
                monitor[key] = 0
            monitor["window_started"] = elapsed_before
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    write_json(output / "run.json", settings)
    reset_memory(device)
    started = time.perf_counter()
    trained_tokens = 0
    last_checkpoint_elapsed = monitor.get("last_checkpoint_elapsed", elapsed_before)
    last_eval_elapsed = monitor.get("last_eval_elapsed", elapsed_before)
    window_sum = monitor.get("window_sum", 0.0)
    window_tokens = monitor.get("window_tokens", 0)
    window_steps = monitor.get("window_steps", 0)
    window_max_grad_norm = monitor.get("window_max_grad_norm", 0.0)
    window_started = monitor.get("window_started", elapsed_before)
    epoch_sum, epoch_tokens = monitor.get("epoch_sum", 0.0), monitor.get("epoch_tokens", 0)

    def elapsed_total():
        return elapsed_before + time.perf_counter() - started

    def emit(entry, phase="training"):
        entry = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                 "pid": os.getpid(), **entry}
        print(json.dumps(entry), flush=True)
        append_jsonl(output / "history.jsonl", entry)
        atomic_status(output, {"state": "running", "phase": phase, "last_event": entry,
                               "checkpoint_elapsed_seconds": last_checkpoint_elapsed,
                               "checkpoint_age_training_seconds": max(0.0, elapsed_total() - last_checkpoint_elapsed),
                               "planned_optimizer_steps": total_steps})

    def save(epoch, next_sample):
        nonlocal last_checkpoint_elapsed
        last_checkpoint_elapsed = elapsed_total()
        state = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "cursor_version": 2, "epoch": epoch, "next_sample": next_sample,
                 "next_batch": math.ceil(next_sample / args.batch_size),
                 "consumed_examples": epoch * len(training) + next_sample,
                 "scheduler_policy": policy,
                 "step": step, "best": best, "best_step": best_step, "stale": stale,
                 "elapsed_seconds": elapsed_before + time.perf_counter() - started,
                 "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                 "train_sha": sha256(train_path), "dev_sha": sha256(dev_path),
                 "monitor": {"last_checkpoint_elapsed": last_checkpoint_elapsed, "last_eval_elapsed": last_eval_elapsed,
                             "window_sum": window_sum, "window_tokens": window_tokens, "window_steps": window_steps,
                             "window_max_grad_norm": window_max_grad_norm, "window_started": window_started,
                             "epoch_sum": epoch_sum, "epoch_tokens": epoch_tokens}}

        def write(stage):
            model.save(stage, tokenizer, {**settings, "checkpoint_step": step})
            torch.save(state, stage / "trainer.pt")

        publish_checkpoint(output / "latest", write)

    if restoring and not args.resume:
        # Publish the imported state before taking a single new optimizer step.
        save(start_epoch, start_sample)
        emit({"event": "transition_handoff", "step": step,
              "next_sample": start_sample, "consumed_examples": start_epoch * len(training) + start_sample,
              "source_checkpoint": transition["source_checkpoint"],
              "next_learning_rate": optimizer.param_groups[0]["lr"]}, "checkpoint")
    emit({"event": "session_start", "step": step, "elapsed_seconds": elapsed_total(),
          "dev_examples": len(development), "dev_selected_ids_sha256": settings["dev_selected_ids_sha256"]}, "starting")
    if args.eval_at_start and not restoring:
        dev_loss = evaluate_loss(model, dev_loader, device, args.precision)
        last_eval_elapsed = elapsed_total()
        emit({"event": "dev_baseline", "step": step, "dev_loss": dev_loss,
              "selection_decision": False, "dev_examples": len(development), "elapsed_seconds": elapsed_total()}, "evaluation")

    stopped = "epochs_complete"
    for epoch in range(start_epoch, args.epochs):
        if step >= total_steps:
            break
        generator = torch.Generator().manual_seed(args.seed + epoch)
        offset = start_sample if epoch == start_epoch else 0
        sampler = RandomSampler(training, generator=generator)
        loader = DataLoader(training, batch_sampler=RemainingBatches(sampler, args.batch_size, offset),
                            generator=generator, collate_fn=collator, num_workers=0)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if offset == 0:
            epoch_sum, epoch_tokens = 0.0, 0
        group_tokens = 0
        batch_index = -1
        next_sample = offset
        for batch_index, batch in enumerate(loader):
            next_sample = min(len(training), offset + (batch_index + 1) * args.batch_size)
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
            window_sum += float(loss.detach()) * count
            window_tokens += count
            group_tokens += count
            trained_tokens += count
            if (batch_index + 1) % args.accumulate == 0 or batch_index + 1 == len(loader):
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(group_tokens)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True))
                lr_used = optimizer.param_groups[0]["lr"]
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                group_tokens = 0
                step += 1
                window_steps += 1
                window_max_grad_norm = max(window_max_grad_norm, grad_norm)
                if step % args.log_every == 0:
                    duration = max(elapsed_total() - window_started, 1e-9)
                    emit({"event": "train_window", "epoch": epoch + 1, "step": step,
                          "epoch_progress": next_sample / len(training),
                          "consumed_examples": epoch * len(training) + next_sample,
                          "train_loss": epoch_sum / epoch_tokens, "train_loss_scope": "token-weighted cumulative epoch",
                          "window_train_loss": window_sum / max(1, window_tokens), "window_target_tokens": window_tokens,
                          "window_optimizer_steps": window_steps, "window_seconds": duration,
                          "window_target_tokens_per_second": window_tokens / duration,
                          "learning_rate": lr_used, "next_learning_rate": optimizer.param_groups[0]["lr"],
                          "grad_norm_preclip": grad_norm, "window_max_grad_norm_preclip": window_max_grad_norm,
                          "finite_loss": True, "finite_gradients": True,
                          "elapsed_seconds": elapsed_total(), **memory_stats(device)})
                    window_sum, window_tokens, window_steps, window_max_grad_norm = 0.0, 0, 0, 0.0
                    window_started = elapsed_total()
                time_up = args.max_minutes is not None and time.perf_counter() - started >= args.max_minutes * 60
                requested = (output / "STOP").exists()
                if step >= total_steps or time_up or requested:
                    stopped = "stop_requested" if requested else ("time_budget" if time_up else "step_budget")
                    break
                if batch_index + 1 < len(loader):
                    if args.checkpoint_minutes and elapsed_total() - last_checkpoint_elapsed >= args.checkpoint_minutes * 60:
                        save(epoch, next_sample)
                        emit({"event": "periodic_checkpoint", "epoch": epoch + 1, "step": step,
                              "next_sample": next_sample, "elapsed_seconds": elapsed_total()}, "checkpoint")
                    if args.eval_minutes and elapsed_total() - last_eval_elapsed >= args.eval_minutes * 60:
                        dev_loss = evaluate_loss(model, dev_loader, device, args.precision)
                        last_eval_elapsed = elapsed_total()
                        emit({"event": "dev_diagnostic", "epoch": epoch + 1, "step": step, "dev_loss": dev_loss,
                              "selection_decision": False, "dev_examples": len(development),
                              "elapsed_seconds": elapsed_total()}, "evaluation")
                        model.train()
        finished_epoch = batch_index + 1 == len(loader)
        eligible = selection_point(finished_epoch, step, total_steps)
        dev_loss = evaluate_loss(model, dev_loader, device, args.precision)
        last_eval_elapsed = elapsed_total()
        entry = {"event": "epoch_evaluation" if finished_epoch else "session_evaluation",
                 "epoch": epoch + 1, "step": step, "train_loss": epoch_sum / max(1, epoch_tokens),
                 "train_loss_scope": "token-weighted cumulative epoch",
                 "dev_loss": dev_loss, "selection_decision": eligible,
                 "dev_examples": len(development), "elapsed_seconds": elapsed_total()}
        emit(entry, "evaluation")
        best, stale, improved = update_selection(best, stale, dev_loss, eligible, args.min_delta)
        if improved:
            best_step = step
            publish_checkpoint(output / "best",
                               lambda stage: model.save(stage, tokenizer, {**settings, "checkpoint_step": step}))
        save(epoch + 1 if finished_epoch else epoch, 0 if finished_epoch else next_sample)
        start_sample = 0
        if eligible and stale >= args.patience:
            stopped = "early_stopping"
            break
        if stopped != "epochs_complete":
            break
    synchronise(device)
    elapsed = time.perf_counter() - started
    result = {"optimizer_steps": step, "best_dev_loss": best if math.isfinite(best) else None, "stop_reason": stopped,
              "session_seconds": elapsed, "total_elapsed_seconds": elapsed_before + elapsed,
              "session_target_tokens": trained_tokens, "target_tokens_per_second": trained_tokens / max(elapsed, 1e-9),
              **memory_stats(device), **settings}
    write_json(output / "training_metrics.json", result)
    atomic_status(output, {"state": "stopped" if stopped in ("time_budget", "stop_requested") else "completed",
                           "pid": os.getpid(), "step": step, "stop_reason": stopped,
                           "elapsed_seconds": elapsed_total(), "checkpoint_elapsed_seconds": last_checkpoint_elapsed})
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
    p.add_argument("--checkpoint-minutes", type=float, default=0.0)
    p.add_argument("--eval-minutes", type=float, default=0.0)
    p.add_argument("--eval-at-start", action="store_true")
    p.add_argument("--seed", type=int, default=3710)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--precision", choices=["auto", "bf16", "fp32"], default="auto")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--transition-from", help="Immutable local checkpoint to continue into a fresh output segment")
    p.add_argument('--restart-schedule', action='store_true', help='Explicit one-epoch full-model extension from a completed source; preserve Adam/RNG but restart LR')
    return p


if __name__ == "__main__":
    main(parser().parse_args())
