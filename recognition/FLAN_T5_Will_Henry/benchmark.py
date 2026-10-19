"""Run bounded local pilots and sample total device memory with nvidia-smi."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from dataset import PROMPT, read_jsonl, select_rows, write_json, write_jsonl


def parse_telemetry(line):
    values = [value.strip() for value in line.strip().split(",")]
    if len(values) != 4:
        raise ValueError("Expected four GPU telemetry fields")

    def optional(value):
        try:
            return float(value)
        except ValueError:
            return None

    return {"used_mib": float(values[0]), "total_mib": float(values[1]),
            "temperature_c": optional(values[2]), "power_w": optional(values[3])}


def run_pilot(command, output):
    samples = []
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            try:
                result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,temperature.gpu,power.draw",
                                         "--format=csv,noheader,nounits", "--id=0"],
                                        capture_output=True, text=True, timeout=5, check=True)
                samples.append({"time": time.time(), **parse_telemetry(result.stdout)})
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                pass
            stop.wait(0.5)

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    started = time.perf_counter()
    try:
        with Path(output).with_suffix(".log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    finally:
        stop.set()
        monitor.join(timeout=6)
    report = {"command": command, "returncode": result.returncode,
              "elapsed_seconds": time.perf_counter() - started,
              "sampled_device_peak_used_mib": max((r["used_mib"] for r in samples), default=None),
              "gpu_samples": samples,
              "note": "Sampled total device use includes other processes; 0.5s sampling may miss brief peaks."
                      " PyTorch allocated/reserved peaks are recorded separately by each run."}
    write_json(output, report)
    return result.returncode


def long_case_indices(source_lengths, target_lengths, source_limit=384, target_limit=256):
    if not source_lengths or len(source_lengths) != len(target_lengths):
        raise ValueError("Need matching nonempty source/target lengths")
    source = max(range(len(source_lengths)), key=lambda i: source_lengths[i])
    target = max(range(len(target_lengths)), key=lambda i: target_lengths[i])
    joint = max(range(len(source_lengths)),
                key=lambda i: min(source_lengths[i] / source_limit, target_lengths[i] / target_limit))
    return sorted({source, target, joint})


def prepare_pilot_data(data, output, tokenizer):
    pool = select_rows(read_jsonl(Path(data) / "train.jsonl"), 1024)
    source_lengths = [len(ids) for ids in tokenizer([PROMPT + r["input"] for r in pool],
                                                   truncation=False, verbose=False)["input_ids"]]
    target_lengths = [len(ids) for ids in tokenizer([r["target"] for r in pool],
                                                   truncation=False, verbose=False)["input_ids"]]
    selected = long_case_indices(source_lengths, target_lengths)
    output.mkdir(parents=True)
    write_jsonl(output / "train.jsonl", [pool[i] for i in selected])
    write_jsonl(output / "dev.jsonl", read_jsonl(Path(data) / "dev.jsonl"))
    return {"candidate_rows": len(pool), "seed": 3710,
            "selection": "Longest source, longest target, and longest joint relative length in a deterministic pool",
            "selected": [{"id": pool[i]["id"], "source_tokens": source_lengths[i], "target_tokens": target_lengths[i],
                           "effective_source_tokens": min(384, source_lengths[i]),
                           "effective_target_tokens": min(256, target_lengths[i])} for i in selected],
            "limitation": "Representative long cases within a sampled pool; not a measured full-data or theoretical memory maximum."}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/prepared")
    p.add_argument("--model", default="google/flan-t5-small")
    p.add_argument("--revision", default="main")
    p.add_argument("--output", default="runs/local-pilot")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--accumulate", type=int, default=2)
    args = p.parse_args()
    root = Path(args.output)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Choose a fresh pilot output directory")
    root.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    pilot_data = root / "pilot-data"
    selection = prepare_pilot_data(args.data, pilot_data, tokenizer)
    write_json(root / "pilot.json", {"started_utc": datetime.now(timezone.utc).isoformat(), "arguments": vars(args),
                                     "scope": "Bounded smoke/resource pilot; not final model-quality evidence",
                                     "long_case_selection": selection,
                                     "training_passes": 2,
                                     "reason": "Repeat every selected long case after AdamW state is materialised in the first pass."})
    for mode in ("full", "lora"):
        command = [sys.executable, "train.py", "--data", str(pilot_data), "--model", args.model,
                   "--revision", args.revision, "--mode", mode, "--output", str(root / mode),
                   "--dev-limit", "8",
                   "--batch-size", str(args.batch_size), "--accumulate", str(args.accumulate),
                   "--epochs", "2", "--log-every", "1", "--device", args.device]
        code = run_pilot(command, root / (mode + "_device.json"))
        if code:
            raise SystemExit(f"{mode} pilot failed; inspect {root / (mode + '_device.log')} before continuing")
    for mode in ("frozen", "full", "lora"):
        command = [sys.executable, "predict.py", "--manifest", str(Path(args.data) / "dev.jsonl"),
                   "--output", str(root / (mode + "-inference")), "--limit", "8",
                   "--batch-size", str(args.batch_size), "--device", args.device,
                   "--source-length", "384", "--target-length", "256", "--beams", "1", "--seed", "3710"]
        if mode == "frozen":
            command += ["--model", args.model, "--revision", args.revision]
        else:
            command += ["--checkpoint", str(root / mode / "latest")]
        code = run_pilot(command, root / (mode + "_inference_device.json"))
        if code:
            raise SystemExit(f"{mode} inference failed; inspect its log before comparing costs")
