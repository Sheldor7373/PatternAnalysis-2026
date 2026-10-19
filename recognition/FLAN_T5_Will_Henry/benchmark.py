"""Run bounded local pilots and sample total device memory with nvidia-smi."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from dataset import write_json


def run_pilot(command, output):
    samples = []
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            try:
                result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,temperature.gpu,power.draw",
                                         "--format=csv,noheader,nounits", "--id=0"],
                                        capture_output=True, text=True, timeout=5, check=True)
                values = result.stdout.strip().split(",")
                samples.append({"time": time.time(), "used_mib": float(values[0]),
                                "total_mib": float(values[1]), "temperature_c": float(values[2]),
                                "power_w": float(values[3])})
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


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/prepared")
    p.add_argument("--model", default="google/flan-t5-small")
    p.add_argument("--revision", default="main")
    p.add_argument("--output", default="runs/local-pilot")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--accumulate", type=int, default=2)
    args = p.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "pilot.json", {"started_utc": datetime.now(timezone.utc).isoformat(), "arguments": vars(args),
                                     "scope": "Bounded smoke/resource pilot; not final model-quality evidence"})
    for mode in ("full", "lora"):
        command = [sys.executable, "train.py", "--data", args.data, "--model", args.model,
                   "--revision", args.revision, "--mode", mode, "--output", str(root / mode),
                   "--train-limit", "32", "--dev-limit", "8", "--max-steps", str(args.steps),
                   "--batch-size", str(args.batch_size), "--accumulate", str(args.accumulate),
                   "--epochs", "1", "--log-every", "1", "--device", args.device]
        code = run_pilot(command, root / (mode + "_device.json"))
        if code:
            raise SystemExit(f"{mode} pilot failed; inspect {root / (mode + '_device.log')} before continuing")
    command = [sys.executable, "predict.py", "--manifest", str(Path(args.data) / "dev.jsonl"),
               "--model", args.model, "--revision", args.revision, "--output", str(root / "frozen"),
               "--limit", "8", "--batch-size", str(args.batch_size), "--device", args.device]
    raise SystemExit(run_pilot(command, root / "frozen_device.json"))
