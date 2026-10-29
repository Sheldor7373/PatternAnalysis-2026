"""Supervise one reviewed local run with durable logs; never auto-restart it."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from status_io import atomic_json


def now():
    return datetime.now(timezone.utc).isoformat()


def gpu_sample():
    try:
        result = subprocess.run(["nvidia-smi", "--id=0", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=True)
        values = result.stdout.strip().split(",")
        fields = ("used_mib", "total_mib", "utilization_percent", "temperature_c", "power_w")
        sample = {}
        for key, value in zip(fields, values):
            try:
                sample[key] = float(value.strip())
            except ValueError:
                sample[key] = None
        return sample
    except (OSError, subprocess.SubprocessError) as error:
        return {"sampling_error": str(error)}


def history_summary(training):
    latest = {}
    try:
        with (training / "history.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("event") == "train_window":
                    latest["train_window"] = entry
                if "dev_loss" in entry:
                    latest["development"] = entry
    except OSError:
        pass
    return latest


def run(config_path, resume=False):
    config_path = Path(config_path).resolve()
    root = config_path.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))
    training = Path(config["training_output"]).resolve()
    if not training.is_relative_to(root):
        raise ValueError("Training output must be inside this run directory")
    command = config["command"]
    if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
        raise ValueError("Expected an explicit argv list")
    for name, expected in config["source_files_sha256"].items():
        path = Path(config["code_directory"]) / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen source changed: {name}")
    if resume:
        if (training / "STOP").exists():
            raise ValueError("STOP remains present; remove it explicitly before resuming")
        command = command + ["--resume"]
    elif training.exists() and any(training.iterdir()):
        raise ValueError("Training output already exists; explicit resume is required")
    lock = (root / "supervisor.lock").open("a+b")
    if lock.tell() == 0:
        lock.write(b"0")
        lock.flush()
    lock.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    environment = os.environ.copy()
    environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    try:
        with (root / "train.stdout.log").open("a", encoding="utf-8") as stdout, \
                (root / "train.stderr.log").open("a", encoding="utf-8") as stderr:
            stdout.write(json.dumps({"supervisor_session_utc": now(), "resume": resume}) + "\n")
            stdout.flush()
            child = subprocess.Popen(command, cwd=config["code_directory"], env=environment,
                                     stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL,
                                     close_fds=True, creationflags=flags)
            identity = {"supervisor_pid": os.getpid(), "launcher_child_pid": child.pid,
                        "started_utc": now(), "resume": resume, "command": command}
            atomic_json(root / "pid.json", identity)
            last_health = 0.0
            with (root / "gpu.jsonl").open("a", encoding="utf-8") as resources, \
                    (root / "health.jsonl").open("a", encoding="utf-8") as health:
                while True:
                    code = child.poll()
                    sample = {"timestamp_utc": now(), **gpu_sample()}
                    resources.write(json.dumps(sample) + "\n")
                    resources.flush()
                    pointer = training / "latest.json"
                    status = {**identity, "observed_utc": now(), "returncode": code,
                              "state": "running" if code is None else ("completed" if code == 0 else "failed"),
                              "gpu": sample,
                              "checkpoint_age_seconds": time.time() - pointer.stat().st_mtime if pointer.exists() else None}
                    try:
                        status["training"] = json.loads((training / "status.json").read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        status["training"] = None
                    atomic_json(root / "supervisor.json", status)
                    if time.monotonic() - last_health >= config.get("health_seconds", 900) or code is not None:
                        health.write(json.dumps({**status, **history_summary(training),
                                                 "scope": "Read-only health observation; no generation or stop decision"}) + "\n")
                        health.flush()
                        last_health = time.monotonic()
                    if code is not None:
                        return code
                    time.sleep(config.get("poll_seconds", 30))
    finally:
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run(args.launch, args.resume))
