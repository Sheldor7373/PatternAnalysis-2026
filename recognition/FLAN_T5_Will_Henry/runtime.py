"""Small helpers shared by training and prediction."""

from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import subprocess
import torch


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def precision_context(device, precision):
    if precision == "bf16":
        if device.type != "cuda" or not torch.cuda.is_bf16_supported():
            raise ValueError("bf16 requested on unsupported device")
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if precision == "auto" and device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def synchronise(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_memory(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def memory_stats(device):
    if device.type != "cuda":
        return {"peak_allocated_mib": None, "peak_reserved_mib": None}
    return {"peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment(device):
    packages = {}
    for name in ("torch", "transformers", "peft", "accelerate", "numpy", "rouge-score", "huggingface-hub"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                         text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
            "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cuda": torch.version.cuda, "git_commit": commit, "git_dirty": dirty,
            "bf16_available": device.type == "cuda" and torch.cuda.is_bf16_supported()}


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value) + "\n")
