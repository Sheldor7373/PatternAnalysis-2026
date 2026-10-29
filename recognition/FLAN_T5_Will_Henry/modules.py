"""FLAN-T5 with optional attention adapters. Model equations: Raffel et al.; Hu et al."""

from pathlib import Path
import json
import torch
from torch import nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from checkpoint_storage import resolve_checkpoint

MODEL = "google/flan-t5-small"


class RadiologySimplifier(nn.Module):
    def __init__(self, model_name=MODEL, revision="main", mode="full", rank=8, checkpoint=None):
        super().__init__()
        if mode not in {"full", "lora", "frozen"}:
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode
        if checkpoint and mode != "lora":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, revision=revision)
        if mode == "lora":
            from peft import LoraConfig, PeftModel, TaskType, get_peft_model
            if checkpoint:
                self.model = PeftModel.from_pretrained(self.model, checkpoint, is_trainable=True)
            else:
                config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=rank, lora_alpha=rank * 2,
                                    lora_dropout=0.05, target_modules=["q", "v"], bias="none")
                self.model = get_peft_model(self.model, config)
        if mode == "frozen":
            self.requires_grad_(False)

    def forward(self, **batch):
        return self.model(**batch)

    def parameter_counts(self):
        return {"total_parameters": sum(p.numel() for p in self.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)}

    def save(self, path, tokenizer, settings):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path, safe_serialization=True)
        tokenizer.save_pretrained(path)
        (path / "experiment.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")


def load_checkpoint(path, device, base_source=None):
    path = resolve_checkpoint(path)
    settings = json.loads((path / "experiment.json").read_text(encoding="utf-8"))
    model_name, revision = settings["model"], settings["revision"]
    if base_source is not None:
        if settings["mode"] != "lora":
            raise ValueError("A base-source binding applies only to adapter checkpoints")
        model_name, revision = base_source
    model = RadiologySimplifier(model_name, revision, settings["mode"],
                               settings.get("rank", 8), checkpoint=path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer, settings


def choose_device(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but PyTorch cannot use the GPU")
    return torch.device(name)
