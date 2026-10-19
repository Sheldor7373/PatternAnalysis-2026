"""Fresh CPU copy preparation with predeclared state/config locks; no loaders.

Caller-supplied factories must return fresh immutable checkpoint copies.
This module never loads a path, downloads, touches a live training object or
chooses a checkpoint. Only the disposable second wrapped copy is merged.
"""
from dataclasses import dataclass
import hashlib
import json

import torch
from transformers import T5ForConditionalGeneration

from negation_summary import sha


def state_fingerprint(model):
    """Hash complete named state plus configuration, without a model forward."""
    peft_config = getattr(model, "peft_config", {})
    adapter_runtime = {}
    for name, module in model.named_modules():
        if hasattr(module, "merged_adapters") and hasattr(module, "disable_adapters"):
            adapter_runtime[name] = {"disabled": module.disable_adapters, "merged": module.merged_adapters,
                                     "scaling": getattr(module, "scaling", {}), "r": getattr(module, "r", {}),
                                     "alpha": getattr(module, "lora_alpha", {})}
    semantics = {"config": model.config.to_dict(), "attention_implementation": model.config._attn_implementation,
                 "peft_config": {name: config.to_dict() for name, config in peft_config.items()},
                 "active_adapters": model.active_adapters if peft_config else None, "adapter_runtime": adapter_runtime}
    result = hashlib.sha256(json.dumps(semantics, sort_keys=True, allow_nan=False,
                                       default=str).encode())
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        result.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        result.update(tensor.view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def frozen_float32_cpu(model):
    if any(module.training for module in model.modules()) or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError("Fresh copies must already be frozen/eval with no gradients")
    for value in list(model.parameters()) + list(model.buffers()):
        if value.device.type != "cpu" or (value.is_floating_point() and value.dtype != torch.float32):
            raise ValueError("CPU float32 copies required; no device/dtype movement is performed")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError("Nonfinite checkpoint state")


def disjoint_storage(*models):
    seen = set()
    for model in models:
        pointers = {value.untyped_storage().data_ptr() for value in list(model.parameters()) + list(model.buffers()) if value.numel()}
        if pointers & seen:
            raise ValueError("Fresh model copies must not share parameter/buffer storage")
        seen.update(pointers)


@dataclass(frozen=True)
class CheckpointLock:
    kind: str
    frozen_state_sha256: str
    wrapped_state_sha256: str
    merged_state_sha256: str
    selection_receipt_sha256: str

    def validate(self, fixture):
        expected = "synthetic_random_fixture" if fixture else "coordinator_locked_checkpoint"
        if type(fixture) is not bool or self.kind != expected:
            raise ValueError("Real checkpoint locks and synthetic random fixtures must be explicitly separate")
        for value in (self.frozen_state_sha256, self.wrapped_state_sha256, self.merged_state_sha256, self.selection_receipt_sha256):
            sha(value)


@dataclass(frozen=True)
class PreparedCopies:
    frozen: object
    wrapped_reference: object
    merged: object
    lock: CheckpointLock
    fixture: bool

    def assert_unchanged(self):
        for model, expected in ((self.frozen, self.lock.frozen_state_sha256),
                                (self.wrapped_reference, self.lock.wrapped_state_sha256),
                                (self.merged, self.lock.merged_state_sha256)):
            frozen_float32_cpu(model)
            if state_fingerprint(model) != expected:
                raise ValueError("Immutable checkpoint state/config changed")
        disjoint_storage(self.frozen, self.wrapped_reference, self.merged)

    @property
    def checkpoint_ids(self):
        return {"frozen": self.lock.frozen_state_sha256, "lora_merged_copy": self.lock.merged_state_sha256}


def prepare_copies(load_frozen, load_wrapped, lock, *, fixture=False):
    """Call fresh-copy factories1+2 times, validate locks, merge copy2 safely.

    No forwards or automatic eval/freeze/device changes. Real factories and
    precomputed lock identities require separate authorisation before execution.
    """
    from peft import PeftModelForSeq2SeqLM
    lock.validate(fixture)
    frozen, reference, disposable = load_frozen(), load_wrapped(), load_wrapped()
    if type(frozen) is not T5ForConditionalGeneration or not all(isinstance(m, PeftModelForSeq2SeqLM) for m in (reference, disposable)):
        raise TypeError("Require one plain T5 and two fresh seq2seq PEFT LoRA copies")
    if len({id(frozen), id(reference), id(disposable)}) != 3:
        raise ValueError("Factories returned the same object instead of fresh copies")
    disjoint_storage(frozen, reference, disposable)
    for model, expected in ((frozen, lock.frozen_state_sha256), (reference, lock.wrapped_state_sha256),
                            (disposable, lock.wrapped_state_sha256)):
        frozen_float32_cpu(model)
        if state_fingerprint(model) != expected:
            raise ValueError("Loaded copy differs from the predeclared checkpoint/config lock")
    for wrapped in (reference, disposable):
        if len(wrapped.peft_config) != 1 or any(str(config.peft_type) not in ("LORA", "PeftType.LORA") for config in wrapped.peft_config.values()):
            raise ValueError("Exactly one fixed LoRA adapter is required")
        if wrapped.active_adapters != list(wrapped.peft_config) or any(
                module.disable_adapters or module.merged_adapters for module in wrapped.modules()
                if hasattr(module, "merged_adapters") and hasattr(module, "disable_adapters")):
            raise ValueError("The one LoRA adapter must be active and initially unmerged")
    merged = disposable.merge_and_unload(safe_merge=True)
    if type(merged) is not T5ForConditionalGeneration:
        raise TypeError("Merged disposable copy must be plain T5")
    result = PreparedCopies(frozen, reference, merged, lock, fixture)
    result.assert_unchanged()  # Also verifies that merge did not alter reference/frozen.
    return result
