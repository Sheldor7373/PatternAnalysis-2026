"""Tiny random CPU T5/LoRA only; no pretrained paths, downloads or fitting."""
import copy
from dataclasses import replace
import unittest

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import T5Config, T5ForConditionalGeneration

from negation_model_copies import CheckpointLock, prepare_copies, state_fingerprint


def tiny_templates():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3710)
        config = T5Config(vocab_size=128, d_model=16, d_kv=4, d_ff=24, num_layers=4,
                          num_decoder_layers=1, num_heads=2, decoder_start_token_id=0,
                          pad_token_id=0, eos_token_id=1, dropout_rate=0.0, use_cache=False)
        base = T5ForConditionalGeneration(config).eval().requires_grad_(False)
        wrapped = get_peft_model(copy.deepcopy(base), LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM,
                    r=1, lora_alpha=1, target_modules=["q"], lora_dropout=0.0, bias="none"))
        with torch.no_grad():
            for name, parameter in wrapped.named_parameters():
                if "lora_B" in name:
                    parameter.fill_(0.01)  # Construct a nonzero random-fixture adapter; no optimisation.
        wrapped.eval().requires_grad_(False)
        merged = copy.deepcopy(wrapped).merge_and_unload(safe_merge=True)
    lock = CheckpointLock("synthetic_random_fixture", state_fingerprint(base), state_fingerprint(wrapped),
                          state_fingerprint(merged), "a"*64)
    return base, wrapped, lock


class CopyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads(); torch.set_num_threads(1)
        cls.base, cls.wrapped, cls.lock = tiny_templates()

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def prepared(self, frozen=None, wrapped=None, lock=None):
        return prepare_copies(frozen or (lambda: copy.deepcopy(self.base)),
                              wrapped or (lambda: copy.deepcopy(self.wrapped)), lock or self.lock, fixture=True)

    def test_separate_copy_merge_matches_locks_and_preserves_templates(self):
        result = self.prepared()
        self.assertIsNot(result.frozen, result.merged)
        self.assertEqual(state_fingerprint(self.wrapped), self.lock.wrapped_state_sha256)
        self.assertEqual(state_fingerprint(self.base), self.lock.frozen_state_sha256)
        self.assertEqual(result.checkpoint_ids["lora_merged_copy"], self.lock.merged_state_sha256)
        result.assert_unchanged()

    def test_reused_object_or_shared_storage_rejected_before_merge(self):
        shared = copy.deepcopy(self.wrapped)
        with self.assertRaises(ValueError): self.prepared(wrapped=lambda: shared)
        second = copy.deepcopy(self.wrapped)
        next(second.parameters()).data = next(shared.parameters()).data
        sequence = iter((shared, second))
        with self.assertRaisesRegex(ValueError, "share"): self.prepared(wrapped=lambda: next(sequence))

    def test_wrong_state_config_or_fixture_lock_rejected(self):
        with self.assertRaises(ValueError): self.prepared(lock=replace(self.lock, frozen_state_sha256="f"*64))
        changed = copy.deepcopy(self.base); changed.config.decoder_start_token_id = 2
        with self.assertRaises(ValueError): self.prepared(frozen=lambda: changed)
        with self.assertRaises(ValueError): self.prepared(lock=replace(self.lock, kind="coordinator_locked_checkpoint"))

    def test_training_gradients_or_dtype_rejected_without_silent_mutation(self):
        for change in ("training", "grad", "double"):
            model = copy.deepcopy(self.base)
            if change == "training": model.train()
            if change == "grad": next(model.parameters()).requires_grad_(True)
            if change == "double": model.double()
            with self.subTest(change=change), self.assertRaises(ValueError): self.prepared(frozen=lambda: model)
        self.assertTrue(model.shared.weight.dtype == torch.float64)

    def test_parameter_or_config_change_after_preparation_is_detected(self):
        for change in ("parameter", "config", "adapter_scaling"):
            result = self.prepared()
            if change == "parameter":
                with torch.no_grad(): next(result.merged.parameters()).add_(0.01)
            elif change == "config": result.frozen.config.decoder_start_token_id = 2
            else:
                layer = next(module for module in result.wrapped_reference.modules() if hasattr(module, "scaling"))
                layer.scaling["default"] *= 2
            with self.subTest(change=change), self.assertRaises(ValueError): result.assert_unchanged()


if __name__ == "__main__":
    unittest.main()
