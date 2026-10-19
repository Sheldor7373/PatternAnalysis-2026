"""Offline checks with a tiny random T5: masking, adapter gradients and reload."""

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from transformers import T5Config, T5ForConditionalGeneration, DataCollatorForSeq2Seq, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from dataset import ReportDataset
from modules import RadiologySimplifier, load_checkpoint
from checkpoint_storage import mark_complete


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        scratch = Path(__file__).parent / "runs" / "tests"
        scratch.mkdir(parents=True, exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(dir=scratch)
        cls.path = Path(cls.temp.name)
        base = Tokenizer(WordLevel({"<pad>": 0, "</s>": 1, "<unk>": 2,
                                    "No": 3, "effusion": 4, "Fluid": 5, "absent": 6}, unk_token="<unk>"))
        base.pre_tokenizer = Whitespace()
        cls.tokenizer = PreTrainedTokenizerFast(tokenizer_object=base, pad_token="<pad>",
                                               eos_token="</s>", unk_token="<unk>")
        cls.tokenizer.save_pretrained(cls.path)
        config = T5Config(vocab_size=7, d_model=16, d_kv=8, d_ff=32, num_layers=1, num_decoder_layers=1,
                          num_heads=2, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1, dropout_rate=0.0)
        T5ForConditionalGeneration(config).save_pretrained(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def batch(self, model):
        rows = [{"input": "No effusion", "target": "Fluid absent"},
                {"input": "No", "target": "absent"}]
        ds = ReportDataset(rows, self.tokenizer)
        collator = DataCollatorForSeq2Seq(self.tokenizer, model=model.model, label_pad_token_id=-100)
        return collator([ds[0], ds[1]])

    def test_padding_does_not_contribute_to_loss(self):
        model = RadiologySimplifier(str(self.path))
        model.eval()
        batch = self.batch(model)
        self.assertEqual(int(batch["labels"].eq(-100).sum()), 1)
        result = model(**batch)
        manual = torch.nn.functional.cross_entropy(result.logits.flatten(0, 1), batch["labels"].flatten(), ignore_index=-100)
        torch.testing.assert_close(result.loss, manual)
        self.assertTrue(torch.isfinite(result.loss))

    def test_only_adapters_receive_gradients_and_reload_exactly(self):
        model = RadiologySimplifier(str(self.path), mode="lora", rank=2)
        batch = self.batch(model)
        loss = model(**batch).loss
        loss.backward()
        self.assertTrue(any(p.grad is not None for n, p in model.named_parameters() if "lora_" in n))
        self.assertFalse(any(p.grad is not None for n, p in model.named_parameters() if "lora_" not in n))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
        optimizer.step()
        model.eval()
        expected = model(**batch).logits.detach()
        with tempfile.TemporaryDirectory(dir=self.path) as folder:
            settings = {"model": str(self.path), "revision": "main", "mode": "lora", "rank": 2}
            model.save(folder, self.tokenizer, settings)
            mark_complete(folder)
            restored, _, _ = load_checkpoint(folder, torch.device("cpu"))
            restored.eval()
            torch.testing.assert_close(restored(**batch).logits, expected)

    def test_frozen_model_has_no_trainable_parameters(self):
        model = RadiologySimplifier(str(self.path), mode="frozen")
        self.assertEqual(model.parameter_counts()["trainable_parameters"], 0)

    def test_full_checkpoint_loads_without_the_original_base(self):
        from transformers import AutoModelForSeq2SeqLM
        model = RadiologySimplifier(str(self.path), mode="full")
        model.eval()
        batch = self.batch(model)
        expected = model(**batch).logits.detach()
        with tempfile.TemporaryDirectory(dir=self.path) as folder:
            model.save(folder, self.tokenizer,
                       {"model": "base-no-longer-available", "revision": "pinned-sha", "mode": "full"})
            mark_complete(folder)
            original_load = AutoModelForSeq2SeqLM.from_pretrained

            def checkpoint_only(path, *args, **kwargs):
                self.assertEqual(Path(path), Path(folder))
                return original_load(path, *args, **kwargs)

            with patch("modules.AutoModelForSeq2SeqLM.from_pretrained", side_effect=checkpoint_only) as loader:
                restored, tokenizer, settings = load_checkpoint(folder, torch.device("cpu"))
                self.assertEqual(loader.call_count, 1)
            restored.eval()
            torch.testing.assert_close(restored(**batch).logits, expected)
            self.assertEqual(settings["revision"], "pinned-sha")
            self.assertEqual(tokenizer.pad_token_id, self.tokenizer.pad_token_id)


if __name__ == "__main__":
    unittest.main()
