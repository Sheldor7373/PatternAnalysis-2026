"""CPU synthetic tensors and tiny random T5 only; no pretrained assets."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import torch
from transformers import T5Config, T5ForConditionalGeneration

from t5_intervention_adapter import (SourceTokens, Candidate, T5SpanMeanAdapter,
                                     span_mean_shift, candidate_logits, score_candidate)

# Float32 integration tolerance; constructed array algebra uses float64 below.
ATOL, RTOL = 1e-6, 1e-6


class T5AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(3710)
            config = T5Config(vocab_size=32, d_model=16, d_kv=4, d_ff=24,
                              num_layers=2, num_decoder_layers=1, num_heads=2,
                              decoder_start_token_id=0, pad_token_id=0, eos_token_id=1,
                              dropout_rate=0.0, use_cache=False)
            self.model = T5ForConditionalGeneration(config).eval().requires_grad_(False)
        self.recipient = SourceTokens(torch.tensor([[3, 4, 5, 6, 1, 0]]),
                                      torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.bool),
                                      torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.bool))
        self.donor = SourceTokens(torch.tensor([[7, 8, 9, 1]]), torch.ones((1, 4), dtype=torch.bool),
                                  torch.tensor([[1, 1, 1, 0]], dtype=torch.bool))
        self.q = torch.eye(16)[:, :2]
        self.a = Candidate(torch.tensor([[10, 11, 1, 0]]), torch.tensor([[1, 1, 1, 0]], dtype=torch.bool), True)
        self.b = Candidate(torch.tensor([[12, 1, 0, 0]]), torch.tensor([[1, 1, 0, 0]], dtype=torch.bool), True)

    def adapter(self, site="encoder.block.0"):
        return T5SpanMeanAdapter(self.model, site, self.q)

    def encode(self):
        return self.adapter().encode_pair(self.recipient, self.recipient,
                                          recipient_span=(1, 3), donor_span=(1, 3))[0]

    def compare(self, adapter=None):
        return (self.adapter() if adapter is None else adapter).compare(
            self.recipient, self.donor, self.a, self.b, recipient_span=(1, 3), donor_span=(0, 3))

    def test_unequal_span_array_shift_preserves_differences_complement_and_inputs(self):
        rec = torch.tensor([[2., 1., 5.], [4., -1., 7.], [9., 8., 7.]], dtype=torch.float64)
        donor = torch.tensor([[10., 3., 2.], [4., 5., 8.], [7., 9., -1.], [3., 6., 2.]], dtype=rec.dtype)
        q = torch.tensor([[1.], [1.], [0.]], dtype=rec.dtype) / 2 ** .5
        before = [value.clone() for value in (rec, donor, q)]
        shifted = span_mean_shift(rec, donor, q, recipient_span=(0, 2), donor_span=(1, 4),
                                  recipient_mask=torch.ones(3, dtype=torch.bool), donor_mask=torch.ones(4, dtype=torch.bool))
        torch.testing.assert_close(shifted[:2].mean(0) @ q, donor[1:4].mean(0) @ q, atol=1e-12, rtol=0)
        torch.testing.assert_close(shifted[1] - shifted[0], rec[1] - rec[0], atol=1e-12, rtol=0)
        complement = torch.eye(3, dtype=rec.dtype) - q @ q.T
        torch.testing.assert_close(shifted[:2] @ complement, rec[:2] @ complement, atol=1e-12, rtol=0)
        torch.testing.assert_close(shifted[2], rec[2], atol=0, rtol=0)
        for actual, expected in zip((rec, donor, q), before):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        identity = span_mean_shift(rec, rec, q, recipient_span=(0, 2), donor_span=(0, 2),
                                   recipient_mask=torch.ones(3, dtype=torch.bool), donor_mask=torch.ones(3, dtype=torch.bool))
        torch.testing.assert_close(identity, rec, atol=0, rtol=0)

    def test_real_t5_self_patch_preserves_logits_margins_and_all_parameters(self):
        before = {name: value.detach().clone() for name, value in self.model.state_dict().items()}
        for site in ("encoder.block.0", "encoder.block.1"):
            adapter = self.adapter(site)
            baseline, shifted = adapter.encode_pair(self.recipient, self.recipient,
                                                     recipient_span=(1, 3), donor_span=(1, 3))
            left = candidate_logits(self.model, baseline, self.recipient.attention_mask, self.a)
            right = candidate_logits(self.model, shifted, self.recipient.attention_mask, self.a)
            torch.testing.assert_close(left, right, atol=ATOL, rtol=RTOL)
            result = adapter.compare(self.recipient, self.recipient, self.a, self.b,
                                     recipient_span=(1, 3), donor_span=(1, 3))
            self.assertAlmostEqual(result["effect"], 0., delta=ATOL)
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)
        self.assertFalse(any(p.requires_grad or p.grad is not None for p in self.model.parameters()))
        self.assertFalse(any(m.training for m in self.model.modules()))

    def test_actual_block_patch_uses_unequal_span_mean_and_preserves_unrelated_hooks(self):
        outputs, following_inputs = [], []
        parameters_before = {name: value.detach().clone() for name, value in self.model.named_parameters()}
        block = self.model.encoder.block[0]
        observer = block.register_forward_hook(lambda module, args, result: outputs.append(result[0].clone()))
        following = self.model.encoder.block[1].register_forward_pre_hook(
            lambda module, args: following_inputs.append(args[0].clone()))
        hooks_before = dict(block._forward_hooks)
        try:
            result = self.compare()
            self.assertEqual(block._forward_hooks, hooks_before)
            self.assertEqual(len(outputs), 3)  # donor capture, baseline, patched
            expected = span_mean_shift(outputs[1][0], outputs[0][0], self.q,
                                        recipient_span=(1, 3), donor_span=(0, 3),
                                        recipient_mask=self.recipient.eligible_mask[0], donor_mask=self.donor.eligible_mask[0])
            torch.testing.assert_close(following_inputs[2][0], expected, atol=ATOL, rtol=RTOL)
            self.assertGreater(float((expected - outputs[1][0]).abs().max()), ATOL)
            self.assertEqual(result["baseline"]["candidate_a"]["token_count"], 3)
            self.assertEqual(result["baseline"]["candidate_b"]["token_count"], 2)
            self.assertTrue(result["patched"]["candidate_a"]["eos_included"])
            for name, value in self.model.named_parameters():
                torch.testing.assert_close(value, parameters_before[name], atol=0, rtol=0)
        finally:
            observer.remove()
            following.remove()

    def test_candidate_sums_match_manual_gathers_and_start_pad_stays_visible(self):
        encoded = self.encode()
        seen = []
        def observe_decoder(module, args, kwargs):
            seen.append((kwargs["input_ids"].clone(), kwargs["attention_mask"].clone()))
        hook = self.model.decoder.register_forward_pre_hook(observe_decoder, with_kwargs=True)
        try:
            logits = candidate_logits(self.model, encoded, self.recipient.attention_mask, self.b)
            score = score_candidate(self.model, encoded, self.recipient.attention_mask, self.b)
        finally:
            hook.remove()
        manual = logits.double().log_softmax(-1)[0, :2].gather(1, self.b.input_ids[0, :2, None]).sum()
        self.assertAlmostEqual(score["summed_logprob"], float(manual), places=12)
        self.assertEqual(score["token_count"], 2)
        self.assertIn("no length normalisation", score["aggregation"])
        for ids, mask in seen:
            self.assertEqual(int(ids[0, 0]), self.model.config.pad_token_id)
            self.assertTrue(bool(mask[0, 0]))
            self.assertTrue(torch.equal(mask, torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)))

    def test_masked_padding_values_do_not_change_sum_and_eos_policy_is_explicit(self):
        encoded = self.encode()
        short = Candidate(torch.tensor([[10, 11, 1]]), torch.ones((1, 3), dtype=torch.bool), True)
        padded = Candidate(torch.tensor([[10, 11, 1, 17, 31]]),
                           torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool), True)
        a = score_candidate(self.model, encoded, self.recipient.attention_mask, short)
        b = score_candidate(self.model, encoded, self.recipient.attention_mask, padded)
        self.assertAlmostEqual(a["summed_logprob"], b["summed_logprob"], delta=ATOL)
        self.assertEqual(a["token_count"], b["token_count"])
        valid_pad_id = Candidate(torch.tensor([[0, 10]]), torch.ones((1, 2), dtype=torch.bool), False)
        result = score_candidate(self.model, encoded, self.recipient.attention_mask, valid_pad_id)
        self.assertEqual(result["token_count"], 2)  # masks, not token ID, define valid positions
        self.assertFalse(result["eos_included"])

    def test_margin_is_difference_of_sums_and_identical_candidates_give_zero(self):
        result = self.compare()
        for phase in ("baseline", "patched"):
            score = result[phase]
            self.assertEqual(score["margin"], score["candidate_a"]["summed_logprob"] - score["candidate_b"]["summed_logprob"])
        identical = self.adapter().compare(self.recipient, self.donor, self.a, self.a,
                                           recipient_span=(1, 3), donor_span=(0, 3))
        self.assertEqual(identical["baseline"]["margin"], 0.)
        self.assertEqual(identical["patched"]["margin"], 0.)

    def test_hook_cleanup_after_capture_patch_and_decoder_exceptions(self):
        block, adapter = self.model.encoder.block[0], self.adapter()
        observer = block.register_forward_hook(lambda module, args, output: None)
        hooks_before = dict(block._forward_hooks)
        try:
            with patch.object(block, "forward", side_effect=RuntimeError("capture failure")):
                with self.assertRaisesRegex(RuntimeError, "capture failure"):
                    self.compare(adapter)
            self.assertEqual(block._forward_hooks, hooks_before)
            with patch("t5_intervention_adapter.span_mean_shift", side_effect=RuntimeError("patch failure")):
                with self.assertRaisesRegex(RuntimeError, "patch failure"):
                    self.compare(adapter)
            self.assertEqual(block._forward_hooks, hooks_before)
            with patch.object(self.model.decoder, "forward", side_effect=RuntimeError("decoder failure")):
                with self.assertRaisesRegex(RuntimeError, "decoder failure"):
                    self.compare(adapter)
            self.assertEqual(block._forward_hooks, hooks_before)
            self.compare(adapter)  # model remains usable after all failures
        finally:
            observer.remove()

    def test_source_site_basis_and_model_state_validation(self):
        for site in ("encoder.block.99", "decoder.block.0", "encoder.block.0.layer.0", ""):
            with self.subTest(site=site), self.assertRaises(ValueError):
                self.adapter(site)
        bad_sources = (replace(self.recipient, attention_mask=self.recipient.attention_mask.long()),
                       replace(self.recipient, eligible_mask=torch.ones((1, 6), dtype=torch.bool)),
                       replace(self.recipient, input_ids=torch.full((1, 6), 32)),
                       replace(self.recipient, input_ids=self.recipient.input_ids.float()))
        for source in bad_sources:
            with self.assertRaises(ValueError):
                self.adapter().encode_pair(source, self.donor, recipient_span=(1, 3), donor_span=(0, 3))
        for span in ((-1, 2), (2, 2), (0, 7), (False, 2), (3, 5)):
            with self.assertRaises(ValueError):
                self.adapter().encode_pair(self.recipient, self.donor, recipient_span=span, donor_span=(0, 3))
        for basis in (torch.ones((16, 2)), self.q.double(), self.q.clone().requires_grad_(),
                      torch.full((16, 1), float("nan"))):
            with self.assertRaises(ValueError):
                T5SpanMeanAdapter(self.model, "encoder.block.0", basis)
        self.model.train()
        with self.assertRaisesRegex(ValueError, "eval"):
            self.adapter()
        self.assertTrue(self.model.training)  # adapter never silently changes state
        self.model.eval().requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "frozen"):
            self.adapter()
        self.assertTrue(next(self.model.parameters()).requires_grad)

    def test_invalid_candidate_eos_masks_ids_and_nonfinite_arrays_fail(self):
        encoded = self.encode()
        for candidate in (replace(self.a, eos_included=False), replace(self.b, eos_included=1),
                          replace(self.a, target_mask=torch.tensor([[1, 0, 1, 0]], dtype=torch.bool)),
                          replace(self.a, target_mask=torch.zeros((1, 4), dtype=torch.bool)),
                          replace(self.a, input_ids=torch.tensor([[1, 10, 11, 0]])),
                          replace(self.a, input_ids=torch.full((1, 4), 32))):
            with self.assertRaises((ValueError, TypeError)):
                score_candidate(self.model, encoded, self.recipient.attention_mask, candidate)
        with self.assertRaises(ValueError):
            span_mean_shift(torch.full((2, 16), float("inf")), torch.zeros((3, 16)), self.q,
                             recipient_span=(0, 1), donor_span=(0, 3), recipient_mask=torch.ones(2, dtype=torch.bool),
                             donor_mask=torch.ones(3, dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()
