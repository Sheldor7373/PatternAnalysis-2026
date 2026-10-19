"""Tiny random CPU forwards only; mocked sums are labelled synthetic flow tests."""
import copy
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from negation_model_copies import prepare_copies
from negation_preflight import digest, preflight
from negation_receipts import validate_row_binding
from negation_runner import (ForwardBudget, LIMITS, _candidate_score, actual_source_receipt, run_screen)
from t5_intervention_adapter import Candidate, SourceTokens
from test_negation_model_copies import tiny_templates
from test_negation_preflight import FixtureTokenizer, fixture_manifest


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads(); torch.set_num_threads(1)
        cls.base, cls.wrapped, cls.lock = tiny_templates()
        cls.manifest = fixture_manifest()
        cls.preflight = preflight(cls.manifest, FixtureTokenizer(cls.manifest))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def prepared(self):
        return prepare_copies(lambda: copy.deepcopy(self.base), lambda: copy.deepcopy(self.wrapped), self.lock, fixture=True)

    def controlled_scores(self, *, flat=False, fail_substantive=False):
        original = _candidate_score
        def score(*args, source_id, name, **kwargs):
            values, logits = original(*args, source_id=source_id, name=name, **kwargs)
            # ONLY unit control-flow testing; not a model quality measurement.
            present = source_id.endswith("/present")
            margin = 0 if flat or (fail_substantive and "/negation-01/" not in source_id) else 2 if present else -2
            values["summed_logprob"] = -1 if (name == "positive") == (margin > 0) else -1-abs(margin)
            return values, logits
        return patch("negation_runner._candidate_score", side_effect=score)

    def test_actual_source_receipt_reads_tensors_and_never_copies_expected_receipt(self):
        record = self.preflight["records"][0]
        source = record["source"]
        actual = SourceTokens(torch.tensor([source["input_ids"]]), torch.tensor([source["attention_mask"]]),
                              torch.tensor([source["eligible_mask"]]))
        with patch("negation_receipts.expected_source_input_receipt", side_effect=AssertionError("expected helper called")):
            captured = actual_source_receipt(record["id"], actual, source["token_span"], record)
        original_id = actual.input_ids[0,0].item(); actual.input_ids[0,0] = original_id + 1
        changed = actual_source_receipt(record["id"], actual, source["token_span"], record)
        self.assertNotEqual(captured["bound_input_sha256"], changed["bound_input_sha256"])
        row = {"recipient_input": changed, "donor_input": None, "checkpoint_id": "a"*64,
               "basis_id": None, "basis_applied": False}
        with self.assertRaises(ValueError):
            validate_row_binding(row, record["id"], None, {record["id"]: record}, "a"*64, None)

    def test_plain_and_wrapped_scores_match_manual_sums_eos_masks_and_decoder_start(self):
        copies = self.prepared()
        source = SourceTokens(torch.tensor([[7,8,1]]), torch.ones((1,3), dtype=torch.bool),
                              torch.tensor([[True,True,False]]))
        candidate = Candidate(torch.tensor([[9,10,1,17]]), torch.tensor([[True,True,True,False]]), True)
        for model in (copies.frozen, copies.wrapped_reference):
            seen = []
            hook = model.get_decoder().register_forward_pre_hook(
                lambda module,args,kwargs: seen.append((kwargs["input_ids"].clone(), kwargs["attention_mask"].clone())), with_kwargs=True)
            try:
                with torch.inference_mode():
                    encoded = model.get_encoder()(input_ids=source.input_ids, attention_mask=source.attention_mask, return_dict=True)
                    score, logits = _candidate_score(model, encoded, source, candidate, source_id="synthetic/fixture", name="positive")
            finally: hook.remove()
            manual = logits.double().log_softmax(-1).gather(-1, candidate.input_ids.unsqueeze(-1)).squeeze(-1)
            self.assertEqual(score["summed_logprob"], float(manual.masked_fill(~candidate.target_mask,0).sum()))
            self.assertEqual(score["token_count"], 3)
            self.assertTrue(score["eos_included"])
            self.assertEqual(score["token_ids_sha256"], digest([9,10,1,17]))
            self.assertEqual(score["target_mask_sha256"], digest([True,True,True,False]))
            self.assertEqual(int(seen[0][0][0,0]), 0); self.assertTrue(bool(seen[0][1][0,0]))

    def test_pilot_gate_failure_stops_before_substantive_with_exact_pilot_budget(self):
        copies = self.prepared()
        with self.controlled_scores(flat=True): result = run_screen(self.manifest, self.preflight, copies)
        self.assertEqual(result["status"], "STOP")
        self.assertEqual(result["pilot_summary"]["gate"]["status"], "FAIL")
        self.assertIsNone(result["substantive_scores"])
        self.assertEqual(result["forward_counts"]["pilot"], LIMITS["pilot"])
        self.assertTrue(all(n == 0 for role in result["forward_counts"]["substantive"].values() for n in role.values()))
        self.assertEqual(result["immutable_state_check"], "PASS")

    def test_full_fixed_flow_and_exact_budget_with_explicit_mocked_synthetic_sums(self):
        copies = self.prepared()
        with self.controlled_scores(): result = run_screen(self.manifest, self.preflight, copies)
        self.assertEqual(result["status"], "REVIEW_REQUIRED", result["failure"])
        self.assertEqual(result["prefit_gate"]["status"], "PASS")
        self.assertEqual(result["forward_counts"], LIMITS)
        self.assertEqual(len(result["pilot_scores"]["baselines"]), 8)
        self.assertEqual(len(result["pilot_scores"]["patches"]), 24)
        self.assertEqual(len(result["substantive_scores"]["baselines"]), 32)
        self.assertFalse(result["fitting_authorised"])
        self.assertEqual(result["pilot_scores"]["data_kind"], "synthetic_test_scores")
        for data in (result["pilot_scores"], result["substantive_scores"]):
            self.assertTrue(all(row["recipient_input"]["source_id"].startswith("fit/") for row in data["baselines"]+data["patches"]))
        result["declared_limits"]["pilot"]["frozen"]["encoder"] = 999
        self.assertEqual(LIMITS["pilot"]["frozen"]["encoder"], 36)

    def test_substantive_failure_preserves_full_panel_and_has_no_fitting_continuation(self):
        with self.controlled_scores(fail_substantive=True):
            result = run_screen(self.manifest, self.preflight, self.prepared())
        self.assertEqual(result["status"], "STOP")
        self.assertEqual(result["substantive_summary"]["gate"]["status"], "FAIL")
        self.assertEqual(len(result["substantive_scores"]["baselines"]), 32)
        self.assertFalse(result["fitting_authorised"])

    def test_unmocked_tiny_pilot_only_is_bounded_without_any_quality_claim(self):
        result = run_screen(self.manifest, self.preflight, self.prepared(), include_substantive=False)
        self.assertIsNotNone(result["pilot_summary"], result["failure"])
        self.assertEqual(result["forward_counts"]["pilot"], LIMITS["pilot"])
        self.assertIsNone(result["substantive_scores"])

    def test_merge_failure_and_decoder_exception_stop_and_cleanup_only_own_hooks(self):
        copies = self.prepared()
        decoder = copies.frozen.get_decoder()
        observer = decoder.register_forward_pre_hook(lambda module,args: None)
        before = dict(decoder._forward_pre_hooks)
        try:
            with patch.object(decoder, "forward", side_effect=RuntimeError("synthetic decoder failure")):
                result = run_screen(self.manifest, self.preflight, copies)
            self.assertEqual(result["status"], "STOP")
            self.assertIn("synthetic decoder failure", result["failure"])
            self.assertEqual(decoder._forward_pre_hooks, before)
            self.assertFalse(copies.frozen.encoder.block[3]._forward_hooks)
        finally: observer.remove()
        original = _candidate_score
        def mismatch(model, *args, **kwargs):
            score, logits = original(model, *args, **kwargs)
            return score, logits + (1 if model is copies.wrapped_reference else 0)
        with patch("negation_runner._candidate_score", side_effect=mismatch):
            result = run_screen(self.manifest, self.preflight, copies)
        self.assertEqual(result["status"], "STOP")
        self.assertIn("merge equivalence", result["failure"])
        self.assertIsNone(result["substantive_scores"])
        self.assertEqual(result["forward_counts"]["pilot"]["lora_merged_copy"], {"encoder":1,"decoder":2})

    def test_budget_guard_and_bad_preflight_or_mutable_lock_reject_before_scoring(self):
        copies = self.prepared(); budget = ForwardBudget(copies)
        budget.counts["pilot"]["frozen"]["encoder"] = LIMITS["pilot"]["frozen"]["encoder"]
        with budget.tracked(), self.assertRaisesRegex(RuntimeError, "budget"):
            copies.frozen.get_encoder()(input_ids=torch.tensor([[3,1]]))
        corrupted = copy.deepcopy(self.preflight); corrupted["status"] = "FAIL"
        with self.assertRaises(ValueError): run_screen(self.manifest, corrupted, copies)
        with torch.no_grad(): next(copies.merged.parameters()).add_(0.1)
        with self.assertRaisesRegex(ValueError, "Immutable"): run_screen(self.manifest, self.preflight, copies)

    def test_state_mutation_during_forwards_is_detected_and_never_continues(self):
        copies = self.prepared()
        changed = []
        def mutate_once(module, args):
            if not changed:
                with torch.no_grad(): next(copies.frozen.parameters()).add_(0.001)
                changed.append(True)
        hook = copies.frozen.get_encoder().register_forward_pre_hook(mutate_once)
        try:
            result = run_screen(self.manifest, self.preflight, copies)
        finally: hook.remove()
        self.assertEqual(result["status"], "STOP")
        self.assertTrue(result["immutable_state_check"].startswith("FAIL"))
        self.assertIsNone(result["substantive_scores"])
        self.assertEqual(result["forward_counts"]["pilot"]["lora_merged_copy"], {"encoder":0,"decoder":0})

    def test_cpu_autocast_is_rejected_before_any_forward(self):
        copies = self.prepared(); seen = []
        hook = copies.frozen.get_encoder().register_forward_pre_hook(lambda module,args: seen.append(True))
        try:
            with torch.autocast("cpu", dtype=torch.bfloat16), self.assertRaisesRegex(ValueError, "autocast"):
                run_screen(self.manifest, self.preflight, copies)
        finally: hook.remove()
        self.assertEqual(seen, [])


if __name__ == "__main__":
    unittest.main()
