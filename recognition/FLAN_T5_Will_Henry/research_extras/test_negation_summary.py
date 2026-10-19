"""Saved-score arithmetic on explicitly synthetic fixtures; no model/tokenizer assets."""

import copy
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

from negation_preflight import MODELS, digest, preflight
from negation_summary import AGGREGATION, main, prefit_gate, select_snapshots, summarise
from test_negation_preflight import FixtureTokenizer, fixture_manifest
from negation_receipts import SCORE_RECEIPT_VERSION, expected_source_input_receipt


def synthetic_scores(manifest, receipt, stage="pilot", condition=None, step=32):
    sources = {r["id"]: r for r in manifest["sources"]}
    records = {r["id"]: r for r in receipt["records"]}
    if stage == "pilot":
        ids = set(manifest["screened_pilot"]["recipient_ids"])
    elif stage == "substantive":
        ids = set(manifest["substantive_signal_before_fitting"]["panel_source_ids"])
    else:
        ids = {r["id"] for r in sources.values() if r["split"] == stage and "recipient" in r["roles"]}
    condition = condition or {"pilot": "coordinate_e0", "substantive": "none"}.get(stage, "learned_true")
    data = {"data_kind": "synthetic_test_scores", "score_receipt_version": SCORE_RECEIPT_VERSION,
            "protocol_sha256": digest(manifest), "preflight_sha256": digest(receipt),
            "stage": stage, "basis_condition": condition,
            "checkpoint_ids": {model: str(i + 1) * 64 for i, model in enumerate(MODELS)},
            "basis_ids": {model: str(i + 3) * 64 for i, model in enumerate(MODELS)},
            "baselines": [], "patches": [], "snapshot_step": step}
    def row(model, source_id, margin, case_id=None):
        candidates = records[source_id]["candidates"]
        values = (0, -margin) if margin >= 0 else (margin, 0)
        result = {"model": model, "source_id" if case_id is None else "case_id": source_id if case_id is None else case_id}
        donor_id = (source_id if case_id and case_id.startswith("self:") else
                    next(c["donor_id"] for c in manifest["intervention_cases"] if c["id"] == case_id) if case_id else None)
        result.update(recipient_input=expected_source_input_receipt(source_id, records[source_id]),
                      donor_input=expected_source_input_receipt(donor_id, records[donor_id]) if donor_id is not None else None,
                      checkpoint_id=data["checkpoint_ids"][model], basis_id=data["basis_ids"][model] if stage != "substantive" else None,
                      basis_applied=case_id is not None)
        for name, value in zip(("positive", "negative"), values):
            result[name] = {"summed_logprob": value, "aggregation": AGGREGATION, "eos_included": True,
                            **{field: candidates[name][field] for field in ("token_count", "token_ids_sha256", "target_mask_sha256")}}
        return result
    for model in MODELS:
        for source_id in sorted(ids):
            margin = 2 if sources[source_id]["assertion"] == "present" else -2
            data["baselines"].append(row(model, source_id, margin))
            if stage != "substantive":
                data["patches"].append(row(model, source_id, margin, "self:" + source_id))
        if stage != "substantive":
            for case in manifest["intervention_cases"]:
                if case["recipient_id"] in ids:
                    margin = 3 if sources[case["donor_id"]]["assertion"] == "present" else -3
                    data["patches"].append(row(model, case["recipient_id"], margin, case["id"]))
    if stage == "heldout":
        data["selection_lock"] = {field: copy.deepcopy(data[field]) for field in
                                  ("protocol_sha256", "preflight_sha256", "checkpoint_ids", "basis_ids", "basis_condition")}
        data["selection_lock"]["selection_summary_sha256"] = "a" * 64
    return data


def change_margin(row, margin):
    row["positive"]["summed_logprob"], row["negative"]["summed_logprob"] = ((0, -margin) if margin >= 0 else (margin, 0))


def add_mechanical_receipts(data):
    ids = sorted({r["source_id"] for r in data["baselines"]})
    data["mechanical_evidence"] = {}
    for model in MODELS:
        item = {"parameters_before_sha256": "b" * 64, "parameters_after_sha256": "b" * 64,
                "self_logits": [{"source_id": identity, "max_normalised_error": 0, "atol": 1e-6, "rtol": 1e-6} for identity in ids]}
        if model == "lora_merged_copy":
            item["merge_equivalence"] = [{"source_id": identity, "max_normalised_logit_error": 0, "atol": 1e-5, "rtol": 1e-5,
                "reference_margin": next(r["positive"]["summed_logprob"] - r["negative"]["summed_logprob"]
                                         for r in data["baselines"] if r["model"] == model and r["source_id"] == identity)} for identity in ids]
        data["mechanical_evidence"][model] = item


class SummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = fixture_manifest()
        cls.receipt = preflight(cls.manifest, FixtureTokenizer(cls.manifest))

    def run_summary(self, data, receipt=None):
        return summarise(self.manifest, receipt or self.receipt, data)

    def test_exact_I_P_directional_drift_flip_and_paired_checkpoint_values(self):
        data = synthetic_scores(self.manifest, self.receipt)
        result = self.run_summary(data)
        self.assertEqual(result, self.run_summary(copy.deepcopy(data)))
        for model in MODELS:
            values = result["models"][model]
            self.assertEqual(values["I"]["equal_family_mean"], 4)
            self.assertEqual(values["P"]["equal_family_mean"], 6)
            for direction, effect in {"AP": 5, "PA": -5, "AA": -1, "PP": 1}.items():
                aggregate = values["directional_aggregates"][direction]
                self.assertEqual(aggregate["signed"]["equal_family_mean"], effect)
                self.assertEqual(aggregate["absolute"]["equal_family_mean"], abs(effect))
            self.assertTrue(all(r["both_cross_directions_flip"] for r in values["bidirectional_flip_pairs"]))
            self.assertTrue(all(r["effect"] == 0 for r in values["self_effects"]))
        self.assertEqual(result["paired_lora_minus_frozen"]["P"]["aggregate"]["equal_family_mean"], 0)
        self.assertEqual(result["gate"]["status"], "UNAVAILABLE")
        self.assertFalse(result["fitting_authorised"])
        self.assertEqual(len(result["raw_patches"]), 24)

    def test_row_order_does_not_change_arithmetic_and_raw_rows_are_sorted(self):
        data = synthetic_scores(self.manifest, self.receipt, "selection")
        original = self.run_summary(data)
        data["baselines"].reverse(); data["patches"].reverse()
        reordered = self.run_summary(data)
        self.assertNotEqual(original.pop("scores_sha256"), reordered.pop("scores_sha256"))  # exact input provenance retained
        self.assertEqual(original, reordered)

    def test_positive_P_without_flip_and_zero_ties_are_reported(self):
        data = synthetic_scores(self.manifest, self.receipt)
        for row in data["patches"]:
            if not row["case_id"].startswith("self:"):
                change_margin(row, 0 if row["case_id"].endswith("/absent") else 1)
        result = self.run_summary(data)["models"]["frozen"]
        self.assertEqual(result["P"]["equal_family_mean"], 1)
        self.assertTrue(all(not r["both_cross_directions_flip"] for r in result["bidirectional_flip_pairs"]))
        pa = next(r for r in result["directional_families"] if r["direction"] == "PA")
        self.assertEqual(pa["post_tie_count"], pa["count"])
        self.assertEqual(pa["flip_count"], 0)
        # Donor-consistent preference does not count as a flip with an incorrect baseline.
        for row in data["baselines"]:
            if row["source_id"].endswith("/absent"):
                change_margin(row, 2)
        ap = [r for r in self.run_summary(data)["models"]["frozen"]["directional_cases"] if r["direction"] == "AP"]
        self.assertTrue(all(r["donor_consistent"] and not r["flip"] and not r["baseline_correct"] for r in ap))

    def test_substantive_threshold_counts_findings_families_and_both_checkpoints(self):
        data = synthetic_scores(self.manifest, self.receipt, "substantive")
        self.assertEqual(self.run_summary(data)["gate"]["status"], "PASS")
        for row in data["baselines"]:
            change_margin(row, math.log(2) / 2 * (1 if row["source_id"].endswith("/present") else -1))
        self.assertEqual(self.run_summary(data)["gate"]["status"], "PASS")  # inclusive ln2 threshold
        for families, finding, expected in [({"negation-01"}, None, "PASS"),
                                            ({"negation-01", "negation-02"}, "effusion", "FAIL")]:
            fixture = synthetic_scores(self.manifest, self.receipt, "substantive")
            for row in fixture["baselines"]:
                parts = row["source_id"].split("/")
                if row["model"] == "lora_merged_copy" and parts[1] in families and (finding is None or parts[2] == finding):
                    change_margin(row, 0.2 if row["source_id"].endswith("/present") else 0)
            self.assertEqual(self.run_summary(fixture)["gate"]["status"], expected)
        fixture = synthetic_scores(self.manifest, self.receipt, "substantive")
        for row in fixture["baselines"]:
            if "/negation-01/" in row["source_id"]:
                change_margin(row, -10 if row["source_id"].endswith("/present") else 10)
        gate = self.run_summary(fixture)["models"]["frozen"]["substantive_gate"]
        self.assertEqual(gate["successful_pairs"], 6)
        self.assertEqual(gate["successes_by_finding"], {"effusion": 3, "pneumothorax": 3})
        self.assertEqual(gate["status"], "FAIL")  # every family must have positive mean I

    def test_pilot_mechanics_missing_pass_fail_and_combined_prefit_gate(self):
        pilot = synthetic_scores(self.manifest, self.receipt)
        self.assertEqual(self.run_summary(pilot)["gate"]["status"], "UNAVAILABLE")
        add_mechanical_receipts(pilot)
        good = self.run_summary(pilot)
        self.assertEqual(good["gate"]["status"], "PASS")
        substantive = self.run_summary(synthetic_scores(self.manifest, self.receipt, "substantive"))
        self.assertEqual(prefit_gate(good, substantive)["status"], "PASS")
        self.assertFalse(prefit_gate(good, substantive)["fitting_authorised"])
        for failure in ("parameter", "self_logit", "merge_logit", "merge_margin", "self_margin", "tiny_input"):
            data = copy.deepcopy(pilot)
            if failure == "parameter": data["mechanical_evidence"]["frozen"]["parameters_after_sha256"] = "c" * 64
            if failure == "self_logit": data["mechanical_evidence"]["frozen"]["self_logits"][0]["max_normalised_error"] = 1.01
            if failure == "merge_logit": data["mechanical_evidence"]["lora_merged_copy"]["merge_equivalence"][0]["max_normalised_logit_error"] = 1.01
            if failure == "merge_margin": data["mechanical_evidence"]["lora_merged_copy"]["merge_equivalence"][0]["reference_margin"] = 9
            if failure == "self_margin":
                row = next(r for r in data["patches"] if r["case_id"].startswith("self:"))
                change_margin(row, 9)
            if failure == "tiny_input":
                for row in data["baselines"]:
                    change_margin(row, 1e-6 if row["source_id"].endswith("/present") else 0)
            with self.subTest(failure=failure):
                self.assertEqual(self.run_summary(data)["gate"]["status"], "FAIL")
        substantive["checkpoint_ids"]["frozen"] = "d" * 64
        with self.assertRaises(ValueError): prefit_gate(good, substantive)

    def test_failed_preflight_or_changed_masks_counts_eos_offsets_hashes_rejected(self):
        data = synthetic_scores(self.manifest, self.receipt)
        for corruption in ("failed", "missing", "mask", "count", "eos", "offset", "hash", "protocol"):
            receipt = copy.deepcopy(self.receipt)
            source = receipt["records"][0]["source"]
            if corruption == "failed": receipt["status"] = "FAIL"
            if corruption == "missing": receipt["records"].pop()
            if corruption == "mask": source["attention_mask"][0] = False
            if corruption == "count": receipt["records"][0]["candidates"]["positive"]["token_count"] += 1
            if corruption == "eos": source["eos_included"] = False
            if corruption == "offset": source["combined_char_span"][0] -= 1
            if corruption == "hash": source["forward_input_sha256"] = "0" * 64
            if corruption == "protocol": receipt["protocol_sha256"] = "0" * 64
            fixture = copy.deepcopy(data); fixture["preflight_sha256"] = digest(receipt)
            with self.subTest(corruption=corruption), self.assertRaises(ValueError): self.run_summary(fixture, receipt)

    def test_score_semantics_and_provenance_corruption_rejected(self):
        original = synthetic_scores(self.manifest, self.receipt)
        for field, value in (("token_count", 999), ("token_count", True), ("eos_included", False),
                             ("aggregation", "mean"), ("token_ids_sha256", "0" * 64),
                             ("target_mask_sha256", "0" * 64), ("summed_logprob", float("nan")),
                             ("summed_logprob", 1), ("summed_logprob", True)):
            data = copy.deepcopy(original); data["baselines"][0]["positive"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError): self.run_summary(data)
        data = copy.deepcopy(original); data["baselines"][0]["margin"] = 99
        with self.assertRaises(ValueError): self.run_summary(data)
        data = copy.deepcopy(original); data["protocol_sha256"] = "f" * 64
        with self.assertRaises(ValueError): self.run_summary(data)

    def test_missing_duplicate_extra_cross_split_cases_and_wrong_condition_rejected(self):
        for corruption in ("missing", "duplicate", "cross_split", "missing_patch", "wrong_condition"):
            data = synthetic_scores(self.manifest, self.receipt, "selection")
            if corruption == "missing": data["baselines"].pop()
            if corruption == "duplicate": data["patches"].append(copy.deepcopy(data["patches"][0]))
            if corruption == "cross_split": data["baselines"][0]["source_id"] = "heldout/negation-08/atelectasis/absent"
            if corruption == "missing_patch": data["patches"].pop()
            if corruption == "wrong_condition": data["basis_condition"] = "random_9999"
            with self.subTest(corruption=corruption), self.assertRaises(ValueError): self.run_summary(data)

    def test_heldout_requires_consistent_lock_and_preserves_true_labels_for_controls(self):
        for condition in ("learned_true", "permuted_3720", "random_3711", "random_3712", "random_3713"):
            data = synthetic_scores(self.manifest, self.receipt, "heldout", condition)
            result = self.run_summary(data)
            self.assertEqual(result["models"]["frozen"]["P"]["equal_family_mean"], 6)
            self.assertEqual(len(result["models"]["frozen"]["primary_contrasts"]), 12)
            self.assertFalse(result["optional_readouts_enabled"])
            data["selection_lock"]["basis_ids"]["frozen"] = "0" * 64
            with self.assertRaises(ValueError): self.run_summary(data)
        data.pop("selection_lock")
        with self.assertRaises(KeyError): self.run_summary(data)

    def test_selection_rule_is_per_checkpoint_ties_choose32_and_forbids_heldout(self):
        a = self.run_summary(synthetic_scores(self.manifest, self.receipt, "selection", step=32))
        data64 = synthetic_scores(self.manifest, self.receipt, "selection", step=64)
        data64["basis_ids"] = {model: "e" * 64 for model in MODELS}
        for row in data64["baselines"] + data64["patches"]:
            row["basis_id"] = data64["basis_ids"][row["model"]]
        for row in data64["patches"]:
            if not row["case_id"].startswith("self:") and row["case_id"].endswith("/present"):
                change_margin(row, 4 if row["model"] == "lora_merged_copy" else 3 + 5e-7)
        b = self.run_summary(data64)
        chosen = select_snapshots(a, b)
        self.assertEqual(chosen["choices"]["frozen"]["step"], 32)
        self.assertEqual(chosen["choices"]["lora_merged_copy"]["step"], 64)
        self.assertEqual(chosen["choices"]["lora_merged_copy"]["basis_id"], "e" * 64)
        with self.assertRaises(ValueError): select_snapshots(a, self.run_summary(synthetic_scores(self.manifest, self.receipt, "heldout")))

    def test_cli_new_output_and_manifest_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); inputs = root / "inputs"; inputs.mkdir()
            manifest, receipt, scores = [inputs / name for name in ("manifest.json", "preflight.json", "scores.json")]
            data = synthetic_scores(self.manifest, self.receipt, "substantive")
            for path, value in ((manifest, self.manifest), (receipt, self.receipt), (scores, data)):
                path.write_text(json.dumps(value), encoding="utf-8")
            before = {path: path.read_bytes() for path in (manifest, receipt, scores)}
            output = root / "summary.json"
            argv = ["--manifest", str(manifest), "--preflight", str(receipt), "--scores", str(scores), "--output", str(output)]
            with redirect_stdout(io.StringIO()): self.assertEqual(main(argv), 0)
            self.assertEqual(json.loads(output.read_text())["data_kind"], "synthetic_test_scores")
            with self.assertRaises(ValueError): main(argv)
            self.assertTrue(all(path.read_bytes() == raw for path, raw in before.items()))


if __name__ == "__main__":
    unittest.main()
