"""Receipt consistency attacks using explicitly synthetic data only."""
import copy
import unittest

from negation_preflight import PINNED_REVISION, preflight
from negation_receipts import (LOAD_OPTIONS, PINNED_ASSETS, expected_source_input_receipt,
                               validate_tokenizer_provenance)
from negation_summary import summarise
from test_negation_preflight import FixtureTokenizer, fixture_manifest
from test_negation_summary import synthetic_scores


class ReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = fixture_manifest()
        cls.receipt = preflight(cls.manifest, FixtureTokenizer(cls.manifest))

    def summary(self, data):
        return summarise(self.manifest, self.receipt, data)

    def test_real_scores_cannot_consume_synthetic_pass_or_old_unbound_schema(self):
        data = synthetic_scores(self.manifest, self.receipt)
        data["data_kind"] = "saved_model_scores"
        with self.assertRaisesRegex(ValueError, "Real scores require"): self.summary(data)
        data["data_kind"] = "synthetic_test_scores"; data.pop("score_receipt_version")
        with self.assertRaisesRegex(ValueError, "versioned actual-input"): self.summary(data)

    def test_pinned_provenance_declarations_require_all_assets_class_options_and_specials(self):
        # This tests a declared provenance schema, not genuine tokenizer/model evidence.
        receipt = {"tokenizer": {"kind": "offline cached tokenizer", "revision": PINNED_REVISION,
                    "files": copy.deepcopy(PINNED_ASSETS),
                    "class": "transformers.models.t5.tokenization_t5_fast.T5TokenizerFast", "is_fast": True,
                    "load_options": copy.deepcopy(LOAD_OPTIONS), "eos_token_id": 1, "pad_token_id": 0},
                   "eos_token_id": 1, "pad_token_id": 0, "special_token_ids": [0,1,2]+list(range(32000,32100))}
        validate_tokenizer_provenance(receipt, "saved_model_scores")
        with self.assertRaises(ValueError): validate_tokenizer_provenance(receipt, "synthetic_test_scores")
        for corruption in ("hash", "revision", "class", "offline", "bool", "specials", "missing_asset"):
            item = copy.deepcopy(receipt)
            if corruption == "hash": item["tokenizer"]["files"]["tokenizer.json"]["sha256"] = "0"*64
            if corruption == "revision": item["tokenizer"]["revision"] = "0"*40
            if corruption == "class": item["tokenizer"]["class"] = "FixtureTokenizer"
            if corruption == "offline": item["tokenizer"]["load_options"]["local_files_only"] = False
            if corruption == "bool": item["tokenizer"]["load_options"]["local_files_only"] = 1
            if corruption == "specials": item["special_token_ids"].pop()
            if corruption == "missing_asset": item["tokenizer"]["files"].pop("spiece.model")
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                validate_tokenizer_provenance(item, "saved_model_scores")

    def test_absent_present_recipient_and_donor_payload_swaps_rejected(self):
        data = synthetic_scores(self.manifest, self.receipt)
        first = data["baselines"][0]
        opposite = next(r for r in data["baselines"] if r["model"] == first["model"]
                        and r["source_id"] == first["source_id"].replace("/absent", "/present"))
        first["recipient_input"] = copy.deepcopy(opposite["recipient_input"])
        with self.assertRaisesRegex(ValueError, "actual IDs"): self.summary(data)
        data = synthetic_scores(self.manifest, self.receipt)
        first = next(r for r in data["patches"] if "<-" in r["case_id"] and r["case_id"].endswith("/absent"))
        opposite = next(r for r in data["patches"] if r["model"] == first["model"]
                        and r["case_id"] == first["case_id"].rsplit("/",1)[0] + "/present")
        first["donor_input"] = copy.deepcopy(opposite["donor_input"])
        with self.assertRaisesRegex(ValueError, "actual IDs"): self.summary(data)

    def test_actual_arrays_masks_eligibility_span_offsets_and_hashes_are_bound(self):
        for field in ("input_ids", "attention_mask", "eligible_mask", "token_span", "raw_offsets_sha256",
                      "forward_input_sha256", "mapping_policy_version", "bound_input_sha256", "mask_type", "id_type"):
            data = synthetic_scores(self.manifest, self.receipt)
            payload = data["baselines"][0]["recipient_input"]
            if field == "input_ids": payload[field][0] += 1
            elif field in ("attention_mask", "eligible_mask"): payload[field][0] = not payload[field][0]
            elif field == "token_span": payload[field][0] += 1
            elif field == "mask_type": payload["attention_mask"][0] = 1  # equals True in Python, differs in canonical JSON
            elif field == "id_type": payload["input_ids"][0] = float(payload["input_ids"][0])
            else: payload[field] = "0"*64
            with self.subTest(field=field), self.assertRaises(ValueError): self.summary(data)

    def test_checkpoint_basis_and_baseline_self_patch_roles_bound_per_row(self):
        for field in ("checkpoint_id", "basis_id", "basis_applied", "donor_input", "recipient_input"):
            for table in ("baselines", "patches"):
                data = synthetic_scores(self.manifest, self.receipt)
                row = data[table][0]
                if field in ("checkpoint_id", "basis_id"): row[field] = "f"*64
                elif field == "basis_applied": row[field] = not row[field]
                else: row.pop(field)
                with self.subTest(field=field, table=table), self.assertRaises(ValueError): self.summary(data)
        data = synthetic_scores(self.manifest, self.receipt)
        data["checkpoint_ids"]["frozen"] = "d"*64
        with self.assertRaises(ValueError): self.summary(data)

    def test_expected_receipt_copies_arrays_without_stamping_execution_evidence(self):
        record = self.receipt["records"][0]
        result = expected_source_input_receipt(record["id"], record)
        result["input_ids"][0] = 999
        self.assertNotEqual(result["input_ids"], record["source"]["input_ids"])


if __name__ == "__main__":
    unittest.main()
