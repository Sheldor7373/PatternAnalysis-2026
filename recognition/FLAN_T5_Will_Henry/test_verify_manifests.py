"""Independent manifest checks, including failure exits and unchanged input bytes."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from verify_manifests import normalise, report_hash, verify_manifests


def record(identity, text, target="plain report", image=None):
    return {"id": identity, "input": text, "target": target,
            "key": report_hash(text), "image": image}


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = {"train": [record("train:0", "no effusion", image="a.png")],
                     "dev": [record("train:1", "small opacity", image="b.png")],
                     "evaluation": [record("validation:0", "normal chest", image="c.png")]}

    def write(self, rows=None):
        rows = self.rows if rows is None else rows
        paths, files = {}, {}
        for split, records in rows.items():
            path = self.root / (split + ".jsonl")
            path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
            paths[split] = path
            files[split] = {"manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        audit = {"dataset": "test/dataset", "revision": "abc123", "files": files,
                 "counts": {name: len(records) for name, records in rows.items()}}
        audit_path = self.root / "audit.json"
        audit_path.write_text(json.dumps(audit), encoding="utf-8")
        return paths, audit_path

    def test_complete_consistent_audit_and_original_ids(self):
        paths, audit = self.write()
        result = verify_manifests(paths, audit, expected_dataset="test/dataset", expected_revision="abc123")
        self.assertTrue(result["ok"])
        self.assertTrue(result["provenance"]["verified"])
        self.assertEqual(result["splits"]["dev"]["rows"], 1)

    def test_multiple_references_in_evaluation_are_retained_not_leakage(self):
        self.rows["evaluation"].append(record("validation:1", "NORMAL  chest", "different reference", "c.png"))
        paths, audit = self.write()
        result = verify_manifests(paths, audit)
        self.assertTrue(result["ok"])
        summary = result["splits"]["evaluation"]
        self.assertEqual(summary["rows"], 2)
        self.assertEqual(summary["repeated_report_rows"], 1)
        self.assertEqual(summary["reports_with_multiple_targets"], 1)

    def test_missing_image_ids_do_not_create_false_overlap(self):
        self.rows["train"][0]["image"] = None
        self.rows["dev"][0].pop("image")
        self.rows["evaluation"][0]["image"] = "  "
        paths, audit = self.write()
        result = verify_manifests(paths, audit)
        self.assertTrue(result["ok"])
        self.assertEqual([s["missing_image_rows"] for s in result["splits"].values()], [1, 1, 1])
        self.assertTrue(all(x["image"] == 0 for x in result["overlaps"]))

    def test_unicode_whitespace_case_and_negation(self):
        self.assertEqual(normalise("  ＮＯ\nEffusion  "), "no effusion")
        self.assertEqual(report_hash("NO\tEffusion"), report_hash(" no effusion "))
        self.assertNotEqual(report_hash("no effusion"), report_hash("effusion"))
        self.rows["evaluation"][0] = record("validation:0", " ＮＯ  Effusion ")
        paths, audit = self.write()
        result = verify_manifests(paths, audit)
        self.assertFalse(result["ok"])
        overlap = next(x for x in result["overlaps"] if x["left"] == "evaluation" and x["right"] == "train")
        self.assertEqual(overlap["report"], 1)

    def test_image_overlap_and_duplicate_record_ids_fail(self):
        self.rows["dev"][0]["image"] = "a.png"
        self.rows["evaluation"].append(record("validation:0", "other report"))
        paths, audit = self.write()
        result = verify_manifests(paths, audit)
        failures = {c["check"] for c in result["checks"] if not c["passed"]}
        self.assertIn("dev/train.disjoint", failures)
        self.assertIn("evaluation.unique_record_ids", failures)

    def test_record_id_overlap_and_wrong_split_provenance_fail(self):
        self.rows["dev"][0]["id"] = "validation:0"
        paths, audit = self.write()
        result = verify_manifests(paths, audit)
        self.assertFalse(result["ok"])
        issues = next(c for c in result["checks"] if c["check"] == "dev.records")["issues"]
        self.assertEqual(issues["wrong_origin_id"]["count"], 1)
        self.assertTrue(any(x["id"] == 1 for x in result["overlaps"]))
        self.assertFalse(result["provenance"]["verified"])

    def test_missing_fields_invalid_objects_and_hash_mismatch_fail(self):
        self.rows["train"] += [record("train:2", "new report"), {"id": "train:3"}, []]
        self.rows["train"][1]["key"] = "incorrect"
        paths, audit = self.write()
        result = verify_manifests(paths, audit)
        issues = next(c for c in result["checks"] if c["check"] == "train.records")["issues"]
        self.assertEqual(set(issues), {"report_hash_mismatch", "missing_or_invalid_input",
                                     "missing_or_invalid_target", "non_object_record"})
        self.assertFalse(result["ok"])

    def test_invalid_json_missing_file_and_empty_split_fail(self):
        paths, _ = self.write()
        paths["train"].write_bytes(b"{bad json}\n\xff\n")
        paths["dev"].unlink()
        paths["evaluation"].write_bytes(b"\n")
        result = verify_manifests(paths)
        failures = {c["check"] for c in result["checks"] if not c["passed"]}
        self.assertEqual(failures, {"train.records", "dev.readable", "evaluation.nonempty"})

    def test_schema_fields_and_origins_can_be_changed(self):
        remap = {"id": "record_id", "input": "report", "target": "reference", "key": "hash", "image": "image_id"}
        custom = {split: [{remap[k]: v for k, v in row.items()} for row in rows]
                  for split, rows in self.rows.items()}
        custom["evaluation"][0]["record_id"] = "test:0"
        paths, audit = self.write(custom)
        result = verify_manifests(paths, audit, fields=remap,
                                  origins={"train": "train", "dev": "train", "evaluation": "test"})
        self.assertTrue(result["ok"])

    def test_missing_keys_are_recomputed_and_image_paths_are_case_sensitive(self):
        for records in self.rows.values():
            records[0].pop("key")
        self.rows["train"][0]["image"] = "A.png"
        self.rows["dev"][0]["image"] = "a.png"
        paths, audit = self.write()
        result = verify_manifests(paths, audit)
        self.assertTrue(result["ok"])
        self.assertEqual(result["splits"]["train"]["unique_reports"], 1)

    def test_audit_count_digest_revision_and_dataset_must_match(self):
        paths, audit = self.write()
        metadata = json.loads(audit.read_text())
        metadata["counts"]["train"] = 99
        metadata["files"]["dev"]["manifest_sha256"] = "wrong"
        audit.write_text(json.dumps(metadata))
        result = verify_manifests(paths, audit, expected_dataset="different", expected_revision="different")
        failures = {c["check"] for c in result["checks"] if not c["passed"]}
        self.assertEqual(failures, {"audit.dataset", "audit.revision", "audit.train.count", "audit.dev.manifest_sha256"})
        self.assertFalse(result["provenance"]["verified"])

    def test_missing_or_malformed_audit_cannot_verify_provenance(self):
        paths, audit = self.write()
        for content in ("[]", "not json"):
            audit.write_text(content)
            self.assertFalse(verify_manifests(paths, audit)["ok"])
        audit.unlink()
        self.assertFalse(verify_manifests(paths, audit)["ok"])
        self.assertFalse(verify_manifests(paths, expected_revision="abc123")["ok"])
        self.assertFalse(verify_manifests(paths)["provenance"]["verified"])

    def test_official_evaluation_count_detects_dropped_references(self):
        paths, audit = self.write()
        metadata = json.loads(audit.read_text())
        metadata["official_usable_counts"] = {"train": 2, "validation": 2}
        audit.write_text(json.dumps(metadata))
        result = verify_manifests(paths, audit)
        failures = {c["check"] for c in result["checks"] if not c["passed"]}
        self.assertEqual(failures, {"audit.evaluation.official_usable_count"})

    def test_all_required_splits_must_be_supplied(self):
        paths, _ = self.write()
        del paths["evaluation"]
        self.assertFalse(verify_manifests(paths)["ok"])

    def test_invalid_image_identity_and_blank_target_fail(self):
        self.rows["train"][0]["image"] = ["a.png"]
        self.rows["dev"][0]["target"] = "  "
        paths, audit = self.write()
        result = verify_manifests(paths, audit)
        failures = {c["check"] for c in result["checks"] if not c["passed"]}
        self.assertEqual(failures, {"train.records", "dev.records"})

    def test_deterministic_output_and_no_input_mutation(self):
        paths, audit = self.write()
        inputs = {p: p.read_bytes() for p in [*paths.values(), audit]}
        first = verify_manifests(paths, audit)
        second = verify_manifests(dict(reversed(list(paths.items()))), audit)
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
        self.assertEqual(inputs, {p: p.read_bytes() for p in inputs})

    def test_cli_success_failure_exit_and_no_data_mutation(self):
        paths, audit = self.write()
        command = [sys.executable, "-B", str(Path(__file__).with_name("verify_manifests.py")),
                   "--data", str(self.root), "--expected-revision", "abc123"]
        inputs = {p: p.read_bytes() for p in [*paths.values(), audit]}
        good = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(good.returncode, 0, good.stderr)
        self.assertTrue(json.loads(good.stdout)["ok"])
        failed = subprocess.run(command[:-1] + ["wrong"], capture_output=True, text=True)
        self.assertEqual(failed.returncode, 1)
        self.assertFalse(json.loads(failed.stdout)["ok"])
        self.assertEqual(inputs, {p: p.read_bytes() for p in inputs})


if __name__ == "__main__":
    unittest.main()
