"""Fixture and split contracts only; no model accuracy claims."""

from collections import Counter, defaultdict
import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from behavioural_templates import families, make_cases, validate, write_suite, canonical_source


class BehaviouralTests(unittest.TestCase):
    def test_balanced_families_and_partitions(self):
        specs, cases = families(), make_cases()
        validate(specs, cases)
        self.assertEqual(Counter(f["split"] for f in specs), {"development": 28, "heldout": 12})
        self.assertEqual(Counter(r["split"] for r in cases), {"development": 336, "heldout": 144})
        for capability in ("negation", "uncertainty", "laterality", "temporal"):
            self.assertEqual(Counter(f["split"] for f in specs if f["capability"] == capability),
                             {"development": 7, "heldout": 3})
        self.assertTrue(all(r["label_status"] == "author_proposed; human review pending" for r in cases))
        self.assertTrue(all(not any(r["human_review"].values()) for r in cases))

    def test_variants_and_relations_do_not_leak_or_change_invariant_labels(self):
        cases = make_cases()
        invariants, contrasts, text_splits = defaultdict(list), defaultdict(list), defaultdict(set)
        for row in cases:
            invariants[row["invariance_group_id"]].append(row)
            contrasts[row["directional_pair_id"]].append(row)
            text_splits[canonical_source(row["input"])].add(row["split"])
        self.assertTrue(all(len(parts) == 1 for parts in text_splits.values()))
        for rows in invariants.values():
            self.assertEqual(len(rows), 2)
            self.assertEqual(" ".join(rows[0]["input"].split()), " ".join(rows[1]["input"].split()))
            self.assertEqual(rows[0]["expected_proposition"], rows[1]["expected_proposition"])
            self.assertEqual(rows[0]["split"], rows[1]["split"])
        for rows in contrasts.values():
            self.assertEqual(len(rows), 4)
            labels = {json.dumps(row["expected_proposition"], sort_keys=True) for row in rows}
            self.assertEqual(len(labels), 2)
            self.assertEqual(len({r["family_id"] for r in rows}), 1)

    def test_scope_uncertainty_laterality_and_temporal_label_controls(self):
        base = {r["id"]: r for r in make_cases()}
        self.assertEqual(base["negation-07-f1-c1-base"]["expected_proposition"],
                         {"concept": "pleural effusion", "assertion": "present",
                          "secondary": {"concept": "cardiac enlargement", "assertion": "absent"}})
        self.assertEqual(base["uncertainty-03-f2-c0-base"]["expected_proposition"]["assertion"], "uncertain")
        self.assertEqual(base["laterality-08-f2-c1-base"]["expected_proposition"]["laterality"], "right")
        self.assertEqual(base["laterality-08-f2-c1-base"]["expected_proposition"]["opposite_laterality"], "left")
        self.assertEqual(base["laterality-08-f2-c1-base"]["expected_proposition"]["opposite_finding_assertion"], "absent")
        self.assertEqual(base["temporal-08-f1-c0-base"]["expected_proposition"],
                         {"concept": "pleural effusion", "assertion": "present",
                          "prior_assertion": "absent", "change": "new"})
        self.assertEqual(base["temporal-03-f3-c0-base"]["expected_proposition"]["assertion"], "absent")

    def test_partition_corruption_is_rejected(self):
        specs, cases = families(), make_cases()
        altered = copy.deepcopy(cases)
        altered[0]["split"] = "heldout"
        with self.assertRaisesRegex(ValueError, "cross partitions"):
            validate(specs, altered)

    def test_cross_family_canonical_source_overlap_is_rejected(self):
        specs, cases = families(), make_cases()
        development = next(row for row in cases if row["split"] == "development")
        heldout = next(row for row in cases if row["split"] == "heldout")
        text = development["input"].upper()
        # Fullwidth ASCII and ideographic/repeated whitespace collapse under NFKC.
        heldout["input"] = "\t" + "".join(chr(ord(c) + 0xFEE0) if "!" <= c <= "~" else "\u3000  "
                                           for c in text) + "\n"
        self.assertNotEqual(development["family_id"], heldout["family_id"])
        self.assertEqual(canonical_source(development["input"]), canonical_source(heldout["input"]))
        with self.assertRaisesRegex(ValueError, "Canonical source text cannot cross partitions"):
            validate(specs, cases)

    def test_reproducible_exports_fingerprints_and_blank_review_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = write_suite(root / "first")
            second = write_suite(root / "second")
            self.assertEqual(first, second)
            self.assertEqual(first["scope"], "synthetic fixtures only; no model outputs")
            for name, metadata in first["files"].items():
                data = (root / "first" / name).read_bytes()
                self.assertEqual(data, (root / "second" / name).read_bytes())
                self.assertEqual(hashlib.sha256(data).hexdigest(), metadata["sha256"])
            with (root / "first" / "human-review.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 480)
            self.assertTrue(all(row["label_approved"] == row["reviewer"] == row["notes"] == "" for row in rows))
            before = (root / "first" / "manifest.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "existing suites are immutable"):
                write_suite(root / "first")
            self.assertEqual((root / "first" / "manifest.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
