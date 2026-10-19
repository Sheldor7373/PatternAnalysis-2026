"""Tests for split leakage and stable sample selection. No downloads needed."""

import unittest
from dataset import clean_rows, normalise, partition_rows, select_rows


def rows(texts, split):
    return clean_rows([{"radiology_report": text, "layman_report": "plain " + text,
                        "images_path": f"{split}/{i}"} for i, text in enumerate(texts)], split)[0]


class SplitTests(unittest.TestCase):
    def test_normalisation_keeps_negation(self):
        self.assertEqual(normalise("  NO\nEffusion "), "no effusion")
        self.assertNotEqual(normalise("no effusion"), normalise("effusion"))

    def test_duplicates_cannot_cross_splits(self):
        training = rows(["normal", "NO EFFUSION", "no  effusion", "fracture", "mass"], "train")
        validation = rows([" Normal ", "oedema", "pneumonia", "pneumonia"], "validation")
        splits, audit = partition_rows(training, validation)
        keys = {name: {r["key"] for r in records} for name, records in splits.items()}
        for left, right in [("train", "dev"), ("train", "evaluation"), ("dev", "evaluation")]:
            self.assertFalse(keys[left] & keys[right])
        self.assertEqual(audit["train_overlap_removed"], 1)
        self.assertEqual(audit["validation_duplicate_reports"], 1)
        self.assertEqual(len(splits["evaluation"]), len(validation))
        self.assertTrue(all(r["id"].startswith("train:") for r in splits["dev"]))
        self.assertEqual(len(splits["train"]) + len(splits["dev"]), len(training) - 1)
        self.assertEqual(partition_rows(training, validation), (splits, audit))

    def test_image_overlap_is_removed(self):
        training = rows(["a", "b", "c"], "train")
        validation = rows(["d", "e", "f"], "validation")
        training[0]["image"] = validation[0]["image"]
        splits, audit = partition_rows(training, validation)
        self.assertEqual(audit["train_overlap_removed"], 1)
        self.assertNotIn(training[0], splits["train"])

    def test_missing_targets_are_not_stringified(self):
        cleaned, missing = clean_rows([{"radiology_report": "abc", "layman_report": None}], "train")
        self.assertEqual((cleaned, missing), ([], 1))

    def test_sampling_is_reproducible_and_not_a_prefix(self):
        sample = select_rows(list(range(100)), 10)
        self.assertEqual(sample, select_rows(list(range(100)), 10))
        self.assertNotEqual(sample, list(range(10)))

    def test_image_connections_keep_different_reports_together(self):
        training = rows(["a", "b", "c", "d", "e"], "train")
        training[1]["image"] = training[0]["image"]
        training[2]["input"] = "a"
        training[2]["key"] = training[0]["key"]
        splits, _ = partition_rows(training, rows(["z"], "validation"), dev_fraction=0.5)
        for split in ("train", "dev"):
            present = {r["id"] for r in splits[split]}
            self.assertTrue(not ({"train:0", "train:1", "train:2"} & present)
                            or {"train:0", "train:1", "train:2"}.issubset(present))


if __name__ == "__main__":
    unittest.main()
