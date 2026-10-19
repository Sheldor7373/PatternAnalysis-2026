"""Tiny deterministic CPU array tests, never model activations or inference."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from representation_analysis import linear_cka, load_saved, compare_saved, MAX_ROWS


def metadata(n=6):
    return {"model_label": "synthetic-A", "model_revision": "synthetic-revision",
            "checkpoint_sha256": "a" * 64, "data_split": "synthetic",
            "source_manifest_sha256": "b" * 64, "tokenizer_revision": "synthetic-tokenizer",
            "prompt_sha256": "c" * 64, "source_length": 32,
            "pooling": "masked mean of content tokens; excludes prompt, padding and special tokens",
            "site": "synthetic.encoder.layer0", "visible_tokens_sha256": ["d" * 64] * n}


class RepresentationTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[1., 2., 0.], [-1., 3., 2.], [4., -2., 3.],
                           [2., 0., -4.], [-3., 1., 2.], [0., -4., 1.]])
        self.y = np.array([[0., 1.], [1., 4.], [-3., 2.], [2., 1.], [1., -2.], [-2., 0.]])
        self.ids = [f"synthetic-{i}" for i in range(6)]

    def cka(self, x, y):
        return linear_cka(x, y, self.ids, self.ids)["cka"]

    def test_centred_gram_matches_independent_feature_formula_and_symmetry(self):
        x, y = self.x - self.x.mean(0), self.y - self.y.mean(0)
        expected = np.linalg.norm(x.T @ y) ** 2 / (np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y))
        self.assertAlmostEqual(self.cka(self.x, self.y), expected, places=13)
        self.assertAlmostEqual(self.cka(self.x, self.y), self.cka(self.y, self.x), places=13)
        self.assertAlmostEqual(self.cka(self.x, self.x), 1., places=13)

    def test_translation_orthogonal_isotropic_and_joint_row_invariance(self):
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        score = self.cka(self.x, self.y)
        self.assertAlmostEqual(self.cka(self.x @ rotation + [10., -2., 7.], self.y * -3. + 4.), score, places=13)
        order = [5, 1, 3, 0, 4, 2]
        ids = [self.ids[i] for i in order]
        self.assertAlmostEqual(linear_cka(self.x[order], self.y[order], ids, ids)["cka"], score, places=13)
        self.assertGreater(abs(self.cka(self.x * [10., 1., 1.], self.y) - score), 0.001)

    def test_finite_extreme_scaling_does_not_overflow_or_underflow(self):
        score = self.cka(self.x, self.y)
        with np.errstate(all="raise"):
            self.assertAlmostEqual(self.cka(self.x * 1e300, self.y * 1e-300), score, places=12)

    def test_degenerate_inputs_are_null_with_reasons_and_serialise_strictly(self):
        for x, y, ids in ((np.ones((6, 3)), self.y, self.ids),
                          (np.zeros((6, 3)), self.y, self.ids),
                          (self.x[:2], self.y[:2], self.ids[:2])):
            result = linear_cka(x, y, ids, ids)
            self.assertFalse(result["available"])
            self.assertIsNone(result["cka"])
            self.assertTrue(result["unavailable_reason"])
            json.dumps(result, allow_nan=False)

    def test_strict_report_ids_order_and_array_validation(self):
        for bad_ids in (self.ids[::-1], ["duplicate"] * 6, self.ids[:-1],
                        [" "] + self.ids[1:], ["a\nb"] + self.ids[1:], list(range(6))):
            with self.subTest(ids=bad_ids), self.assertRaises(ValueError):
                linear_cka(self.x, self.y, self.ids, bad_ids)
        for bad in (self.x.ravel(), np.empty((6, 0)), np.full((6, 2), np.nan),
                    np.full((6, 2), np.inf), self.x.astype(complex), self.x.astype(str),
                    np.ones((MAX_ROWS + 1, 2))):
            ids = [str(i) for i in range(len(bad))]
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                linear_cka(bad, bad, ids, ids)

    def saved(self, root, name, values, meta=None, ids=None):
        path = root / name
        np.savez_compressed(path, activations=values, report_ids=np.array(self.ids if ids is None else ids),
                            metadata=np.array(json.dumps(metadata() if meta is None else meta)))
        return path

    def test_saved_provenance_versions_and_exact_exposure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left_path = self.saved(root, "left.npz", self.x)
            right_meta = metadata()
            right_meta.update(model_label="synthetic-B", checkpoint_sha256="e" * 64)
            right_path = self.saved(root, "right.npz", self.y, right_meta)
            before = {p: p.read_bytes() for p in (left_path, right_path)}
            left, right = load_saved(left_path), load_saved(right_path)
            result = compare_saved(left, right)
            self.assertTrue(result["available"])
            self.assertEqual(result["right_metadata"]["model_label"], "synthetic-B")
            self.assertEqual(before, {p: p.read_bytes() for p in before})
            for key, value in (("source_length", 64), ("pooling", "different pool"), ("site", "another site"),
                               ("visible_tokens_sha256", ["f" * 64] * 6), ("prompt_sha256", "f" * 64)):
                altered = copy.deepcopy(right)
                altered["metadata"][key] = value
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "exposure/settings differ"):
                    compare_saved(left, altered)
            for bad_meta in ({**metadata(), "data_split": "validation"},
                             {**metadata(), "visible_tokens_sha256": []},
                             {**metadata(), "model_revision": ""}):
                path = self.saved(root, "invalid.npz", self.x, bad_meta)
                with self.assertRaises(ValueError):
                    load_saved(path)

    def test_saved_loader_refuses_object_arrays_and_unexpected_members(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.saved(root, "object.npz", self.x.astype(object))
            with self.assertRaises(ValueError):
                load_saved(path)
            np.savez(root / "extra.npz", activations=self.x, report_ids=np.array(self.ids),
                     metadata=np.array(json.dumps(metadata())), extra=np.array([1]))
            with self.assertRaises(ValueError):
                load_saved(root / "extra.npz")

    def test_cli_builds_synthetic_result_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = self.saved(root, "left.npz", self.x)
            right = self.saved(root, "right.npz", self.y)
            output = root / "review.json"
            command = [sys.executable, "-B", str(Path(__file__).with_name("representation_analysis.py")),
                       "--left", str(left), "--right", str(right), "--output", str(output)]
            first = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertAlmostEqual(json.loads(output.read_text())["cka"], self.cka(self.x, self.y))
            before = output.read_bytes()
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
