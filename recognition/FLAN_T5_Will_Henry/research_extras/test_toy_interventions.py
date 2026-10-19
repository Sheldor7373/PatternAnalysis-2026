"""Projection algebra, explicit site/span checks and toy callback lifecycle."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from toy_interventions import projection_swap, ToySite, temporary_toy_hook, run_toy


class ToyInterventionTests(unittest.TestCase):
    def setUp(self):
        self.base = np.array([[1., 2., 3.], [4., -2., 1.], [2., 0., 1.]])
        self.source = np.array([[5., 3., 7.], [-1., 3., 2.], [4., 2., -2.]])
        self.options = {"base_span": (0, 2), "source_span": (1, 3), "site": "toy.hidden",
                        "source_site": "toy.hidden", "base_mask": [True] * 3, "source_mask": [True] * 3}

    def swap(self, basis, source=None, **changes):
        return projection_swap(self.base, self.source if source is None else source, basis,
                               **{**self.options, **changes})

    def test_rank_k_swap_preserves_complement_and_outside_span_without_mutation(self):
        q = np.array([[1., 0.], [0., 1.], [0., 0.]])
        before = [a.copy() for a in (self.base, self.source, q)]
        patched = self.swap(q)
        np.testing.assert_allclose(patched[:2] @ q, self.source[1:3] @ q)
        np.testing.assert_array_equal(patched[:2, 2], self.base[:2, 2])
        np.testing.assert_array_equal(patched[2], self.base[2])
        for actual, expected in zip((self.base, self.source, q), before):
            np.testing.assert_array_equal(actual, expected)

    def test_self_patch_identity_and_full_rank_replacement(self):
        q, _ = np.linalg.qr(np.array([[1., 2., 0.], [2., -1., 1.], [0., 1., 2.]]))
        patched = self.swap(q[:, :2], source=self.base, source_span=(0, 2))
        np.testing.assert_array_equal(patched, self.base)
        np.testing.assert_allclose(self.swap(q)[:2], self.source[1:3], atol=1e-13)

    def test_nonaxis_causal_coordinate_has_known_effect_and_nuisance_has_zero(self):
        result = run_toy()
        self.assertAlmostEqual(result["effect"], -9., places=12)
        self.assertAlmostEqual(result["nuisance_control_effect"], 0., places=12)
        self.assertEqual(result["remaining_toy_hooks"], 0)
        self.assertIn("synthetic array toy", result["scope"])

    def test_site_span_mask_shape_and_basis_errors_are_rejected(self):
        q = np.eye(3)[:, :1]
        for change in ({"source_site": "another.site"}, {"site": "", "source_site": ""},
                       {"base_span": (-1, 1)}, {"base_span": (0, 4)}, {"base_span": (1, 1)},
                       {"base_span": (0, 1)}, {"base_span": (False, 2)},
                       {"base_mask": [False, True, True]}, {"source_mask": [True, True, False]},
                       {"base_mask": [1, 1, 1]}, {"source_mask": [True]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.swap(q, **change)
        for bad in (np.ones((3, 2)), np.eye(2), np.zeros((3, 1)), np.full((3, 1), np.nan),
                    np.empty((3, 0)), np.eye(3).astype(complex), np.ones((3, 4))):
            with self.subTest(basis=bad), self.assertRaises(ValueError):
                self.swap(bad)
        with self.assertRaises(ValueError):
            self.swap(q, source=np.ones((3, 4)))

    def test_overflow_fails_without_mutating_inputs(self):
        base, source = np.full((1, 1), -1e308), np.full((1, 1), 1e308)
        with self.assertRaisesRegex(ValueError, "overflowed"):
            projection_swap(base, source, [[1.]], base_span=(0, 1), source_span=(0, 1),
                            site="toy", source_site="toy", base_mask=[True], source_mask=[True])
        self.assertEqual(base[0, 0], -1e308)
        self.assertEqual(source[0, 0], 1e308)

    def test_toy_hook_cleanup_on_success_exception_and_nested_context(self):
        site = ToySite("toy.hidden")
        with temporary_toy_hook(site, lambda value: value + 1):
            self.assertEqual(site.active_hooks, 1)
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                with temporary_toy_hook(site, lambda value: (_ for _ in ()).throw(RuntimeError("synthetic failure"))):
                    site.forward(self.base)
            self.assertEqual(site.active_hooks, 1)
            np.testing.assert_array_equal(site.forward(self.base), self.base + 1)
        self.assertEqual(site.active_hooks, 0)
        np.testing.assert_array_equal(site.forward(self.base), self.base)
        with self.assertRaises(TypeError):
            with temporary_toy_hook(object(), lambda value: value):
                pass
        with self.assertRaises(ValueError):
            with temporary_toy_hook(site, lambda value: value[:1]):
                site.forward(self.base)
        self.assertEqual(site.active_hooks, 0)

    def test_cli_exports_toy_only_and_refuses_existing_result(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "toy.json"
            command = [sys.executable, "-B", str(Path(__file__).with_name("toy_interventions.py")),
                       "--output", str(output)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertAlmostEqual(json.loads(output.read_text())["effect"], -9.)
            before = output.read_bytes()
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
