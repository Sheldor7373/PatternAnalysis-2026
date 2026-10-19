"""Synthetic CPU/stdlib contracts; no pretrained model or official evaluation."""

import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

from research_diagnostics import (Reader, build_report, compare_results, align_predictions,
                                  report_groups, bootstrap_differences, analyse_predictions, ROUGE)


def metric(mode="frozen"):
    return {"model": {"mode": mode, "model": "base", "revision": "pinned"},
            "manifest_sha256": "manifest", "selected_ids_sha256": "ids", "examples": 2,
            "generation": {"seed": 3710, "prompt": "simplify", "source_length": 384,
                           "max_new_tokens": 256, "beams": 1, "sampling": False},
            "evaluation_scope": "sampled pilot; not final evaluation", "precision": "bf16", "batch_size": 1,
            "environment": {"gpu": "test GPU"}, "rouge": {"rouge1": 0.3, "rouge2": 0.2,
                                                            "rougeL": 0.3, "rougeLsum": 0.3}}


class DiagnosticsTests(unittest.TestCase):
    def test_one_component_retains_mean_without_an_uncertainty_interval(self):
        scores = {"frozen": [{key: 0.0 for key in ROUGE} for _ in range(3)],
                  "lora": [{key: value for key in ROUGE} for value in (0.1, 0.1, 0.7)]}
        result = bootstrap_differences(scores, [[0, 1, 2]], draws=100, seed=3710)
        self.assertFalse(result["uncertainty_available"])
        self.assertEqual(result["draws"], 0)
        for interval in result["intervals"]:
            self.assertAlmostEqual(interval["mean"], 0.3)
            self.assertIsNone(interval["lower_95"])
            self.assertIsNone(interval["upper_95"])

    def test_visible_alignment_compares_nonnull_values_when_first_mode_is_missing(self):
        row = {"id": "r1", "input": "source", "target": "reference", "prediction": "output"}
        predictions = {"frozen": [dict(row)], "full": [{**row, "visible_input": "FIRST"}],
                       "lora": [{**row, "visible_input": "DIFFERENT"}]}
        with self.assertRaisesRegex(ValueError, "model-visible source differs"):
            align_predictions(predictions)
        predictions["lora"][0]["visible_input"] = "FIRST"
        self.assertEqual(len(align_predictions(predictions)), 3)
        predictions["frozen"][0]["visible_input"] = ""
        with self.assertRaisesRegex(ValueError, "model-visible source differs"):
            align_predictions(predictions)

    def setUp(self):
        scratch = Path(__file__).parent / "runs" / "tests"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def test_active_jsonl_tail_is_reported_but_complete_corruption_fails(self):
        path = self.root / "history.jsonl"
        path.write_bytes(b'{"step":1}\n{"step":')
        reader = Reader()
        self.assertEqual(reader.read(path, lines=True), [{"step": 1}])
        self.assertEqual(len(reader.warnings), 1)
        self.assertIn(str(path.resolve()), reader.sources)
        path.write_bytes(b'{"step":1}\nBROKEN\n')
        with self.assertRaises(ValueError):
            Reader().read(path, lines=True)

    def test_diagnostic_losses_do_not_become_best_selection(self):
        (self.root / "run.json").write_text(json.dumps({"mode": "lora", "arguments": {"seed": 3710}}))
        history = [{"step": 1, "dev_loss": 0.1, "selection_decision": False},
                   {"step": 2, "dev_loss": 0.5, "selection_decision": True}]
        path = self.root / "history.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in history))
        before = path.read_bytes()
        report = build_report([self.root])
        self.assertEqual(report["runs"][0]["summary"]["best_selection_dev_loss"], 0.5)
        self.assertIsNone(report["runs"][0]["summary"]["sampled_device_peak_used_mib"])
        self.assertEqual(path.read_bytes(), before)

    def test_matching_pilot_remains_pilot_and_has_no_invented_resource_values(self):
        rows = [{"path": mode, "metrics": metric(mode)} for mode in ("frozen", "full", "lora")]
        report = compare_results(rows)
        self.assertTrue(report["quality_settings_match"])
        self.assertTrue(report["inference_cost_settings_match"])
        self.assertTrue(report["all_three_modes_present"])
        self.assertIn("pilot", report["rows"][0]["evaluation_scope"])
        self.assertIsNone(report["rows"][0]["tokens_per_second"])

    def test_different_exposure_or_ids_or_seed_refuses_comparison(self):
        for key, value in [("selected_ids_sha256", "other"), ("manifest_sha256", "other"),
                           ("generation", {"seed": 42}), ("evaluation_scope", "complete manifest")]:
            rows = [{"path": "a", "metrics": metric()}, {"path": "b", "metrics": metric("lora")}]
            rows[1]["metrics"][key] = value
            with self.subTest(key=key):
                self.assertFalse(compare_results(rows)["quality_settings_match"])

    def test_missing_identity_is_not_matching_identity(self):
        rows = [{"path": mode, "metrics": metric(mode)} for mode in ("frozen", "lora")]
        for row in rows:
            del row["metrics"]["manifest_sha256"]
        self.assertFalse(compare_results(rows)["quality_settings_match"])

    def test_cost_and_quality_comparability_are_distinct(self):
        rows = [{"path": mode, "metrics": metric(mode)} for mode in ("frozen", "lora")]
        rows[1]["metrics"]["batch_size"] = 2
        result = compare_results(rows)
        self.assertTrue(result["quality_settings_match"])
        self.assertFalse(result["inference_cost_settings_match"])

    def test_checkpoint_reader_refuses_path_escape_without_reading_weights(self):
        (self.root / "latest.json").write_text(json.dumps({"directory": "../outside"}))
        with self.assertRaises(ValueError):
            build_report([self.root])

    def predictions(self):
        rows = [{"id": "a", "input": "No effusion", "target": "No fluid", "image": "X", "source": "test", "prediction": "No fluid"},
                {"id": "b", "input": " no   EFFUSION ", "target": "Fluid absent", "image": "Y", "source": "test", "prediction": "No fluid"},
                {"id": "c", "input": "Different", "target": "Different", "image": "Y", "source": "test", "prediction": ""}]
        return {"frozen": rows, "lora": [dict(row) for row in reversed(rows)]}

    def test_pairs_align_by_id_preserving_references_and_reject_corruption(self):
        predictions = self.predictions()
        self.assertEqual(len(align_predictions(predictions)["lora"]), 3)
        predictions["lora"][0]["target"] = "changed"
        with self.assertRaises(ValueError):
            align_predictions(predictions)
        predictions = self.predictions()
        predictions["frozen"][0]["visible_input"] = "No effusion"
        predictions["lora"][-1]["visible_input"] = "No"
        with self.assertRaises(ValueError):
            align_predictions(predictions)
        predictions = self.predictions()
        predictions["lora"].append(predictions["lora"][0])
        with self.assertRaises(ValueError):
            align_predictions(predictions)

    def test_bootstrap_groups_transitive_report_image_chains(self):
        rows = self.predictions()["frozen"]
        self.assertEqual(report_groups(rows), [[0, 1, 2]])
        for row in rows:
            row["image"] = ""
        self.assertEqual(report_groups(rows), [[0, 1], [2]])

    def test_bootstrap_fixed_shared_draws_and_row_weighted_mean(self):
        scores = {"frozen": [{key: 0.0 for key in ROUGE} for _ in range(3)],
                  "lora": [{key: value for key in ROUGE} for value in (0.1, 0.1, 0.7)]}
        first = bootstrap_differences(scores, [[0, 1], [2]], draws=100, seed=3710)
        self.assertEqual(first, bootstrap_differences(scores, [[0, 1], [2]], draws=100, seed=3710))
        self.assertAlmostEqual(first["intervals"][0]["mean"], 0.3)
        self.assertEqual(first["intervals"][0]["lower_95"], first["intervals"][1]["lower_95"])
        self.assertIn("not training/random-seed", first["uncertainty"])
        with self.assertRaises(ValueError):
            bootstrap_differences(scores, [[0, 1], [1, 2]], draws=10)

    def test_analysis_keeps_full_text_and_marks_unmeasured_visibility_eos(self):
        predictions = self.predictions()
        def score(row):
            return {key: 0.5 for key in ROUGE}
        analysis = analyse_predictions(predictions, score=score, draws=10)
        self.assertEqual(len(analysis["per_example"]), 6)
        self.assertTrue(any(row["empty_output"] for row in analysis["per_example"]))
        self.assertEqual(analysis["per_example"][0]["repeated_output_count"], 2)
        self.assertIsNone(analysis["per_example"][0]["output_has_eos"])
        self.assertIsNone(analysis["per_example"][0]["source_tokens"])
        case = next(row for row in analysis["manual_audit_candidates"] if row["id"] == "a")
        self.assertEqual(case["full_source"], "No effusion")
        self.assertEqual(case["full_reference"], "No fluid")
        self.assertIsNone(case["visible_source"]["lora"])
        self.assertIn("failure-selected", case["selection"])
        self.assertEqual(case["manual_review"]["lora"]["confirmed_failure"], "")
        self.assertEqual(sum(row["rows"] for row in analysis["strata"] if row["stratum"] == "source"), 6)

    def test_window_loss_is_not_reconstructed_from_cumulative_means(self):
        (self.root / "history.jsonl").write_text(json.dumps({"train_loss": 2.0, "step": 1}))
        report = build_report([self.root])
        self.assertIsNone(report["runs"][0]["summary"]["window_train_loss"])
        self.assertEqual(report["runs"][0]["summary"]["train_loss"], 2.0)

    def test_render_synthetic_measurements_and_refuse_source_output(self):
        from plot_research import render
        (self.root / "run.json").write_text(json.dumps({"mode": "lora"}))
        (self.root / "history.jsonl").write_text(json.dumps({"event": "train_window", "elapsed_seconds": 60,
                                                            "window_train_loss": 1.5,
                                                            "train_loss": 2.0, "learning_rate": 0.0001}))
        report = build_report([self.root])
        with self.assertRaises(ValueError):
            render(report, self.root / "plots")
        with tempfile.TemporaryDirectory(dir=self.root.parent) as output:
            paths = render(report, output)
            self.assertEqual(len(paths), 1)
            self.assertGreater(Path(paths[0]).stat().st_size, 1000)

    def test_health_cli_exports_status_without_touching_source(self):
        source = self.root / "source"
        source.mkdir()
        config = source / "run.json"
        config.write_text(json.dumps({"mode": "lora", "arguments": {"seed": 3710}}))
        before = config.read_bytes()
        script = Path(__file__).parent / "research_diagnostics.py"
        output = self.root / "review"
        result = subprocess.run([sys.executable, str(script), "--run", str(source), "--output", str(output)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        saved = json.loads((output / "review.json").read_text())
        self.assertEqual(saved["runs"][0]["summary"]["seed"], 3710)
        self.assertTrue((output / "status.csv").exists())
        self.assertEqual(config.read_bytes(), before)
        refused = subprocess.run([sys.executable, str(script), "--run", str(source), "--output", str(source / "review")],
                                 capture_output=True, text=True)
        self.assertNotEqual(refused.returncode, 0)
        self.assertFalse((source / "review").exists())

    def test_actual_telemetry_contract_and_supervisor_clock(self):
        from plot_research import telemetry_series
        supervisor = self.root / "supervision"
        supervisor.mkdir()
        samples = [{"timestamp_utc": "2026-10-19T11:00:00Z", "used_mib": 1000},
                   {"timestamp_utc": "2026-10-19T11:05:00Z", "used_mib": 1200}]
        (supervisor / "gpu.jsonl").write_text("\n".join(json.dumps(row) for row in samples))
        (supervisor / "supervisor.json").write_text(json.dumps({"state": "running", "checkpoint_age_seconds": 20}))
        window = {"event": "train_window", "step": 100, "elapsed_seconds": 30,
                  "window_train_loss": 1.25, "window_target_tokens": 600, "window_optimizer_steps": 10,
                  "window_seconds": 30, "window_target_tokens_per_second": 20,
                  "learning_rate": 0.0001, "next_learning_rate": 0.00009,
                  "grad_norm_preclip": 0.5, "window_max_grad_norm_preclip": 1.5,
                  "train_loss": 2.0, "train_loss_scope": "token-weighted cumulative epoch",
                  "finite_loss": True, "finite_gradients": True,
                  "peak_allocated_mib": 900, "peak_reserved_mib": 1024}
        (self.root / "history.jsonl").write_text(json.dumps(window))
        report = build_report([self.root], supervisors=[supervisor])
        summary = report["runs"][0]["summary"]
        self.assertEqual(summary["window_train_loss"], 1.25)
        self.assertEqual(summary["window_target_tokens_per_second"], 20)
        self.assertEqual(summary["peak_allocated_mib"], 900)
        self.assertEqual(summary["sampled_device_peak_used_mib"], 1200)
        self.assertEqual(summary["supervisor_state"], "running")
        self.assertEqual(summary["checkpoint_age_seconds"], 20)
        self.assertEqual(telemetry_series(samples, "used_mib"), [(0.0, 1000), (5.0, 1200)])
        self.assertIn(str((supervisor / "gpu.jsonl").resolve()), report["sources"])


if __name__ == "__main__":
    unittest.main()
