"""Exercise training and mid-epoch resume using a small random offline model."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from dataset import write_jsonl
from modules import RadiologySimplifier, load_checkpoint
import test_model as model_tests
from train import main, parser
import train
from checkpoint_storage import publish_checkpoint, resolve_checkpoint


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_tests.ModelTests.setUpClass()

    @classmethod
    def tearDownClass(cls):
        model_tests.ModelTests.tearDownClass()

    def test_resume_keeps_optimizer_progress_and_final_weights(self):
        with tempfile.TemporaryDirectory(dir=model_tests.ModelTests.path) as folder:
            root = Path(folder)
            data = root / "data"
            data.mkdir()
            rows = [{"id": f"train:{i}", "input": "No effusion", "target": "Fluid absent" if i % 2 else "absent"}
                    for i in range(6)]
            write_jsonl(data / "train.jsonl", rows)
            write_jsonl(data / "dev.jsonl", rows[:2])
            common = ["--model", str(model_tests.ModelTests.path), "--data", str(data), "--mode", "full", "--device", "cpu",
                      "--batch-size", "1", "--accumulate", "2", "--epochs", "2", "--max-steps", "4",
                      "--dev-limit", "2", "--cpu-threads", "1", "--precision", "fp32"]
            whole, resumed = root / "whole", root / "resumed"

            def initially_pinned(*args, **kwargs):
                model = RadiologySimplifier(*args, **kwargs)
                model.model.config._commit_hash = "original-pinned-sha"
                return model

            def local_reload(*args, **kwargs):
                model, tokenizer, settings = load_checkpoint(*args, **kwargs)
                model.model.config._commit_hash = None
                return model, tokenizer, settings

            with redirect_stdout(io.StringIO()):
                with patch("train.RadiologySimplifier", side_effect=initially_pinned):
                    main(parser().parse_args(common + ["--output", str(whole)]))
                    main(parser().parse_args(common + ["--output", str(resumed), "--max-minutes", "0.0000001"]))
                first = json.loads((resumed / "training_metrics.json").read_text())
                self.assertEqual(first["optimizer_steps"], 1)
                self.assertEqual(first["stop_reason"], "time_budget")
                with patch("train.load_checkpoint", side_effect=local_reload):
                    main(parser().parse_args(common + ["--output", str(resumed), "--resume"]))
            final = json.loads((resumed / "training_metrics.json").read_text())
            self.assertEqual(final["optimizer_steps"], 4)
            self.assertEqual(final["revision"], "original-pinned-sha")
            whole_decisions = [r["step"] for r in map(json.loads, (whole / "history.jsonl").read_text().splitlines())
                               if r.get("selection_decision")]
            resumed_decisions = [r["step"] for r in map(json.loads, (resumed / "history.jsonl").read_text().splitlines())
                                 if r.get("selection_decision")]
            self.assertEqual(whole_decisions, resumed_decisions)
            left, _, _ = load_checkpoint(whole / "latest", torch.device("cpu"))
            right, _, _ = load_checkpoint(resumed / "latest", torch.device("cpu"))
            for name, tensor in left.state_dict().items():
                torch.testing.assert_close(tensor, right.state_dict()[name], atol=1e-6, rtol=1e-5)

    def run_arguments(self, root):
        data = root / "data"
        data.mkdir()
        rows = [{"id": f"train:{i}", "input": "No effusion", "target": "Fluid absent" if i % 2 else "absent"}
                for i in range(6)]
        write_jsonl(data / "train.jsonl", rows)
        write_jsonl(data / "dev.jsonl", rows[:2])
        return ["--model", str(model_tests.ModelTests.path), "--data", str(data), "--mode", "full",
                "--device", "cpu", "--batch-size", "1", "--accumulate", "2", "--epochs", "2",
                "--max-steps", "4", "--dev-limit", "2", "--cpu-threads", "1", "--precision", "fp32",
                "--checkpoint-minutes", "0.000000001", "--log-every", "1"]

    def test_periodic_saves_do_not_select_or_force_development_evaluation(self):
        with tempfile.TemporaryDirectory(dir=model_tests.ModelTests.path) as folder:
            root = Path(folder)
            common = self.run_arguments(root)
            output = root / "run"
            periodic_states = []

            def capture(alias, writer):
                stage = publish_checkpoint(alias, writer)
                if alias.name == "latest":
                    state = torch.load(stage / "trainer.pt", weights_only=False)
                    if state["step"] in (1, 2):
                        periodic_states.append(state)
                return stage

            with redirect_stdout(io.StringIO()), patch("train.publish_checkpoint", side_effect=capture), \
                    patch("train.evaluate_loss", wraps=train.evaluate_loss) as evaluate:
                main(parser().parse_args(common + ["--output", str(output)]))
            self.assertEqual(evaluate.call_count, 2)
            self.assertEqual([s["step"] for s in periodic_states], [1, 2])
            self.assertEqual([s["next_batch"] for s in periodic_states], [2, 4])
            for state in periodic_states:
                self.assertEqual(state["best"], float("inf"))
                self.assertIsNone(state["best_step"])
                self.assertEqual(state["stale"], 0)
            events = list(map(json.loads, (output / "history.jsonl").read_text().splitlines()))
            windows = [r for r in events if r["event"] == "train_window"]
            self.assertEqual(len(windows), 4)
            self.assertTrue(all(r["finite_loss"] and r["finite_gradients"] for r in windows))
            self.assertTrue(all(r["window_optimizer_steps"] == 1 for r in windows))
            self.assertEqual(sum(r["window_target_tokens"] for r in windows),
                             json.loads((output / "training_metrics.json").read_text())["session_target_tokens"])

    def test_abrupt_interruption_after_periodic_save_resumes_with_dropout(self):
        import shutil
        with tempfile.TemporaryDirectory(dir=model_tests.ModelTests.path) as folder:
            root = Path(folder)
            common = self.run_arguments(root)
            base = root / "base"
            base.mkdir()
            for source in model_tests.ModelTests.path.iterdir():
                if source.is_file():
                    shutil.copy2(source, base / source.name)
            config = json.loads((base / "config.json").read_text())
            config["dropout_rate"] = 0.2
            (base / "config.json").write_text(json.dumps(config))
            common[common.index("--model") + 1] = str(base)
            common += ["--eval-minutes", "0.000000001", "--eval-at-start"]
            whole, resumed = root / "whole", root / "resumed"

            def fail_after_publish(alias, writer):
                stage = publish_checkpoint(alias, writer)
                if alias.name == "latest":
                    raise InterruptedError("Simulated process loss after a completed periodic checkpoint")
                return stage

            with redirect_stdout(io.StringIO()):
                main(parser().parse_args(common + ["--output", str(whole)]))
                with patch("train.publish_checkpoint", side_effect=fail_after_publish), self.assertRaises(InterruptedError):
                    main(parser().parse_args(common + ["--output", str(resumed)]))
                state = torch.load(resolve_checkpoint(resumed / "latest") / "trainer.pt", weights_only=False)
                self.assertEqual((state["step"], state["next_batch"]), (1, 2))
                main(parser().parse_args(common + ["--output", str(resumed), "--resume"]))
            left, _, _ = load_checkpoint(whole / "latest", torch.device("cpu"))
            right, _, _ = load_checkpoint(resumed / "latest", torch.device("cpu"))
            for name, tensor in left.state_dict().items():
                torch.testing.assert_close(tensor, right.state_dict()[name], atol=1e-6, rtol=1e-5)

    def test_stop_request_finishes_accumulation_and_saves_resumable_state(self):
        with tempfile.TemporaryDirectory(dir=model_tests.ModelTests.path) as folder:
            root = Path(folder)
            common = self.run_arguments(root)
            output = root / "run"
            original = RadiologySimplifier.forward

            def request_after_microbatch(model, **batch):
                result = original(model, **batch)
                (output / "STOP").touch()
                return result

            with redirect_stdout(io.StringIO()), patch.object(RadiologySimplifier, "forward", request_after_microbatch):
                main(parser().parse_args(common + ["--output", str(output)]))
            result = json.loads((output / "training_metrics.json").read_text())
            self.assertEqual((result["optimizer_steps"], result["stop_reason"]), (1, "stop_requested"))
            state = torch.load(resolve_checkpoint(output / "latest") / "trainer.pt", weights_only=False)
            self.assertEqual(state["next_batch"], 2)
            self.assertEqual(state["stale"], 0)
            self.assertEqual(json.loads((output / "status.json").read_text())["state"], "stopped")


if __name__ == "__main__":
    unittest.main()
