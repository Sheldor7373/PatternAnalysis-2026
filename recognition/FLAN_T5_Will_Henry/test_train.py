"""Exercise training and mid-epoch resume using a small random offline model."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
import torch

from dataset import write_jsonl
from modules import load_checkpoint
import test_model as model_tests
from train import main, parser


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_tests.ModelTests.setUpClass()

    @classmethod
    def tearDownClass(cls):
        model_tests.ModelTests.tearDownClass()

    def test_resume_keeps_optimizer_progress_and_final_weights(self):
        with tempfile.TemporaryDirectory() as folder:
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
            with redirect_stdout(io.StringIO()):
                main(parser().parse_args(common + ["--output", str(whole)]))
                main(parser().parse_args(common + ["--output", str(resumed), "--max-minutes", "0.0000001"]))
                first = json.loads((resumed / "training_metrics.json").read_text())
                self.assertEqual(first["optimizer_steps"], 1)
                self.assertEqual(first["stop_reason"], "time_budget")
                checkpoint_settings_path = resumed / "latest" / "experiment.json"
                checkpoint_settings = json.loads(checkpoint_settings_path.read_text())
                checkpoint_settings["revision"] = "original-pinned-sha"
                checkpoint_settings_path.write_text(json.dumps(checkpoint_settings))
                main(parser().parse_args(common + ["--output", str(resumed), "--resume"]))
            final = json.loads((resumed / "training_metrics.json").read_text())
            self.assertEqual(final["optimizer_steps"], 4)
            self.assertEqual(final["revision"], "original-pinned-sha")
            left, _, _ = load_checkpoint(whole / "latest", torch.device("cpu"))
            right, _, _ = load_checkpoint(resumed / "latest", torch.device("cpu"))
            for name, tensor in left.state_dict().items():
                torch.testing.assert_close(tensor, right.state_dict()[name], atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
