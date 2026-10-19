import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from supervise_run import run


class SupervisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = Path(__file__).parent / "runs" / "tests"
        cls.scratch.mkdir(parents=True, exist_ok=True)

    def prepare(self, root, body):
        code = root / "code"
        code.mkdir()
        script = code / "child.py"
        script.write_text(body)
        config = {"training_output": str(root / "training"), "code_directory": str(code),
                  "command": [sys.executable, "-u", str(script)],
                  "source_files_sha256": {"child.py": hashlib.sha256(script.read_bytes()).hexdigest()},
                  "poll_seconds": 0.02, "health_seconds": 0.02}
        path = root / "launch.json"
        path.write_text(json.dumps(config))
        return path

    def test_child_failure_is_recorded_and_logs_remain_available(self):
        with tempfile.TemporaryDirectory(dir=self.scratch) as folder:
            root = Path(folder)
            config = self.prepare(root, "import sys\nprint('stdout', flush=True)\nprint('stderr', file=sys.stderr, flush=True)\nsys.exit(7)\n")
            with patch("supervise_run.gpu_sample", return_value={"used_mib": 0}):
                self.assertEqual(run(config), 7)
            self.assertIn("stdout", (root / "train.stdout.log").read_text())
            self.assertIn("stderr", (root / "train.stderr.log").read_text())
            status = json.loads((root / "supervisor.json").read_text())
            self.assertEqual((status["state"], status["returncode"]), ("failed", 7))
            self.assertTrue((root / "health.jsonl").read_text().strip())

    def test_changed_snapshot_is_rejected_before_launch(self):
        with tempfile.TemporaryDirectory(dir=self.scratch) as folder:
            root = Path(folder)
            config = self.prepare(root, "print('original')\n")
            (root / "code" / "child.py").write_text("print('changed')\n")
            with self.assertRaisesRegex(ValueError, "Frozen source changed"), patch("supervise_run.subprocess.Popen") as popen:
                run(config)
            popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
