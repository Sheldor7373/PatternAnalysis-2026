import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from checkpoint_storage import MARKER, publish_checkpoint, resolve_checkpoint


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        # Restricted Windows runtimes may redirect system temp to a location
        # that permits writes but not rename. Use the project's writable scratch.
        scratch = Path(__file__).parent / "runs" / "tests"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.alias = Path(self.temp.name) / "latest"

    @staticmethod
    def writer(value):
        def write(stage):
            (stage / "model.safetensors").write_text("weights " + value)
            (stage / "experiment.json").write_text(json.dumps({"step": value}))
            (stage / "trainer.pt").write_text("optimizer " + value)
        return write

    def test_completed_pointer_and_direct_path_resolve(self):
        stage = publish_checkpoint(self.alias, self.writer("1"))
        self.assertEqual(resolve_checkpoint(self.alias), stage.resolve())
        self.assertEqual(resolve_checkpoint(stage), stage)

    def test_interruption_after_weights_before_metadata_keeps_old_checkpoint(self):
        previous = publish_checkpoint(self.alias, self.writer("1"))

        def interrupted(stage):
            (stage / "model.safetensors").write_text("different partial weights")
            raise RuntimeError("simulated interruption")

        with self.assertRaises(RuntimeError):
            publish_checkpoint(self.alias, interrupted)
        self.assertEqual(resolve_checkpoint(self.alias), previous.resolve())
        for stage in self.alias.parent.joinpath("checkpoints").iterdir():
            if stage != previous:
                with self.assertRaises(ValueError):
                    resolve_checkpoint(stage)

    def test_failed_pointer_publication_keeps_old_checkpoint(self):
        previous = publish_checkpoint(self.alias, self.writer("1"))
        with patch.object(Path, "replace", side_effect=OSError("simulated interruption")):
            with self.assertRaises(OSError):
                publish_checkpoint(self.alias, self.writer("2"))
        self.assertEqual(resolve_checkpoint(self.alias), previous.resolve())

    def test_changed_weights_with_old_matching_steps_are_rejected(self):
        stage = publish_checkpoint(self.alias, self.writer("1"))
        (stage / "model.safetensors").write_text("different or partial weights")
        with self.assertRaises(ValueError):
            resolve_checkpoint(self.alias)

    def test_missing_optimizer_and_changed_marker_are_rejected(self):
        stage = publish_checkpoint(self.alias, self.writer("1"))
        (stage / "trainer.pt").unlink()
        with self.assertRaises(ValueError):
            resolve_checkpoint(self.alias)
        (stage / MARKER).write_text("{}")
        with self.assertRaises(ValueError):
            resolve_checkpoint(self.alias)
