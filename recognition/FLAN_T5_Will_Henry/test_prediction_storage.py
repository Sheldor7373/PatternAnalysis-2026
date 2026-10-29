"""CPU/stdlib failure-recovery tests; no model or GPU required."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from prediction_storage import PredictionRun


class PredictionStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'result'
        self.rows = [{'id': str(i), 'input': 'source '+str(i), 'target': 'reference'} for i in range(5)]
        self.identity = {'checkpoint_sha256': 'pinned', 'manifest_sha256': 'data', 'batch_size': 2}

    def batch(self, start, count=2):
        return {'start': start, 'rows': [{**row, 'prediction': 'summary'} for row in self.rows[start:start+count]],
                'generation_seconds': .1, 'wall_seconds': .2, 'generated_tokens': count*3}

    def test_interrupted_run_recovers_exact_prefix_and_preserves_final_order(self):
        with PredictionRun(self.root) as run:
            run.configure(self.identity, {'original_environment': True}, self.rows)
            run.commit(self.batch(0))
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                run.export_predictions()
        with PredictionRun(self.root, resume=True) as run:
            run.configure(self.identity, {'new_environment': True}, self.rows)
            self.assertEqual(len(run.predictions), 2)
            self.assertEqual(run.metadata, {'original_environment': True})
            run.commit(self.batch(2))
            run.commit(self.batch(4, 1))
            path = run.export_predictions()
            self.assertEqual([json.loads(line)['id'] for line in path.read_text().splitlines()], ['0','1','2','3','4'])

    def test_crash_after_batch_publication_before_progress_keeps_committed_work(self):
        with PredictionRun(self.root) as run:
            run.configure(self.identity, {}, self.rows)
            with patch.object(run, 'progress', side_effect=RuntimeError('simulated crash')):
                with self.assertRaises(RuntimeError):
                    run.commit(self.batch(0))
        with PredictionRun(self.root, resume=True) as run:
            run.configure(self.identity, {}, self.rows)
            self.assertEqual(len(run.predictions), 2)

    def test_changed_model_data_or_batch_refuses_resume(self):
        with PredictionRun(self.root) as run:
            run.configure(self.identity, {}, self.rows)
        for key in self.identity:
            with self.subTest(key=key), PredictionRun(self.root, resume=True) as run:
                with self.assertRaisesRegex(ValueError, 'identity changed'):
                    run.configure({**self.identity, key:'different'}, {}, self.rows)

    def test_corrupt_batch_fails_before_any_generation_can_continue(self):
        with PredictionRun(self.root) as run:
            run.configure(self.identity, {}, self.rows)
            run.commit(self.batch(0))
        path = self.root/'batches/00000000.json'
        document=json.loads(path.read_text())
        document['payload']['rows'][0]['prediction']='tampered'
        path.write_text(json.dumps(document))
        with PredictionRun(self.root, resume=True) as run:
            with self.assertRaisesRegex(ValueError,'checksum'):
                run.configure(self.identity, {}, self.rows)

    def test_gap_duplicate_and_changed_source_are_rejected(self):
        with PredictionRun(self.root) as run:
            run.configure(self.identity, {}, self.rows)
            with self.assertRaisesRegex(ValueError,'contiguous'):
                run.commit(self.batch(2))
            wrong=self.batch(0)
            wrong['rows'][0]['input']='wrong source'
            with self.assertRaisesRegex(ValueError,'differ'):
                run.commit(wrong)
            run.commit(self.batch(0))
            with self.assertRaisesRegex(ValueError,'contiguous'):
                run.commit(self.batch(0))

    def test_orphan_temporary_file_does_not_become_a_committed_batch(self):
        with PredictionRun(self.root) as run:
            run.configure(self.identity, {}, self.rows)
        (self.root/'batches/.00000000.json.dead.tmp').write_text('partial')
        with PredictionRun(self.root,resume=True) as run:
            run.configure(self.identity, {}, self.rows)
            self.assertEqual(run.predictions, [])

    def test_fresh_run_does_not_overwrite_existing_outputs(self):
        self.root.mkdir()
        (self.root/'metrics.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'not empty'):
            with PredictionRun(self.root):
                pass
        with PredictionRun(self.root,resume=True) as run:
            with self.assertRaisesRegex(ValueError,'preserving'):
                run.configure(self.identity, {}, self.rows)


if __name__ == '__main__':
    unittest.main()
