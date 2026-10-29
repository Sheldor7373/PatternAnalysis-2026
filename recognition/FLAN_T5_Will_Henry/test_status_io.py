"""CPU/stdlib status publication and actual Windows reader-sharing regressions."""
import ast
import errno
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SNAPSHOT = Path(__file__).parent
sys.path.insert(0, str(SNAPSHOT))
import status_io as io


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def windows_error(code):
    error = PermissionError(errno.EACCES, 'simulated Windows contention')
    error.winerror = code
    return error


class StatusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'status.json'
        self.path.write_text('{"step": 10}', encoding='utf-8')

    def test_real_success_and_unicode_json_reader(self):
        io.atomic_json(self.path, {'step': 11, 'text': 'radiology \u8f90\u5c04'})
        self.assertEqual(read_json(self.path), {'step': 11, 'text': 'radiology \u8f90\u5c04'})
        self.assertEqual(list(self.root.glob('*.tmp')), [])

    def test_only_known_windows_replace_errors_retry_then_publish(self):
        real_replace = os.replace
        for code in (5, 32, 33):
            with self.subTest(code=code):
                calls = []
                def replace(source, dest):
                    calls.append((source, dest))
                    if len(calls) == 1:
                        self.assertEqual(read_json(dest)['step'], 10)
                        raise windows_error(code)
                    return real_replace(source, dest)
                self.path.write_text('{"step": 10}')
                with patch.object(io.os, 'replace', side_effect=replace), patch.object(io.time, 'sleep') as sleep:
                    io.atomic_json(self.path, {'step': 11})
                sleep.assert_called_once_with(0.05)
                self.assertEqual(len(calls), 2)
                self.assertEqual(read_json(self.path)['step'], 11)

    def test_retry_exhaustion_is_bounded_retains_old_and_complete_new_temp(self):
        error = windows_error(5)
        with patch.object(io.os, 'replace', side_effect=error) as replace, patch.object(io.time, 'sleep') as sleep:
            with self.assertRaises(PermissionError) as raised:
                io.atomic_json(self.path, {'step': 11})
        self.assertIs(raised.exception, error)
        self.assertEqual(replace.call_count, 6)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], list(io.RETRY_DELAYS))
        self.assertAlmostEqual(sum(io.RETRY_DELAYS), 1.55)
        self.assertEqual(read_json(self.path)['step'], 10)
        temps = list(self.root.glob('.*.tmp'))
        self.assertEqual(len(temps), 1)
        self.assertEqual(json.loads(temps[0].read_text())['step'], 11)

    def test_non_sharing_error_is_immediate_and_not_suppressed(self):
        for error in (OSError(errno.ENOSPC, 'disk full'), windows_error(112), PermissionError(errno.EACCES, 'no Windows code')):
            with self.subTest(error=error), patch.object(io.os, 'replace', side_effect=error) as replace, patch.object(io.time, 'sleep') as sleep:
                with self.assertRaises(OSError):
                    io.atomic_json(self.path, {'step': 11})
                replace.assert_called_once()
                sleep.assert_not_called()
                self.assertEqual(read_json(self.path)['step'], 10)

    def test_write_or_fsync_failure_does_not_attempt_replace(self):
        with patch.object(io.os, 'fsync', side_effect=OSError(errno.EIO, 'I/O failure')), patch.object(io.os, 'replace') as replace:
            with self.assertRaises(OSError):
                io.atomic_json(self.path, {'step': 11})
        replace.assert_not_called()
        self.assertEqual(read_json(self.path)['step'], 10)

    def test_serialization_failure_does_not_create_stage(self):
        with self.assertRaises(TypeError):
            io.atomic_json(self.path, {'step': object()})
        self.assertEqual(list(self.root.glob('*.tmp')), [])
        self.assertEqual(read_json(self.path)['step'], 10)

    def test_failed_stages_are_unique_and_never_truncated(self):
        with patch.object(io.os, 'replace', side_effect=windows_error(5)), patch.object(io.time, 'sleep'):
            for step in (11, 12):
                with self.assertRaises(PermissionError):
                    io.atomic_json(self.path, {'step': step})
        stages = list(self.root.glob('.*.tmp'))
        self.assertEqual(len(stages), 2)
        self.assertEqual(sorted(json.loads(p.read_text())['step'] for p in stages), [11, 12])

    def test_missing_or_invalid_json_reader_closes_handle_and_surfaces_error(self):
        with self.assertRaises(OSError):
            read_json(self.root / 'missing.json')
        self.path.write_text('invalid')
        with self.assertRaises(json.JSONDecodeError):
            read_json(self.path)
        io.atomic_json(self.path, {'step': 12})
        self.assertEqual(read_json(self.path)['step'], 12)

    @unittest.skipUnless(os.name == 'nt', 'Windows sharing semantics')
    def test_actual_ordinary_reader_blocks_replace_then_retry_recovers(self):
        reader = self.path.open('rb')
        self.addCleanup(reader.close)
        with patch.object(io.time, 'sleep', side_effect=lambda seconds: reader.close()) as sleep:
            io.atomic_json(self.path, {'step': 12})
        sleep.assert_called_once_with(0.05)
        self.assertEqual(read_json(self.path)['step'], 12)

    def test_trainer_status_wrapper_and_supervisor_share_exact_helper(self):
        tree = ast.parse((SNAPSHOT / 'train.py').read_text(encoding='utf-8'))
        function = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'atomic_status')
        calls = []
        namespace = {'atomic_json': lambda path, value: calls.append((path, value))}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<status wrapper>', 'exec'), namespace)
        namespace['atomic_status'](self.root, {'step': 12})
        self.assertEqual(calls, [(self.path, {'step': 12})])
        supervisor = ast.parse((SNAPSHOT / 'supervise_run.py').read_text(encoding='utf-8'))
        imports = [x for x in supervisor.body if isinstance(x, ast.ImportFrom) and x.module == 'status_io']
        self.assertEqual({n.name for x in imports for n in x.names}, {'atomic_json'})
        self.assertFalse(any(isinstance(x, ast.FunctionDef) and x.name == 'atomic_json' for x in supervisor.body))


if __name__ == '__main__':
    unittest.main()
