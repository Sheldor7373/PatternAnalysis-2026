"""CPU-only artifact and runtime-identity regressions; no remote downloads."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from inference_identity import artifact_manifest,bind_source,verify_source,resolve_precision,effective_runtime


class InferenceIdentityTests(unittest.TestCase):
    def test_binary_weights_indexed_shards_and_tokenizer_are_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'weights').mkdir()
            (root/'pytorch_model.bin').write_bytes(b'base weights')
            (root/'weights/part.bin').write_bytes(b'shard A')
            (root/'tokenizer.json').write_text('{"version":"A"}')
            (root/'pytorch_model.bin.index.json').write_text(json.dumps({'weight_map':{'x':'weights/part.bin'}}))
            bound=bind_source(str(root),'main')
            files=bound['identity']['files_sha256']
            self.assertIn('pytorch_model.bin',files)
            self.assertIn('weights/part.bin',files)
            self.assertIn('tokenizer.json',files)
            (root/'weights/part.bin').write_bytes(b'shard B')
            self.assertNotEqual(artifact_manifest(root),files)
            with self.assertRaisesRegex(ValueError,'changed while'):
                verify_source(bound)

    def test_remote_model_and_tokenizer_share_resolved_commit(self):
        with patch('inference_identity.AutoConfig.from_pretrained',return_value=SimpleNamespace(_commit_hash='1'*40)) as loader:
            bound=bind_source('example/not-local','main')
        self.assertEqual(bound['revision'],'1'*40)
        self.assertEqual(bound['identity']['model_revision'],bound['identity']['tokenizer_revision'])
        loader.assert_called_once_with('example/not-local',revision='main')
        with patch('inference_identity.AutoConfig.from_pretrained',return_value=SimpleNamespace(_commit_hash=None)):
            with self.assertRaisesRegex(ValueError,'immutable'):
                bind_source('example/not-local','main')

    def test_auto_precision_resolves_capability_without_gpu_execution(self):
        self.assertEqual(resolve_precision('auto','cuda',True),'bf16')
        self.assertEqual(resolve_precision('auto','cuda',False),'fp32')
        self.assertEqual(resolve_precision('auto','cpu',False),'fp32')
        self.assertEqual(resolve_precision('fp32','cuda',True),'fp32')
        with self.assertRaisesRegex(ValueError,'unsupported'):
            resolve_precision('bf16','cpu',False)
        with self.assertRaisesRegex(ValueError,'Unknown'):
            resolve_precision('fp16','cpu',False)

    def test_effective_cpu_runtime_records_actual_thread_settings(self):
        runtime=effective_runtime('auto',torch.device('cpu'))
        self.assertEqual(runtime['effective_precision'],'fp32')
        self.assertEqual(runtime['cpu_threads'],torch.get_num_threads())
        self.assertEqual(runtime['cpu_interop_threads'],torch.get_num_interop_threads())
        self.assertEqual(runtime['device_type'],'cpu')


if __name__=='__main__':
    unittest.main()
