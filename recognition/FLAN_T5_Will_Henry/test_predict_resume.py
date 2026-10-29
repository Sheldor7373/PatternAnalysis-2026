"""Actual tiny offline T5 generation on CPU; no training data or GPU use."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from transformers import T5ForConditionalGeneration
from checkpoint_storage import mark_complete
from dataset import write_jsonl
from modules import RadiologySimplifier
from test_model import ModelTests
import predict


class PredictionResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ModelTests.setUpClass()

    @classmethod
    def tearDownClass(cls):
        ModelTests.tearDownClass()

    def arguments(self, root, output, mode='frozen'):
        manifest=root/'dev.jsonl'
        if not manifest.exists():
            write_jsonl(manifest,[{'id':str(i),'input':'No effusion' if i%2 else 'No',
                                  'target':'Fluid absent','source':'test'} for i in range(5)])
        argv=['--manifest',str(manifest),'--output',str(output),'--device','cpu',
              '--precision','fp32','--batch-size','2','--source-length','12',
              '--target-length','4','--cpu-threads','1']
        if mode=='frozen':
            argv+=['--model',str(ModelTests.path)]
        else:
            checkpoint=root/mode
            if not checkpoint.exists():
                model=RadiologySimplifier(str(ModelTests.path),mode=mode,rank=2)
                model.save(checkpoint,ModelTests.tokenizer,
                           {'model':str(ModelTests.path),'revision':'main','mode':mode,'rank':2})
                mark_complete(checkpoint)
            argv+=['--checkpoint',str(checkpoint)]
        return predict.parser().parse_args(argv)

    def test_full_lora_and_frozen_resume_match_uninterrupted_outputs(self):
        for mode in ('frozen','full','lora'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(dir=ModelTests.path) as folder, redirect_stdout(io.StringIO()):
                root=Path(folder)
                whole=root/'whole'
                expected=predict.main(self.arguments(root,whole,mode))
                resumed=root/'resumed'
                args=self.arguments(root,resumed,mode)
                args.max_minutes=0
                partial=predict.main(args)
                self.assertEqual(partial['committed_examples'],2)
                self.assertFalse((resumed/'metrics.json').exists())
                self.assertFalse((resumed/'predictions.jsonl').exists())
                args.resume=True
                args.max_minutes=None
                actual=predict.main(args)
                self.assertEqual((whole/'predictions.jsonl').read_text(),(resumed/'predictions.jsonl').read_text())
                self.assertEqual(actual['rouge'],expected['rouge'])
                self.assertEqual(actual['generated_tokens'],expected['generated_tokens'])
                self.assertEqual(actual['examples'],5)
                self.assertEqual(actual['resumable_batch_count'],3)
                self.assertEqual(actual['selected_ids_sha256'],expected['selected_ids_sha256'])
                rows=[json.loads(line) for line in (resumed/'predictions.jsonl').read_text().splitlines()]
                self.assertTrue(all('visible_input' in row and 'output_has_eos' in row for row in rows))
                self.assertEqual(json.loads((resumed/'generation_status.json').read_text())['state'],'completed')

    def test_completed_resume_does_not_generate_again_and_changed_decoder_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ModelTests.path) as folder, redirect_stdout(io.StringIO()):
            root=Path(folder)
            args=self.arguments(root,root/'result')
            predict.main(args)
            original=(Path(args.output)/'predictions.jsonl').read_bytes()
            args.resume=True
            with patch.object(T5ForConditionalGeneration,'generate',side_effect=AssertionError('must not regenerate')):
                predict.main(args)
            self.assertEqual((Path(args.output)/'predictions.jsonl').read_bytes(),original)
            args.beams=2
            with self.assertRaisesRegex(ValueError,'identity changed'):
                predict.main(args)
            self.assertEqual((Path(args.output)/'predictions.jsonl').read_bytes(),original)

    def test_generation_failure_keeps_prior_batch_for_resume(self):
        with tempfile.TemporaryDirectory(dir=ModelTests.path) as folder, redirect_stdout(io.StringIO()):
            root=Path(folder)
            args=self.arguments(root,root/'result')
            original=T5ForConditionalGeneration.generate
            calls=[]
            def fail_second(model,*a,**kw):
                calls.append(1)
                if len(calls)==2:
                    raise RuntimeError('simulated inference failure')
                return original(model,*a,**kw)
            with patch.object(T5ForConditionalGeneration,'generate',new=fail_second):
                with self.assertRaisesRegex(RuntimeError,'inference failure'):
                    predict.main(args)
            self.assertEqual(len(list((Path(args.output)/'batches').glob('*.json'))),1)
            args.resume=True
            result=predict.main(args)
            self.assertEqual(result['examples'],5)
            self.assertEqual(len((Path(args.output)/'predictions.jsonl').read_text().splitlines()),5)


if __name__=='__main__':
    unittest.main()
