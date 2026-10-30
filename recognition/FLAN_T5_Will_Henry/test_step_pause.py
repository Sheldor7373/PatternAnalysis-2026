"""A replay pause must preserve the complete schedule and exact continuation."""
from contextlib import redirect_stdout
import io,json,tempfile,unittest
from pathlib import Path
import torch
import test_model as mt
import test_training_transition as helpers
import train
from dataset import write_jsonl
from checkpoint_storage import resolve_checkpoint
from modules import load_checkpoint

class StepPauseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):mt.ModelTests.setUpClass()
    @classmethod
    def tearDownClass(cls):mt.ModelTests.tearDownClass()
    assert_nested_equal=helpers.TransitionIntegrationTests.assert_nested_equal
    def test_step_pause_keeps_original_schedule_and_resume_matches(self):
        with tempfile.TemporaryDirectory(dir=mt.ModelTests.path) as folder,redirect_stdout(io.StringIO()):
            root=Path(folder);data=root/'data';data.mkdir()
            rows=[{'id':str(i),'input':'No effusion','target':'Fluid absent' if i%2 else 'absent'} for i in range(11)]
            write_jsonl(data/'train.jsonl',rows);write_jsonl(data/'dev.jsonl',rows[:2])
            common=['--model',str(mt.ModelTests.path),'--data',str(data),'--mode','full','--device','cpu','--precision','fp32','--batch-size','1','--accumulate','2','--epochs','2','--cpu-threads','1','--dev-limit','2']
            train.main(train.parser().parse_args(common+['--output',str(root/'whole')]))
            train.main(train.parser().parse_args(common+['--output',str(root/'paused'),'--stop-at-step','3']))
            cp=resolve_checkpoint(root/'paused/latest');state=torch.load(cp/'trainer.pt',weights_only=False)
            self.assertEqual((state['step'],state['consumed_examples']),(3,6))
            self.assertEqual(state['scheduler']['last_epoch'],3)
            self.assertAlmostEqual(state['optimizer']['param_groups'][0]['lr'],1e-4*9/12)
            self.assertEqual(json.loads((root/'paused/training_metrics.json').read_text())['stop_reason'],'step_pause')
            self.assertEqual(json.loads((root/'paused/status.json').read_text())['state'],'stopped')
            train.main(train.parser().parse_args(common+['--output',str(root/'paused'),'--resume']))
            a=torch.load(resolve_checkpoint(root/'whole/latest')/'trainer.pt',weights_only=False)
            b=torch.load(resolve_checkpoint(root/'paused/latest')/'trainer.pt',weights_only=False)
            for k in ('optimizer','scheduler','step','consumed_examples','python_rng','torch_rng'):self.assert_nested_equal(a[k],b[k])
            x,_,_=load_checkpoint(root/'whole/latest',torch.device('cpu'));y,_,_=load_checkpoint(root/'paused/latest',torch.device('cpu'))
            self.assert_nested_equal(x.state_dict(),y.state_dict())
    def test_invalid_pause_is_rejected_before_model_loading(self):
        args=train.parser().parse_args(['--mode','full','--output','unused','--stop-at-step','0'])
        with self.assertRaisesRegex(ValueError,'stop_at_step'):train.main(args)

if __name__=='__main__':unittest.main()
