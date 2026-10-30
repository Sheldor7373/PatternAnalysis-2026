"""Offline safeguards for an explicit extra epoch with preserved optimizer lineage."""
from contextlib import redirect_stdout
import io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
import test_model as mt
import train
from dataset import write_jsonl
from checkpoint_storage import resolve_checkpoint,file_digests
from modules import load_checkpoint
from training_transition import epoch_extension_policy,validate_arguments,lr_factor
from test_training_transition import TransitionIntegrationTests as Helpers

class EpochExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):mt.ModelTests.setUpClass()
    @classmethod
    def tearDownClass(cls):mt.ModelTests.tearDownClass()
    assert_nested_equal=Helpers.assert_nested_equal

    def setup_source(self,root):
        data=root/'data';data.mkdir()
        rows=[{'id':str(i),'input':'No effusion','target':'Fluid absent' if i%2 else 'absent'} for i in range(11)]
        write_jsonl(data/'train.jsonl',rows);write_jsonl(data/'dev.jsonl',rows[:2])
        args=['--model',str(mt.ModelTests.path),'--data',str(data),'--mode','full','--device','cpu','--precision','fp32','--batch-size','4','--accumulate','1','--epochs','3','--cpu-threads','1','--dev-limit','2','--log-every','1']
        train.main(train.parser().parse_args(args+['--output',str(root/'source')]))
        parent=resolve_checkpoint(root/'source/latest')
        extra=args+['--epochs','4','--lr','0.00003','--transition-from',str(parent),'--restart-schedule']
        return parent,extra

    def state(self,root):return torch.load(resolve_checkpoint(root/'latest')/'trainer.pt',weights_only=False,map_location='cpu')

    def test_handoff_preserves_weights_moments_rng_and_exposure_changes_only_schedule(self):
        with tempfile.TemporaryDirectory(dir=mt.ModelTests.path) as folder,redirect_stdout(io.StringIO()):
            root=Path(folder);parent,extra=self.setup_source(root);old=self.state(root/'source');hashes=file_digests(parent)
            real=train.append_jsonl
            def stop(path,entry):
                real(path,entry)
                if entry['event']=='transition_handoff':raise RuntimeError('verified handoff')
            with patch('train.append_jsonl',side_effect=stop),self.assertRaisesRegex(RuntimeError,'verified handoff'):
                train.main(train.parser().parse_args(extra+['--output',str(root/'handoff')]))
            new=self.state(root/'handoff')
            for key in ('python_rng','torch_rng','cuda_rng','step','best','best_step','stale'):
                self.assert_nested_equal(old[key],new[key])
            self.assert_nested_equal(old['optimizer']['state'],new['optimizer']['state'])
            self.assertEqual((new['epoch'],new['next_sample'],new['consumed_examples']),(3,0,33))
            self.assertEqual(new['optimizer']['param_groups'][0]['lr'],3e-5)
            self.assertEqual(new['scheduler']['last_epoch'],old['step'])
            self.assertEqual(new['scheduler']['base_lrs'],[3e-5])
            self.assertEqual(new['scheduler_policy']['planned_steps'],12)
            a,_,_=load_checkpoint(parent,torch.device('cpu'));b,_,_=load_checkpoint(root/'handoff/latest',torch.device('cpu'))
            self.assert_nested_equal(a.state_dict(),b.state_dict());self.assertEqual(hashes,file_digests(parent))

    def test_extra_epoch_resume_matches_uninterrupted_and_terminates_exactly(self):
        with tempfile.TemporaryDirectory(dir=mt.ModelTests.path) as folder,redirect_stdout(io.StringIO()):
            root=Path(folder);_,extra=self.setup_source(root)
            train.main(train.parser().parse_args(extra+['--output',str(root/'whole')]))
            train.main(train.parser().parse_args(extra+['--output',str(root/'resumed'),'--max-minutes','0.0000001']))
            train.main(train.parser().parse_args(extra+['--output',str(root/'resumed'),'--resume']))
            a,b=self.state(root/'whole'),self.state(root/'resumed')
            self.assertEqual((b['step'],b['epoch'],b['next_sample'],b['consumed_examples']),(12,4,0,44))
            for key in ('optimizer','scheduler','scheduler_policy','best','best_step','stale'):self.assert_nested_equal(a[key],b[key])
            x,_,_=load_checkpoint(root/'whole/latest',torch.device('cpu'));y,_,_=load_checkpoint(root/'resumed/latest',torch.device('cpu'))
            self.assert_nested_equal(x.state_dict(),y.state_dict())
            self.assertEqual(b['optimizer']['param_groups'][0]['lr'],0)

    def test_reject_silent_extension_incomplete_source_and_batch_change(self):
        old={'epochs':3,'lr':1e-4,'batch_size':4,'accumulate':1,'max_steps':None}
        new={**old,'epochs':4,'lr':3e-5,'restart_schedule':True}
        with self.assertRaises(ValueError):validate_arguments(new,old,transition=True)
        validate_arguments(new,old,transition=True,schedule_restart=True)
        state={'epoch':3,'next_sample':0,'next_batch':0,'consumed_examples':33,'step':9,'optimizer':{'param_groups':[{'lr':0}]},'scheduler':{'last_epoch':9,'_last_lr':[0]}}
        policy=epoch_extension_policy(state,old,new,11)
        self.assertEqual(lr_factor(policy,9),1);self.assertEqual(lr_factor(policy,12),0)
        for change in ({'epochs':5},{'batch_size':8},{'lr':float('nan')}):
            with self.assertRaises(ValueError):epoch_extension_policy(state,old,{**new,**change},11)
        with self.assertRaises(ValueError):epoch_extension_policy({**state,'epoch':2},old,new,11)

if __name__=='__main__':unittest.main()
