import hashlib,json,tempfile,unittest
from pathlib import Path
from supervise_prediction import check_training_gate

class ExtendedPredictionGateTests(unittest.TestCase):
    def setup_run(self,root,epochs=4,exposures=389644,step=117802):
        (root/'training').mkdir();cp=root/'immutable';cp.mkdir();(cp/'COMPLETE.json').write_text('verified-marker')
        for path,value in [('supervisor.json',{'state':'completed','returncode':0}),('training/status.json',{'state':'completed','step':step}),('training/training_metrics.json',{'optimizer_steps':step}),('receipt.json',{'optimizer_steps':step,'verified_complete':True,'epochs':epochs,'exposures':exposures,'latest_checkpoint':str(cp),'latest_completion_sha256':hashlib.sha256((cp/'COMPLETE.json').read_bytes()).hexdigest()})]:
            (root/path).write_text(json.dumps(value))
        return root/'receipt.json'

    def test_explicit_four_epoch_target_and_historical_defaults(self):
        for epochs,exposures,step in [(3,292233,93449),(4,389644,117802)]:
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder);receipt=self.setup_run(root,epochs,exposures,step)
                result=check_training_gate(root,step,receipt,exposures,epochs)
                self.assertEqual(result['optimizer_steps'],step)
                if epochs==3:self.assertEqual(check_training_gate(root,step,receipt)['optimizer_steps'],step)
                else:
                    with self.assertRaises(ValueError):check_training_gate(root,step,receipt)

    def test_reject_running_or_wrong_budget_or_changed_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);receipt=self.setup_run(root)
            for args in [(117802,292233,3),(117801,389644,4),(117802,389643,4),(117802,389644,0)]:
                with self.assertRaises(ValueError):check_training_gate(root,args[0],receipt,args[1],args[2])
            (root/'supervisor.json').write_text(json.dumps({'state':'running','returncode':None}))
            with self.assertRaises(ValueError):check_training_gate(root,117802,receipt,389644,4)
            (root/'supervisor.json').write_text(json.dumps({'state':'completed','returncode':0}))
            (root/'immutable/COMPLETE.json').write_text('changed')
            with self.assertRaises(ValueError):check_training_gate(root,117802,receipt,389644,4)

if __name__=='__main__':unittest.main()
