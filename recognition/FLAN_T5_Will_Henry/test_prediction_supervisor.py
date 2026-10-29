"""CPU process tests for the observer, worker lease and real batch journal."""
from contextlib import redirect_stderr
import hashlib,io,json,os,shutil,signal,subprocess,sys,tempfile,textwrap,threading,time,unittest
from pathlib import Path
from unittest.mock import patch
import supervise_prediction as supervisor
from gpu_lease import file_lock

class SupervisorChecks(unittest.TestCase):
    def fixture(self,root,hold=0,fail=False):
        training=root/'train';(training/'training').mkdir(parents=True)
        completion=training/'final-checkpoint';completion.mkdir();(completion/'COMPLETE.json').write_text('{"test":true}')
        receipt=training/'completion-audit.json'
        receipt.write_text(json.dumps({'verified_complete':True,'optimizer_steps':93449,'exposures':292233,'epochs':3,
             'latest_checkpoint':str(completion),'latest_completion_sha256':hashlib.sha256((completion/'COMPLETE.json').read_bytes()).hexdigest()}))
        for path,value in ((training/'supervisor.json',{'state':'completed','returncode':0}),
                           (training/'training/status.json',{'state':'completed','step':93449}),
                           (training/'training/training_metrics.json',{'optimizer_steps':93449})):
            path.write_text(json.dumps(value))
        output=root/'work/result';code=root/'code';code.mkdir()
        for name in ('gpu_lease.py','status_io.py','prediction_storage.py'):
            shutil.copy2(Path(__file__).parent/name,code/name)
        script=code/'fake.py'
        script.write_text(textwrap.dedent('''
            import argparse,json,os,pathlib,time
            from gpu_lease import gpu_lease
            from prediction_storage import PredictionRun
            p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--resume',action='store_true')
            p.add_argument('--gpu-lock',required=True);p.add_argument('--hold',type=float,default=0)
            p.add_argument('--fail-after-first',action='store_true');a=p.parse_args()
            with gpu_lease(a.gpu_lock):
                with PredictionRun(a.output,a.resume) as run:
                    rows=[{'id':str(i),'input':'small input','target':'small target'} for i in range(3)]
                    run.configure({'producer':'cpu-test'},{},rows)
                    (pathlib.Path(a.output)/'worker-ready.json').write_text(json.dumps({'pid':os.getpid()}))
                    time.sleep(a.hold)
                    for offset in range(len(run.predictions),len(rows)):
                        run.commit({'start':offset,'rows':[{**rows[offset],'prediction':'small output'}],
                                    'generation_seconds':0.001,'wall_seconds':0.002})
                        if a.fail_after_first and not a.resume: raise SystemExit(3)
                    run.export_predictions();run.progress('completed')
            '''))
        command=[sys.executable,str(script),'--output',str(output),'--hold',str(hold)]
        if fail: command.append('--fail-after-first')
        config={'training_run':str(training),'required_training_steps':93449,'training_completion_receipt':str(receipt),
                'work_root':str(root/'work'),'supervisor_output':str(root/'work/job'),'prediction_output':str(output),
                'code_directory':str(code),'command':command,
                'source_files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in code.glob('*.py')},
                'poll_seconds':0.02,'health_seconds':0.05,'sample_gpu':False}
        path=root/'launch.json';path.write_text(json.dumps(config));return path,config

    def wait_for(self,condition,seconds=10):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            if condition(): return
            time.sleep(0.02)
        self.fail('Controlled CPU process did not reach the expected state')

    def test_active_training_blocks_child_launch(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            path,c=self.fixture(Path(d));(Path(c['training_run'])/'supervisor.json').write_text('{"state":"running"}')
            with patch.object(supervisor.subprocess,'Popen') as launch:
                with self.assertRaisesRegex(ValueError,'Training has not'): supervisor.run(path)
                launch.assert_not_called()

    def test_wrong_training_target_blocks_launch(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            path,c=self.fixture(Path(d));(Path(c['training_run'])/'training/training_metrics.json').write_text('{"optimizer_steps":69096}')
            with self.assertRaises(ValueError): supervisor.run(path)

    def test_incomplete_exposure_audit_blocks_launch(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            path,c=self.fixture(Path(d));audit=Path(c['training_completion_receipt']);value=json.loads(audit.read_text())
            value['exposures']=194822;audit.write_text(json.dumps(value))
            with self.assertRaises(ValueError): supervisor.run(path)

    def test_changed_audited_checkpoint_blocks_launch(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            path,c=self.fixture(Path(d));audit=json.loads(Path(c['training_completion_receipt']).read_text())
            (Path(audit['latest_checkpoint'])/'COMPLETE.json').write_text('changed')
            with self.assertRaisesRegex(ValueError,'checkpoint identity changed'): supervisor.run(path)

    def test_source_change_blocks_launch(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            path,c=self.fixture(Path(d));(Path(c['code_directory'])/'fake.py').write_text('changed')
            with self.assertRaisesRegex(AssertionError,'Producer source changed'): supervisor.run(path)

    def test_completed_child_and_health_are_recorded(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            path,c=self.fixture(Path(d));self.assertEqual(supervisor.run(path),0)
            status=json.loads((Path(c['supervisor_output'])/'supervisor.json').read_text())
            self.assertEqual(status['state'],'completed');self.assertEqual(status['generation']['committed_examples'],3)
            self.assertIsNone(status['elapsed_time_cutoff']);self.assertEqual(status['gpu_lease_owner'],'prediction worker')
            self.assertTrue((Path(c['supervisor_output'])/'health.jsonl').read_text())

    def test_failed_child_real_batches_resume_without_replacing_prefix(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            path,c=self.fixture(Path(d),fail=True);self.assertEqual(supervisor.run(path),3)
            output=Path(c['prediction_output']);batch=output/'batches/00000000.json';saved=batch.read_bytes()
            self.assertFalse((output/'predictions.jsonl').exists())
            self.assertEqual(supervisor.run(path,resume=True),0)
            self.assertEqual(batch.read_bytes(),saved)
            rows=[json.loads(s) for s in (output/'predictions.jsonl').read_text().splitlines()]
            self.assertEqual([r['id'] for r in rows],['0','1','2'])
            self.assertEqual(len(list((output/'batches').glob('*.json'))),3)

    def test_live_worker_lease_blocks_a_second_owner_and_stale_file_is_reusable(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            path=Path(d)/'gpu.lock'
            with file_lock(path):
                with self.assertRaises(RuntimeError):
                    with file_lock(path): pass
            self.assertTrue(path.exists())
            with file_lock(path): pass

    def test_actual_observation_write_failure_recovers_while_child_runs(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d,redirect_stderr(io.StringIO()) as errors:
            path,c=self.fixture(Path(d),hold=3)
            job=Path(c['supervisor_output']);job.mkdir(parents=True)
            blocked=job/'supervisor.json';blocked.mkdir()
            unblock=threading.Timer(1.9,blocked.rmdir);unblock.start()
            try: self.assertEqual(supervisor.run(path),0)
            finally: unblock.join()
            status=json.loads((job/'supervisor.json').read_text())
            self.assertGreaterEqual(status['observation_error_count'],1)
            self.assertEqual(status['state'],'completed');self.assertIn('retrying',errors.getvalue())
            health=[json.loads(s) for s in (job/'health.jsonl').read_text().splitlines()]
            self.assertTrue(any(s['state']=='running' and s['observation_error_count']>0 for s in health))

    def test_abrupt_supervisor_death_keeps_worker_gpu_lease(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            root=Path(d);path,c=self.fixture(root,hold=10)
            with (root/'observer.log').open('w') as log:
                observer=subprocess.Popen([sys.executable,str(Path(supervisor.__file__)),'--launch',str(path)],
                    stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                worker_pid=None
                try:
                    ready=Path(c['prediction_output'])/'worker-ready.json';pidfile=Path(c['supervisor_output'])/'pid.json'
                    self.wait_for(lambda:ready.exists() and pidfile.exists())
                    worker_pid=json.loads(ready.read_text())['pid']
                    supervisor_pid=json.loads(pidfile.read_text())['supervisor_pid']
                    self.assertNotEqual(worker_pid,supervisor_pid)
                    # Both PIDs came from this test's uniquely scoped CPU fixture.
                    os.kill(supervisor_pid,signal.SIGTERM);observer.wait(timeout=5)
                    with self.assertRaises(RuntimeError):
                        with file_lock(Path(c['work_root'])/'evaluation.gpu.lock'): pass
                    replacement={**c,'prediction_output':str(root/'work/other'),'supervisor_output':str(root/'work/other-job')}
                    replacement['command']=list(c['command']);position=replacement['command'].index('--output')+1
                    replacement['command'][position]=replacement['prediction_output']
                    newpath=root/'replacement.json';newpath.write_text(json.dumps(replacement))
                    self.assertNotEqual(supervisor.run(newpath),0)
                    self.assertFalse((Path(replacement['prediction_output'])/'worker-ready.json').exists())
                finally:
                    if worker_pid is not None:
                        try: os.kill(worker_pid,signal.SIGTERM)
                        except ProcessLookupError: pass
                    if observer.poll() is None:
                        observer.terminate();observer.wait(timeout=5)
                # OS ownership, not lock-file deletion, controls stale-lease recovery.
                self.wait_for(lambda:self.can_lock(Path(c['work_root'])/'evaluation.gpu.lock'))

    def test_transient_terminal_status_failure_is_retried(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d,redirect_stderr(io.StringIO()):
            path,c=self.fixture(Path(d));real=supervisor.atomic_json;failed=[]
            def write(destination,value):
                if Path(destination).name=='supervisor.json' and value['state']=='completed' and not failed:
                    failed.append(True);raise OSError('transient terminal rename failure')
                return real(destination,value)
            with patch.object(supervisor,'atomic_json',side_effect=write):
                self.assertEqual(supervisor.run(path),0)
            status=json.loads((Path(c['supervisor_output'])/'supervisor.json').read_text())
            self.assertEqual(status['state'],'completed');self.assertGreaterEqual(status['observation_error_count'],1)

    def test_persistent_terminal_status_failure_cannot_report_clean_success(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d,redirect_stderr(io.StringIO()):
            path,c=self.fixture(Path(d));real=supervisor.atomic_json
            def write(destination,value):
                if Path(destination).name=='supervisor.json' and value['state']=='completed':
                    raise OSError('persistent terminal rename failure')
                return real(destination,value)
            with patch.object(supervisor,'atomic_json',side_effect=write):
                with self.assertRaisesRegex(RuntimeError,'terminal monitoring status'):
                    supervisor.run(path)
            health=[json.loads(s) for s in (Path(c['supervisor_output'])/'health.jsonl').read_text().splitlines()]
            self.assertEqual(health[-1]['state'],'completed')
            self.assertTrue((Path(c['prediction_output'])/'predictions.jsonl').exists())

    def can_lock(self,path):
        try:
            with file_lock(path): return True
        except RuntimeError: return False

if __name__=='__main__': unittest.main()
