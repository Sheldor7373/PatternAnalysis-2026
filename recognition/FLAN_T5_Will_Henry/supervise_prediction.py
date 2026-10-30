"""Observe one durable prediction job after training; no auto-restart or cutoff."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from gpu_lease import file_lock
from status_io import atomic_json

def now():
    return datetime.now(timezone.utc).isoformat()

def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError,json.JSONDecodeError):
        return None

def check_training_gate(directory,expected_steps,receipt_path,expected_exposures=292233,expected_epochs=3):
    if any(not isinstance(value,int) or isinstance(value,bool) or value<1 for value in (expected_steps,expected_exposures,expected_epochs)):
        raise ValueError('Training gate requires explicit positive finite-target counts')
    root=Path(directory)
    supervisor=read_json(root/'supervisor.json') or {}
    status=read_json(root/'training/status.json') or {}
    metrics=read_json(root/'training/training_metrics.json') or {}
    receipt=read_json(receipt_path) or {}
    if not (supervisor.get('state')=='completed' and supervisor.get('returncode')==0
            and status.get('state')=='completed' and status.get('step')==expected_steps
            and metrics.get('optimizer_steps')==expected_steps
            and receipt.get('optimizer_steps')==expected_steps
            and receipt.get('verified_complete') is True
            and receipt.get('exposures')==expected_exposures and receipt.get('epochs')==expected_epochs):
        raise ValueError('Training has not verifiably finished its finite target; no inference launched')
    completion=Path(receipt['latest_checkpoint'])/'COMPLETE.json'
    if hashlib.sha256(completion.read_bytes()).hexdigest()!=receipt.get('latest_completion_sha256'):
        raise ValueError('Audited final checkpoint identity changed; no inference launched')
    return {'optimizer_steps':metrics['optimizer_steps'],'completion_receipt':receipt}

def gpu_sample():
    try:
        sample=subprocess.run(['nvidia-smi','--id=0','--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw',
                               '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,check=True)
        names=('used_mib','total_mib','utilization_percent','temperature_c','power_w')
        return dict(zip(names,(float(s.strip()) for s in sample.stdout.strip().split(','))))
    except (OSError,ValueError,subprocess.SubprocessError) as error:
        return {'sampling_error':str(error)}

def run(config_path,resume=False):
    config=json.loads(Path(config_path).read_text(encoding='utf-8'))
    gate=check_training_gate(config['training_run'],config['required_training_steps'],config['training_completion_receipt'],
                             config.get('required_training_exposures',292233),config.get('required_training_epochs',3))
    work=Path(config['work_root']).resolve();job=Path(config['supervisor_output']).resolve()
    output=Path(config['prediction_output']).resolve();code=Path(config['code_directory']).resolve()
    assert job.is_relative_to(work) and output.is_relative_to(work)
    assert job!=output and not job.is_relative_to(output) and not output.is_relative_to(job)
    command=list(config['command'])
    assert command and all(isinstance(s,str) for s in command)
    assert '--output' in command and Path(command[command.index('--output')+1]).resolve()==output
    assert '--resume' not in command and '--gpu-lock' not in command
    for name,digest in config['source_files_sha256'].items():
        assert hashlib.sha256((code/name).read_bytes()).hexdigest()==digest,f'Producer source changed: {name}'
    command+=['--gpu-lock',str(work/'evaluation.gpu.lock')]
    if resume: command.append('--resume')
    work.mkdir(exist_ok=True,parents=True);job.mkdir(exist_ok=True,parents=True)
    environment=os.environ.copy()
    environment.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    flags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP if os.name=='nt' else 0
    observation_errors=[];error_count=0
    def attempt(operation,callback):
        nonlocal error_count
        try:
            callback()
            return True
        except (OSError,ValueError,TypeError) as error:
            error_count+=1
            event={'observed_utc':now(),'operation':operation,'error':str(error)}
            if len(observation_errors)<20: observation_errors.append(event)
            # The supervisor's own stderr is separate from the child's output.
            try: print('Observation failed; retrying: '+json.dumps(event),file=sys.stderr,flush=True)
            except OSError: pass
            return False
    with file_lock(work/'evaluation.supervisor.lock','Evaluation supervisor'):
        # Open logs before launching so an initial open failure cannot orphan a job.
        with (job/'stdout.log').open('a',encoding='utf-8') as stdout,(job/'stderr.log').open('a',encoding='utf-8') as stderr, \
             (job/'gpu.jsonl').open('a',encoding='utf-8') as gpu,(job/'health.jsonl').open('a',encoding='utf-8') as health:
            started=time.monotonic();stamp=now()
            child=subprocess.Popen(command,cwd=code,env=environment,stdin=subprocess.DEVNULL,
                                   stdout=stdout,stderr=stderr,close_fds=True,creationflags=flags)
            # The ACTUAL prediction worker, not this observer, owns evaluation.gpu.lock.
            identity={'supervisor_pid':os.getpid(),'child_launcher_pid':child.pid,'started_utc':stamp,
                      'command':command,'prediction_output':str(output),'resume':resume,
                      'training_gate_optimizer_steps':gate['optimizer_steps']}
            attempt('pid_status',lambda:atomic_json(job/'pid.json',identity))
            previous=None;last_progress=time.monotonic();last_health=-float('inf');peak_used=0.;peak_temp=0.
            while True:
                returncode=child.poll()
                sample={'observed_utc':now(),**(gpu_sample() if config.get('sample_gpu',True) else {'sampling_disabled':'CPU test'})}
                attempt('gpu_log',lambda:(gpu.write(json.dumps(sample)+'\n'),gpu.flush()))
                peak_used=max(peak_used,sample.get('used_mib',0.));peak_temp=max(peak_temp,sample.get('temperature_c',0.))
                progress=read_json(output/'generation_status.json') or {}
                committed=progress.get('committed_examples')
                if committed is not None and committed!=previous:
                    previous=committed;last_progress=time.monotonic()
                flags_seen=[]
                if returncode not in (None,0): flags_seen.append('prediction_process_failed')
                if returncode is None and progress.get('state') not in ('generated','completed') and time.monotonic()-last_progress>600:
                    flags_seen.append('no_committed_progress_for_ten_minutes')
                try: free=shutil.disk_usage(work).free/2**30
                except OSError: free=None;flags_seen.append('disk_probe_failed')
                if free is not None and free<10: flags_seen.append('disk_free_below_ten_gib')
                if error_count: flags_seen.append('observation_errors_recorded')
                state='running' if returncode is None else ('completed' if returncode==0 and progress.get('state')=='completed' else 'paused' if returncode==0 else 'failed')
                status={**identity,'observed_utc':now(),'state':state,'returncode':returncode,
                        'session_wall_seconds':time.monotonic()-started,'generation':progress,'gpu':sample,
                        'sampled_peak_used_mib':peak_used,'sampled_peak_temperature_c':peak_temp,'disk_free_gib':free,
                        'flags':flags_seen,'observation_error_count':error_count,'observation_errors':list(observation_errors),
                        'automatic_restart':False,'elapsed_time_cutoff':None,'gpu_lease_owner':'prediction worker'}
                status_saved=attempt('supervisor_status',lambda:atomic_json(job/'supervisor.json',status))
                if returncode is not None:
                    # There will be no next poll after exit: retry terminal state
                    # explicitly instead of leaving a stale durable "running".
                    for delay in (0.05,0.1,0.2,0.4,0.8):
                        if status_saved: break
                        time.sleep(delay)
                        status.update(observation_error_count=error_count,
                                      observation_errors=list(observation_errors))
                        if 'observation_errors_recorded' not in status['flags']:
                            status['flags'].append('observation_errors_recorded')
                        status_saved=attempt('terminal_supervisor_status',lambda:atomic_json(job/'supervisor.json',status))
                if time.monotonic()-last_health>=config.get('health_seconds',600) or returncode is not None:
                    if attempt('health_log',lambda:(health.write(json.dumps(status)+'\n'),health.flush(),os.fsync(health.fileno()))):
                        last_health=time.monotonic()
                if returncode is not None:
                    if not status_saved:
                        raise RuntimeError(f'Prediction exited with code{returncode}, but terminal monitoring status could not be saved; outputs and final health evidence retained')
                    return returncode
                time.sleep(config.get('poll_seconds',2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch',required=True);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();raise SystemExit(run(args.launch,args.resume))
