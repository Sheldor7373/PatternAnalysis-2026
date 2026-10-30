"""Frozen joint encoder resource pilot and resumable content-addressed vectors."""
import argparse
import json
import os
from pathlib import Path
import time
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from transformers import AutoTokenizer,T5EncoderModel
from confidence_core import load_assembly,encode_pair,collate,pooled_states,atomic_json,sha,JointCritic,ConfidenceMLP
from inference_identity import artifact_manifest,effective_runtime
from gpu_lease import gpu_lease
from runtime import precision_context,synchronise,memory_stats

def save_vectors(path, keys, vectors, identity):
    temp=path.with_suffix('.tmp')
    with temp.open('wb') as f:
        np.savez(f,keys=np.array(keys),vectors=np.asarray(vectors,dtype=np.float32),identity=np.array(identity))
        f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

def main(args):
    torch.set_num_threads(args.cpu_threads);torch.set_num_interop_threads(1)
    device=torch.device(args.device);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    rows,assembly=load_assembly(args.data,args.view)
    atomic_json(out/'assembly.json',assembly)
    model=Path(args.encoder).resolve()
    files=artifact_manifest(model)
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    recipe={'model_files_sha256':files,'joint_input':'source report: {source} candidate rewrite: {candidate}',
            'pooling':'concat contextual mean(source), mean(candidate); prefix/EOS/padding excluded',
            'source_cap':args.source_cap,'candidate_cap':args.candidate_cap,
            'runtime':effective_runtime(args.precision,device),'schema':1}
    import hashlib
    identity=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()
    if (out/'recipe.json').exists() and json.loads((out/'recipe.json').read_text())!=recipe:raise ValueError('Cached representation recipe changed')
    atomic_json(out/'recipe.json',recipe)
    flat=[r for part in rows.values() for r in part]
    encoded={};lengths=[];start=time.perf_counter()
    for row in flat:
        x=encode_pair(tokenizer,row['source'],row['candidate'],args.source_cap,args.candidate_cap)
        encoded[row['pair_sha256']]=x
        lengths.append([x['source_tokens'],x['candidate_tokens'],len(x['input_ids']),x['source_truncated'],x['candidate_truncated']])
    lengths=np.array(lengths)
    audit={'rows':len(flat),'quantiles':{name:dict(zip(['min','p50','p90','p95','p99','max'],np.quantile(lengths[:,i],[0,.5,.9,.95,.99,1]).tolist())) for i,name in enumerate(['source','candidate','combined'])},
           'source_truncated':int(lengths[:,3].sum()),'candidate_truncated':int(lengths[:,4].sum()),'tokenization_seconds':time.perf_counter()-start,
           'action':'Truncation is recorded; these pairs remain explicitly limited evidence, never silently treated as complete.'}
    atomic_json(out/'length-audit.json',audit);print(json.dumps({'length_audit':audit}),flush=True)
    encoder=T5EncoderModel.from_pretrained(model,local_files_only=True).to(device).eval().requires_grad_(False)
    if artifact_manifest(model)!=files:raise ValueError('Encoder changed while loading')
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    # Pilot balances ordinary and longest examples, never using labels for selection.
    order=sorted(range(len(flat)),key=lambda i:lengths[i,2])
    chosen=sorted(set(np.linspace(0,len(flat)-1,min(96,len(flat))).astype(int).tolist()+order[-32:]))
    pilot=[flat[i] for i in chosen]
    measurements=[]
    for batch_size in (8,16,32):
        start=time.perf_counter();count=0
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
        try:
            with torch.inference_mode(),precision_context(device,args.precision):
                for offset in range(0,len(pilot),batch_size):
                    batch=collate([encoded[r['pair_sha256']] for r in pilot[offset:offset+batch_size]],tokenizer.pad_token_id,device)
                    vectors=pooled_states(encoder,batch)
                    if not torch.isfinite(vectors).all():raise ValueError('Nonfinite encoder features')
                    count+=len(vectors)
            synchronise(device);elapsed=time.perf_counter()-start
            measurements.append({'batch_size':batch_size,'rows':count,'seconds':elapsed,'rows_per_second':count/elapsed,**memory_stats(device)})
        except torch.OutOfMemoryError:
            measurements.append({'batch_size':batch_size,'oom':True});torch.cuda.empty_cache();break
    safe=[x for x in measurements if not x.get('oom') and (x['peak_reserved_mib'] is None or x['peak_reserved_mib']<3300)]
    if not safe:raise RuntimeError('No pilot batch with memory headroom')
    selected=max(safe,key=lambda x:x['rows_per_second']);batch_size=selected['batch_size']
    pilot_report={'measurements':measurements,'selected':selected,'estimated_vector_seconds':len(flat)/selected['rows_per_second'],
                  'stage_b':'JointCritic path prepared; no end-to-end fitting started'}
    if args.stage_b_pilot:
        # Resource-only backward pass on a disposable head, never a saved fitted model.
        encoder.requires_grad_(True).train();head=ConfidenceMLP(encoder.config.d_model*2).to(device)
        critic=JointCritic(encoder,head,torch.zeros(encoder.config.d_model*2,device=device),torch.ones(encoder.config.d_model*2,device=device))
        measurements_b=[]
        for bs in (2,4,8):
            if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter()
            try:
                batch=collate([encoded[r['pair_sha256']] for r in sorted(pilot,key=lambda r:len(encoded[r['pair_sha256']]['input_ids']),reverse=True)[:bs]],tokenizer.pad_token_id,device)
                with precision_context(device,args.precision):loss=critic(batch).square().mean()
                loss.backward();synchronise(device)
                measurements_b.append({'batch_size':bs,'forward_backward_seconds':time.perf_counter()-start,**memory_stats(device)})
            except torch.OutOfMemoryError:
                measurements_b.append({'batch_size':bs,'oom':True});break
            finally:critic.zero_grad(set_to_none=True)
        pilot_report['stage_b_resource_only']=measurements_b
        pilot_report['stage_b_optimizer_state_estimate_mib']=sum(p.numel() for p in encoder.parameters())*8/2**20
        del critic,head;encoder.eval().requires_grad_(False)
        if device.type=='cuda':torch.cuda.empty_cache()
    atomic_json(out/'pilot.json',pilot_report);print(json.dumps({'pilot':pilot_report}),flush=True)
    if args.pilot_only:return
    chunks=out/'chunks';chunks.mkdir(exist_ok=True);completed=set()
    for path in sorted(chunks.glob('*.npz')):
        with np.load(path,allow_pickle=False) as chunk:
            if str(chunk['identity'])!=identity:raise ValueError('Chunk encoder identity mismatch')
            if not np.isfinite(chunk['vectors']).all() or len(chunk['keys'])!=len(chunk['vectors']):raise ValueError('Invalid cached vectors')
            keys=chunk['keys'].tolist()
            if completed.intersection(keys):raise ValueError('Duplicated cached keys')
            completed.update(keys)
    pending=sorted([r for r in flat if r['pair_sha256'] not in completed],key=lambda r:len(encoded[r['pair_sha256']]['input_ids']))
    started=time.perf_counter();new=0
    for first in range(0,len(pending),256):
        part=pending[first:first+256];vectors=[]
        with torch.inference_mode(),precision_context(device,args.precision):
            for offset in range(0,len(part),batch_size):
                batch=collate([encoded[r['pair_sha256']] for r in part[offset:offset+batch_size]],tokenizer.pad_token_id,device)
                value=pooled_states(encoder,batch)
                if not torch.isfinite(value).all():raise ValueError('Nonfinite cached vector')
                vectors.append(value.cpu().numpy())
        keys=[r['pair_sha256'] for r in part]
        name=__import__('hashlib').sha256(''.join(keys).encode()).hexdigest()+'.npz'
        save_vectors(chunks/name,keys,np.concatenate(vectors),identity);completed.update(keys);new+=len(part)
        atomic_json(out/'status.json',{'phase':'extracting','durable_pairs':len(completed),'required_pairs':len(flat),'new_pairs':new,'seconds':time.perf_counter()-started,'rows_per_second':new/(time.perf_counter()-started),**memory_stats(device)})
    atomic_json(out/'status.json',{'phase':'complete','durable_pairs':len(completed),'required_pairs':len(flat),'new_pairs':new,'seconds':time.perf_counter()-started,**memory_stats(device)})
    print(json.dumps({'status':'complete','durable_pairs':len(completed)}),flush=True)

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--encoder',required=True);p.add_argument('--output',required=True)
    p.add_argument('--view',default='conservative',choices=['conservative','all_weak'])
    p.add_argument('--source-cap',type=int,default=1024);p.add_argument('--candidate-cap',type=int,default=1024)
    p.add_argument('--device',default='cuda');p.add_argument('--precision',default='bf16',choices=['bf16','fp32'])
    p.add_argument('--cpu-threads',type=int,default=2);p.add_argument('--gpu-lease',required=True)
    p.add_argument('--pilot-only',action='store_true');p.add_argument('--stage-b-pilot',action='store_true')
    return p

if __name__=='__main__':
    args=parser().parse_args()
    with gpu_lease(args.gpu_lease):main(args)
