"""Stage B: continue the saved MLP and exact epoch-3 encoder with separate rates.

The generator decoder is never loaded. Baseline and frozen controls stay intact.
Resumable latest checkpoints are independent from validation-selected best.
"""
import argparse
import json
import os
from pathlib import Path
import random
import time
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from transformers import AutoTokenizer,T5EncoderModel
from confidence_core import load_assembly,encode_pair,collate,JointCritic,ConfidenceMLP,atomic_json,atomic_replace,metrics,sha
from inference_identity import artifact_manifest
from runtime import precision_context,memory_stats
from gpu_lease import gpu_lease

def save_checkpoint(path,state):
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
    atomic_replace(temporary,path)

def main(args):
    torch.set_num_threads(args.cpu_threads)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    device=torch.device(args.device);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    rows,assembly=load_assembly(args.data,args.view)
    if assembly['input_groups']!=10000:raise ValueError('Wait for the complete 10,000-source TRAIN assembly')
    stage=Path(args.stage_a);stage_run=json.loads((stage/'run.json').read_text());stage_results=json.loads((stage/'results.json').read_text())
    if 'mlp' not in stage_results:raise ValueError('Stage A MLP must finish before adaptation')
    if assembly!=stage_run['assembly']:raise ValueError('Stage A and B must use the same assembly and view')
    recipe=stage_run['cache']['recipe'];encoder_root=Path(args.encoder).resolve()
    if artifact_manifest(encoder_root)!=recipe['model_files_sha256']:raise ValueError('Not the exact Stage A encoder')
    head_state=torch.load(stage/'mlp.pt',map_location='cpu',weights_only=False)
    tokenizer=AutoTokenizer.from_pretrained(encoder_root,local_files_only=True)
    encoder=T5EncoderModel.from_pretrained(encoder_root,local_files_only=True)
    if artifact_manifest(encoder_root)!=recipe['model_files_sha256']:raise ValueError('Encoder changed while loading')
    head=ConfidenceMLP(head_state['dimensions'],head_state['width'],head_state['dropout']);head.load_state_dict(head_state['head'])
    critic=JointCritic(encoder,head,head_state['mean'],head_state['scale']).to(device)
    optimizer=torch.optim.AdamW([{'params':critic.encoder.parameters(),'lr':args.encoder_lr},
                                  {'params':critic.head.parameters(),'lr':args.head_lr}],weight_decay=.01)
    encoded={p:[encode_pair(tokenizer,r['source'],r['candidate'],recipe['source_cap'],recipe['candidate_cap']) for r in group] for p,group in rows.items()}
    generator=torch.Generator().manual_seed(args.seed);start_epoch=0;cursor=0;order=None;best=float('inf');stale=0;step=0;history=[]
    continuation=getattr(args,'continue_from',None)
    identity={'assembly':assembly,'stage_a_head_sha256':sha(stage/'mlp.pt'),'encoder_recipe_sha256':stage_run['cache']['recipe_sha256'],
              'config':{k:v for k,v in vars(args).items() if k not in {'data','stage_a','encoder','output','gpu_lease','resume','continue_from'}}}
    if continuation:identity['continuation_source_sha256']=sha(continuation)
    if (out/'run.json').exists():
        if json.loads((out/'run.json').read_text())!=identity:raise ValueError('Changed adaptation identity')
        if not args.resume:raise ValueError('Existing run requires --resume')
        state=torch.load(out/'latest.pt',map_location=device,weights_only=False)
        critic.load_state_dict(state['critic']);optimizer.load_state_dict(state['optimizer'])
        start_epoch=state['epoch'];cursor=state['cursor'];order=state['order'];best=state['best'];stale=state['stale'];step=state['step'];history=state['history']
        torch.set_rng_state(state['torch_rng'].cpu());generator.set_state(state['order_rng'].cpu());random.setstate(state['python_rng'])
        if device.type=='cuda':torch.cuda.set_rng_state_all([r.cpu() for r in state['cuda_rng']])
    else:
        if continuation:
            parent_identity=json.loads(Path(continuation).with_name('run.json').read_text())
            if any(parent_identity[k]!=identity[k] for k in ('assembly','stage_a_head_sha256','encoder_recipe_sha256')):
                raise ValueError('Continuation comes from a different assembly, frozen head or encoder')
            state=torch.load(continuation,map_location=device,weights_only=False)
            if sha(continuation)!=identity['continuation_source_sha256']:raise ValueError('Continuation source changed while loading')
            if state['cursor']!=0 or state['order'] is not None:raise ValueError('Late-rate continuation requires a completed epoch boundary')
            if not 0<state['epoch']<args.epochs:raise ValueError('Continuation must increase the finite total epoch target')
            if abs(state['history'][-1]['validation']['log_loss']-state['best'])>1e-12:
                raise ValueError('Do not extend a worse terminal model; assess the validation curve first')
            critic.load_state_dict(state['critic']);optimizer.load_state_dict(state['optimizer'])
            old_rates=[g['lr'] for g in optimizer.param_groups]
            if not (args.encoder_lr<old_rates[0] and args.head_lr<old_rates[1]):raise ValueError('Late-stage learning rates must decrease')
            for group,rate in zip(optimizer.param_groups,[args.encoder_lr,args.head_lr]):group['lr']=rate
            start_epoch=state['epoch'];step=state['step'];best=state['best'];stale=0;history=state['history']
            torch.set_rng_state(state['torch_rng'].cpu());generator.set_state(state['order_rng'].cpu());random.setstate(state['python_rng'])
            if device.type=='cuda':torch.cuda.set_rng_state_all([r.cpu() for r in state['cuda_rng']])
            atomic_json(out/'continuation.json',{'parent_sha256':identity['continuation_source_sha256'],'completed_epochs':start_epoch,
                       'parent_step':step,'new_total_epochs':args.epochs,'old_rates':old_rates,'new_rates':[args.encoder_lr,args.head_lr],
                       'preserved':'encoder/head, Adam moments, Torch/CUDA/Python/order RNG; original run unchanged'})
            save_checkpoint(out/'best.pt',{'critic':critic.state_dict(),'epoch':start_epoch,'step':step,'validation':history[-1]['validation'],'identity':identity})
        atomic_json(out/'run.json',identity)
    def save(epoch,position,indices):
        state={'critic':critic.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'cursor':position,'order':indices,
               'best':best,'stale':stale,'step':step,'history':history,'torch_rng':torch.get_rng_state(),
               'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],'order_rng':generator.get_state(),'python_rng':random.getstate()}
        save_checkpoint(out/'latest.pt',state)
    start=time.perf_counter();last_save=start;run_updates=0
    if not (out/'latest.pt').exists():save(start_epoch,0,None)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch,args.epochs):
        if order is None:order=torch.randperm(len(rows['fit']),generator=generator).tolist();cursor=0
        critic.train()
        while cursor<len(order):
            # A full effective-batch window is one recoverable optimiser update.
            chosen=order[cursor:cursor+args.batch_size*args.accumulate]
            optimizer.zero_grad(set_to_none=True);window_loss=0
            for offset in range(0,len(chosen),args.batch_size):
                ids=chosen[offset:offset+args.batch_size]
                batch=collate([encoded['fit'][i] for i in ids],tokenizer.pad_token_id,device)
                target=torch.tensor([rows['fit'][i]['label'] for i in ids],device=device,dtype=torch.float32)
                with precision_context(device,args.precision):loss=torch.nn.functional.binary_cross_entropy_with_logits(critic(batch),target,reduction='sum')/len(chosen)
                if not torch.isfinite(loss):raise ValueError('Nonfinite adaptation loss')
                loss.backward();window_loss+=float(loss.detach())
            norm=torch.nn.utils.clip_grad_norm_(critic.parameters(),1.)
            if not torch.isfinite(norm):raise ValueError('Nonfinite adaptation gradient')
            optimizer.step();cursor+=len(chosen);step+=1;run_updates+=1
            atomic_json(out/'status.json',{'phase':'training','epoch':epoch+1,'cursor':cursor,'epoch_rows':len(order),'step':step,
                                          'loss':window_loss,'seconds':time.perf_counter()-start,**memory_stats(device)})
            if time.perf_counter()-last_save>=300:save(epoch,cursor,order);last_save=time.perf_counter()
            if (out/'STOP').exists():
                save(epoch,cursor,order)
                atomic_json(out/'status.json',{'phase':'stopped','step':step,'epoch':epoch+1,'cursor':cursor,'reason':'STOP file; remove before --resume'})
                return
        critic.eval();predictions=[]
        with torch.inference_mode(),precision_context(device,args.precision):
            for first in range(0,len(rows['validation']),args.batch_size):
                batch=collate(encoded['validation'][first:first+args.batch_size],tokenizer.pad_token_id,device)
                predictions.extend(critic(batch).float().cpu().tolist())
        score=metrics([r['label'] for r in rows['validation']],predictions)
        history.append({'epoch':epoch+1,'validation':score,'seconds':time.perf_counter()-start})
        if score['log_loss']<best-1e-5:
            best=score['log_loss'];stale=0
            save_checkpoint(out/'best.pt',{'critic':critic.state_dict(),'epoch':epoch+1,'step':step,'validation':score,'identity':identity})
        else:stale+=1
        atomic_json(out/'history.json',history);save(epoch+1,0,None);order=None;cursor=0
        print(json.dumps({'epoch':epoch+1,'step':step,'validation':score,'stale':stale}),flush=True)
        if stale>=args.patience:break
    atomic_json(out/'status.json',{'phase':'scoring','step':step,'epochs_run':len(history),'best_validation_log_loss':best,'session_seconds':time.perf_counter()-start,**memory_stats(device)})
    # Evaluate the selected checkpoint on reserved TRAIN partitions, never TEST.
    selected=torch.load(out/'best.pt',map_location=device,weights_only=False);critic.load_state_dict(selected['critic']);critic.eval()
    predictions={}
    with torch.inference_mode(),precision_context(device,args.precision):
        for part,records in rows.items():
            chunks=[]
            for first in range(0,len(records),args.batch_size):
                batch=collate(encoded[part][first:first+args.batch_size],tokenizer.pad_token_id,device)
                chunks.append(critic(batch).float().cpu().numpy())
            predictions[part]=np.concatenate(chunks)
    from train_confidence import temperature
    t=temperature([r['label'] for r in rows['calibration']],predictions['calibration'])
    results={'selected_epoch':selected['epoch'],'temperature':t,
             'uncalibrated':{p:metrics([r['label'] for r in rows[p]],z) for p,z in predictions.items()},
             'calibrated':{p:metrics([r['label'] for r in rows[p]],z/t) for p,z in predictions.items()},
             'warning':'Synthetic weak-label diagnostics; calibration scores are fitted, not fresh TEST evaluation'}
    with (out/'predictions.npz').open('wb') as f:
        np.savez(f,**predictions,**{p+'_pair_sha256':np.array([r['pair_sha256'] for r in rows[p]]) for p in rows})
    atomic_json(out/'results.json',results)
    atomic_json(out/'status.json',{'phase':'complete','step':step,'epochs_run':len(history),'best_validation_log_loss':best,'session_seconds':time.perf_counter()-start,**memory_stats(device)})

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','stage-a','encoder','output','gpu-lease'):p.add_argument('--'+name,required=True)
    p.add_argument('--view',default='conservative',choices=['conservative','all_weak'])
    p.add_argument('--epochs',type=int,default=3);p.add_argument('--patience',type=int,default=2)
    p.add_argument('--encoder-lr',type=float,default=1e-5);p.add_argument('--head-lr',type=float,default=1e-4)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--accumulate',type=int,default=4)
    p.add_argument('--device',default='cuda');p.add_argument('--precision',default='bf16',choices=['bf16','fp32'])
    p.add_argument('--cpu-threads',type=int,default=2);p.add_argument('--seed',type=int,default=3751);p.add_argument('--resume',action='store_true')
    p.add_argument('--continue-from',help='Immutable latest checkpoint at an improved completed epoch boundary; starts a separate lower-rate run')
    return p

if __name__=='__main__':
    args=parser().parse_args()
    with gpu_lease(args.gpu_lease):main(args)
