"""Stage A: two boosted-tree heads and an MLP on identical frozen representations.

Final fitting refuses incomplete TRAIN pools. No TEST files are opened. Labelled
calibration is separate from validation selection; labels are synthetic and weak.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import time
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from confidence_core import load_assembly,ConfidenceMLP,atomic_json,atomic_replace,metrics,sha
from gpu_lease import gpu_lease

def load_vectors(cache, rows):
    cache=Path(cache);recipe=json.loads((cache/'recipe.json').read_text())
    import hashlib
    identity=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()
    vectors={};chunks={}
    for path in sorted((cache/'chunks').glob('*.npz')):
        chunks[path.name]=sha(path)
        with np.load(path,allow_pickle=False) as obj:
            if str(obj['identity'])!=identity:raise ValueError('Cached encoder recipe mismatch')
            if len(obj['keys'])!=len(obj['vectors']) or not np.isfinite(obj['vectors']).all():raise ValueError('Invalid vectors')
            for key,value in zip(obj['keys'].tolist(),obj['vectors']):
                if key in vectors:raise ValueError('Duplicate cached pair')
                vectors[key]=value
    result={}
    for part,records in rows.items():
        if any(r['pair_sha256'] not in vectors for r in records):raise ValueError('Required pair embeddings not yet complete')
        result[part]=(np.stack([vectors[r['pair_sha256']] for r in records]),np.array([r['label'] for r in records],dtype=np.int64))
    return result,{'recipe':recipe,'recipe_sha256':identity,'chunks_sha256':chunks}

def train_mlp(data,out,args):
    x,y=data['fit'];xv,yv=data['validation']
    mean=x.mean(0);scale=np.maximum(x.std(0),1e-6)
    x=(x-mean)/scale;xv=(xv-mean)/scale
    device=torch.device(args.device);head=ConfidenceMLP(x.shape[1],args.width,args.dropout).to(device)
    optimizer=torch.optim.AdamW(head.parameters(),lr=args.lr,weight_decay=.01)
    generator=torch.Generator().manual_seed(args.seed);best=float('inf');best_state=None;stale=0;history=[];start=time.perf_counter();first_epoch=0
    def save_latest(epoch):
        state={'head':head.state_dict(),'optimizer':optimizer.state_dict(),'best':best,'best_head':best_state,'stale':stale,'history':history,
               'next_epoch':epoch,'order_rng':generator.get_state(),'torch_rng':torch.get_rng_state(),
               'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else []}
        temporary=out/'mlp-latest.pt.tmp'
        with temporary.open('wb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
        atomic_replace(temporary,out/'mlp-latest.pt')
    if (out/'mlp-latest.pt').exists():
        if not getattr(args,'resume',False):raise ValueError('MLP recovery requires --resume')
        state=torch.load(out/'mlp-latest.pt',map_location=device,weights_only=False)
        head.load_state_dict(state['head']);optimizer.load_state_dict(state['optimizer'])
        best=state['best'];best_state=state['best_head'];stale=state['stale'];history=state['history'];first_epoch=state['next_epoch']
        generator.set_state(state['order_rng'].cpu());torch.set_rng_state(state['torch_rng'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state_all([r.cpu() for r in state['cuda_rng']])
    else:save_latest(0)
    for epoch in range(first_epoch,args.epochs):
        head.train();total=0
        for indexes in torch.randperm(len(x),generator=generator).split(args.batch_size):
            a=torch.from_numpy(x[indexes.numpy()]).to(device);b=torch.from_numpy(y[indexes.numpy()]).float().to(device)
            optimizer.zero_grad(set_to_none=True);loss=torch.nn.functional.binary_cross_entropy_with_logits(head(a),b)
            if not torch.isfinite(loss):raise ValueError('Nonfinite confidence loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1);optimizer.step();total+=float(loss.detach())*len(indexes)
        head.eval()
        with torch.no_grad():z=torch.cat([head(torch.from_numpy(xv[i:i+args.batch_size]).to(device)).cpu() for i in range(0,len(xv),args.batch_size)]).numpy()
        score=metrics(yv,z);history.append({'epoch':epoch+1,'fit_log_loss':total/len(x),'validation':score,'elapsed_seconds':time.perf_counter()-start})
        atomic_json(out/'mlp-history.json',history)
        if score['log_loss']<best-1e-5:
            best=score['log_loss'];best_state=copy.deepcopy({k:v.cpu() for k,v in head.state_dict().items()});stale=0
            state={'head':best_state,'mean':mean,'scale':scale,'width':args.width,'dropout':args.dropout,'dimensions':x.shape[1],
                   'epoch':epoch+1,'optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),'data_order_rng':generator.get_state(),
                   'selection':'minimum grouped-validation unweighted log loss; calibration not used'}
            temporary=out/'mlp.pt.tmp'
            with temporary.open('wb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
            atomic_replace(temporary,out/'mlp.pt')
        else:stale+=1
        save_latest(epoch+1)
        print(json.dumps({'mlp_epoch':epoch+1,'validation_log_loss':score['log_loss'],'stale':stale}),flush=True)
        if stale>=args.patience:break
    head.load_state_dict(best_state);head.eval();outputs={}
    with torch.no_grad():
        for part,(a,b) in data.items():
            a=(a-mean)/scale
            outputs[part]=torch.cat([head(torch.from_numpy(a[i:i+args.batch_size]).to(device)).cpu() for i in range(0,len(a),args.batch_size)]).numpy()
    return outputs,{'seconds':time.perf_counter()-start,'best_validation_log_loss':best,'epochs_run':len(history)}

def train_trees(data,out,args):
    import lightgbm as lgb
    import xgboost as xgb
    x,y=data['fit'];xv,yv=data['validation'];result={}
    for name in ('lightgbm','xgboost'):
        start=time.perf_counter()
        if name=='lightgbm':
            model=lgb.LGBMClassifier(n_estimators=args.tree_rounds,num_leaves=15,max_depth=-1,learning_rate=.05,min_child_samples=30,
                                    reg_lambda=1.,colsample_bytree=.8,n_jobs=args.cpu_threads,random_state=args.seed,verbosity=-1,deterministic=True,force_col_wise=True)
            model.fit(x,y,eval_set=[(xv,yv)],eval_metric='binary_logloss',callbacks=[lgb.early_stopping(30,verbose=False)])
            model.booster_.save_model(str(out/'lightgbm.txt'));iteration=model.best_iteration_
        else:
            model=xgb.XGBClassifier(n_estimators=args.tree_rounds,max_depth=4,learning_rate=.05,min_child_weight=5,reg_lambda=1.,
                                    colsample_bytree=.8,n_jobs=args.cpu_threads,random_state=args.seed,tree_method='hist',device='cpu',
                                    eval_metric='logloss',early_stopping_rounds=30)
            model.fit(x,y,eval_set=[(xv,yv)],verbose=False);model.save_model(out/'xgboost.json');iteration=model.best_iteration+1
        predictions={part:np.log(np.clip(p:=model.predict_proba(a)[:,1],1e-7,1-1e-7)/np.clip(1-p,1e-7,1)) for part,(a,b) in data.items()}
        result[name]=(predictions,{'seconds':time.perf_counter()-start,'best_iteration':iteration,'parameters':model.get_params()})
        print(json.dumps({'tree':name,'seconds':result[name][1]['seconds'],'best_iteration':iteration}),flush=True)
    return result

def temperature(labels,logits):
    # One scalar fitted only on the reserved calibration partition.
    y=torch.tensor(labels,dtype=torch.float64);z=torch.tensor(logits,dtype=torch.float64)
    log_t=torch.zeros((),dtype=torch.float64,requires_grad=True)
    opt=torch.optim.LBFGS([log_t],lr=.2,max_iter=60,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad();t=log_t.clamp(-3,3).exp();loss=torch.nn.functional.binary_cross_entropy_with_logits(z/t,y);loss.backward();return loss
    opt.step(closure);return float(log_t.detach().clamp(-3,3).exp())

def main(args):
    torch.set_num_threads(args.cpu_threads)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    rows,assembly=load_assembly(args.data,args.view)
    if assembly['input_groups']!=10000:raise ValueError('Final fitting waits for exactly 10,000 assembled TRAIN source components')
    data,cache=load_vectors(args.cache,rows)
    if any(len(set(y))!=2 for x,y in data.values()):raise ValueError('Both labels required in every TRAIN partition')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    config={k:v for k,v in vars(args).items() if k not in {'data','cache','output','gpu_lease','resume'}}
    identity={'config':config,'assembly':assembly,'cache':cache,'selection':'TRAIN validation only; never official TEST'}
    if (out/'run.json').exists():
        if not args.resume or json.loads((out/'run.json').read_text())!=identity:raise ValueError('Existing experiment requires --resume with unchanged identity')
        if (out/'results.json').exists():print('Already complete; saved models and outputs preserved',flush=True);return
    else:atomic_json(out/'run.json',identity)
    runs={}
    if args.heads in ('trees','all'):runs.update(train_trees(data,out,args))
    if args.heads in ('mlp','all'):
        with gpu_lease(args.gpu_lease if args.device=='cuda' else None):runs['mlp']=train_mlp(data,out,args)
    scores={}
    for name,(predictions,resource) in runs.items():
        # Temperature tuning does not affect ranking; do not use calibration scores for model selection.
        t=temperature(data['calibration'][1],predictions['calibration'])
        scores[name]={'resource':resource,'temperature':t,'uncalibrated':{p:metrics(data[p][1],v) for p,v in predictions.items()},
                      'calibrated':{p:metrics(data[p][1],v/t) for p,v in predictions.items()},
                      'warning':'Calibration-set scores are fitted diagnostics, not unbiased final-test performance'}
        with (out/(name+'-predictions.npz')).open('wb') as f:
            np.savez(f,**{p:v for p,v in predictions.items()},**{p+'_pair_sha256':np.array([r['pair_sha256'] for r in rows[p]]) for p in rows})
    atomic_json(out/'results.json',scores)
    print(json.dumps({'complete_heads':list(scores),'validation':{n:s['uncalibrated']['validation'] for n,s in scores.items()}}),flush=True)

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','cache','output','gpu-lease'):p.add_argument('--'+name,required=True)
    p.add_argument('--view',default='conservative',choices=['conservative','all_weak'])
    p.add_argument('--heads',default='all',choices=['all','mlp','trees']);p.add_argument('--device',default='cuda',choices=['cuda','cpu'])
    p.add_argument('--cpu-threads',type=int,default=2);p.add_argument('--seed',type=int,default=3750)
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--patience',type=int,default=5)
    p.add_argument('--width',type=int,default=128);p.add_argument('--dropout',type=float,default=.2)
    p.add_argument('--lr',type=float,default=.001);p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--tree-rounds',type=int,default=400)
    p.add_argument('--resume',action='store_true')
    return p

if __name__=='__main__':main(parser().parse_args())
