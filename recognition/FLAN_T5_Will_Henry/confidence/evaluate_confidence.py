"""One locked, resumable final stress-test pass; never fits or tunes a model."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
import time
import numpy as np
import torch
from transformers import AutoTokenizer,T5EncoderModel
from confidence_core import encode_pair,collate,pooled_states,ConfidenceMLP,JointCritic,atomic_json,atomic_replace,sha
from confidence_evaluation import load_test,score_metrics,report_weights,bootstrap,diagnostics,THRESHOLDS
from inference_identity import artifact_manifest,effective_runtime
from gpu_lease import gpu_lease
from runtime import precision_context,memory_stats,synchronise

def batches(rows,encoded,maximum=16,area=3534400):
    result=[];current=[];longest=0
    for row in rows:
        n=len(encoded[row['pair_sha256']]['input_ids']);next_max=max(n,longest)
        if current and (len(current)>=maximum or (len(current)+1)*next_max**2>area):
            result.append(current);current=[];longest=0
        current.append(row);longest=max(longest,n)
    if current:result.append(current)
    return result

def main(args):
    torch.set_num_threads(2);torch.set_num_interop_threads(1);device=torch.device('cuda')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if sha(Path(args.test)/'manifest.json')!=args.test_sha256:raise ValueError('Wrong locked TEST manifest')
    stage=Path(args.experiment);adapt=Path(args.adapted_root) if args.adapted_root else stage/'adapted';frozen=stage/'frozen-mlp';trees=stage/'frozen-trees'
    if not all((p/'results.json').exists() for p in (adapt,frozen,trees)):raise ValueError('All predeclared models must finish before TEST')
    stage_a=json.loads((frozen/'run.json').read_text());recipe=stage_a['cache']['recipe']
    if artifact_manifest(args.encoder)!=recipe['model_files_sha256']:raise ValueError('Frozen encoder identity mismatch')
    temperatures={'mlp':json.loads((frozen/'results.json').read_text())['mlp']['temperature'],
                  'adapted':json.loads((adapt/'results.json').read_text())['temperature']}
    tree_results=json.loads((trees/'results.json').read_text());temperatures.update({n:tree_results[n]['temperature'] for n in ('lightgbm','xgboost')})
    files={'frozen_mlp':frozen/'mlp.pt','adapted':adapt/'best.pt','lightgbm':trees/'lightgbm.txt','xgboost':trees/'xgboost.json'}
    selection=json.loads(Path(args.selection_plan).read_text())
    if not selection.get('frozen') or not selection.get('extension_decision_complete'):
        raise ValueError('Complete the TRAIN-validation extension decision and freeze selection before TEST')
    if selection['test_manifest_sha256']!=args.test_sha256 or selection['models_sha256']!={k:sha(p) for k,p in files.items()}:
        raise ValueError('Selected model or TEST identity differs from the frozen decision')
    if selection['calibration_temperatures']!=temperatures:raise ValueError('Temperature differs from frozen TRAIN calibration')
    identity={'test_manifest_sha256':args.test_sha256,'models_sha256':{k:sha(p) for k,p in files.items()},
              'frozen_encoder_files_sha256':recipe['model_files_sha256'],'runtime':effective_runtime('bf16',device),
              'representation_recipe_sha256':stage_a['cache']['recipe_sha256'],'source_cap':recipe['source_cap'],'candidate_cap':recipe['candidate_cap'],
              'calibration_temperatures':temperatures,'maximum_microbatch':16,'attention_area_limit':3534400,
              'head_precision':{'frozen_mlp':'fp32 on float32 pooled BF16 encoder vectors','adapted':'bf16 autocast, matching its calibration'},
              'selection_plan_sha256':sha(args.selection_plan),
              'producer_sha256':{n:sha(Path(__file__).with_name(n)) for n in ('evaluate_confidence.py','confidence_core.py','confidence_evaluation.py')},
              'selection':'Checkpoint/hyperparameters from TRAIN validation; temperatures from TRAIN calibration; none fitted on TEST',
              'thresholds':list(THRESHOLDS),'positive_class':'synthetic material meaning error or important omission',
              'created_utc':args.locked_utc}
    if (out/'recipe.json').exists() and json.loads((out/'recipe.json').read_text())!=identity:raise ValueError('Changed final-evaluation identity')
    atomic_json(out/'recipe.json',identity)
    rows,receipt=load_test(args.test);atomic_json(out/'test-assembly.json',receipt)
    # Assert TEST groups are disjoint from all critic fitting/calibration components.
    from confidence_core import load_assembly
    training,_=load_assembly(args.train)
    training_groups={r['group_sha256'] for p in training.values() for r in p}
    if training_groups & {r['group_sha256'] for r in rows}:raise ValueError('TRAIN/TEST source-group overlap')
    del training
    tokenizer=AutoTokenizer.from_pretrained(args.encoder,local_files_only=True)
    encoded={r['pair_sha256']:encode_pair(tokenizer,r['source'],r['candidate'],recipe['source_cap'],recipe['candidate_cap']) for r in rows}
    atomic_json(out/'length-audit.json',{'all_pairs':len(rows),'source_truncated':sum(x['source_truncated'] for x in encoded.values()),
                'candidate_truncated':sum(x['candidate_truncated'] for x in encoded.values()),
                'maximum_joint_tokens':max(len(x['input_ids']) for x in encoded.values()),'policy':'Predeclared caps unchanged; length-dependent batching affects memory only'})
    encoder=T5EncoderModel.from_pretrained(args.encoder,local_files_only=True).to(device).eval().requires_grad_(False)
    selected=torch.load(files['frozen_mlp'],map_location='cpu',weights_only=False)
    head=ConfidenceMLP(selected['dimensions'],selected['width'],selected['dropout']);head.load_state_dict(selected['head'])
    frozen_critic=JointCritic(encoder,head,selected['mean'],selected['scale']).to(device).eval().requires_grad_(False)
    adapted_encoder=T5EncoderModel.from_pretrained(args.encoder,local_files_only=True)
    adapted_head=ConfidenceMLP(selected['dimensions'],selected['width'],selected['dropout'])
    adapted_critic=JointCritic(adapted_encoder,adapted_head,selected['mean'],selected['scale']).to(device)
    adapted_state=torch.load(files['adapted'],map_location=device,weights_only=False);adapted_critic.load_state_dict(adapted_state['critic']);adapted_critic.eval().requires_grad_(False)
    del selected,adapted_state
    import lightgbm as lgb
    import xgboost as xgb
    light=lgb.Booster(model_file=str(files['lightgbm']));xg=xgb.XGBClassifier(n_jobs=1);xg.load_model(files['xgboost']);xg.set_params(n_jobs=1,device='cpu')
    chunks=out/'chunks';chunks.mkdir(exist_ok=True);key_to_scores={};recipe_hash=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    for path in sorted(chunks.glob('*.npz')):
        with np.load(path,allow_pickle=False) as obj:
            if str(obj['identity'])!=recipe_hash:raise ValueError('Changed score-chunk identity')
            if not np.isfinite(obj['scores']).all():raise ValueError('Nonfinite saved TEST score')
            for key,values in zip(obj['keys'].tolist(),obj['scores']):
                if key in key_to_scores:raise ValueError('Duplicated score chunk')
                key_to_scores[key]=values
    pending=sorted([r for r in rows if r['pair_sha256'] not in key_to_scores],key=lambda r:len(encoded[r['pair_sha256']]['input_ids']))
    start=time.perf_counter();timing={'frozen_encoder_and_mlp_seconds':0.,'adapted_encoder_and_mlp_seconds':0.,'cpu_tree_seconds':0.};new=0
    torch.cuda.reset_peak_memory_stats()
    for first in range(0,len(pending),256):
        part=pending[first:first+256];scores=[];keys=[]
        for group in batches(part,encoded):
            batch=collate([encoded[r['pair_sha256']] for r in group],tokenizer.pad_token_id,device)
            synchronise(device);t=time.perf_counter()
            with torch.inference_mode(),precision_context(device,'bf16'):
                vectors=pooled_states(frozen_critic.encoder,batch)
            # Frozen head was fitted/calibrated in FP32; do not silently recast it.
            with torch.inference_mode():mlp=frozen_critic.head((vectors-frozen_critic.mean)/frozen_critic.scale)
            synchronise(device);timing['frozen_encoder_and_mlp_seconds']+=time.perf_counter()-t;t=time.perf_counter()
            with torch.inference_mode(),precision_context(device,'bf16'):joint=adapted_critic(batch)
            synchronise(device);timing['adapted_encoder_and_mlp_seconds']+=time.perf_counter()-t;t=time.perf_counter()
            values=vectors.cpu().numpy();lp=light.predict(values,num_threads=1);xp=xg.predict_proba(values)[:,1]
            # Match the frozen-stage calibration transformation exactly.
            lz=np.log(np.clip(lp,1e-7,1-1e-7)/np.clip(1-lp,1e-7,1));xz=np.log(np.clip(xp,1e-7,1-1e-7)/np.clip(1-xp,1e-7,1))
            scores.append(np.stack([mlp.float().cpu().numpy(),joint.float().cpu().numpy(),lz,xz],axis=-1))
            timing['cpu_tree_seconds']+=time.perf_counter()-t;keys.extend(r['pair_sha256'] for r in group)
        value=np.concatenate(scores)
        if not np.isfinite(value).all():raise ValueError('Nonfinite TEST score')
        destination=chunks/(hashlib.sha256(''.join(keys).encode()).hexdigest()+'.npz');temporary=destination.with_suffix('.tmp')
        with temporary.open('wb') as f:np.savez(f,keys=np.array(keys),scores=value,identity=np.array(recipe_hash));f.flush();os.fsync(f.fileno())
        atomic_replace(temporary,destination)
        key_to_scores.update(zip(keys,value));new+=len(keys)
        atomic_json(out/'status.json',{'phase':'predicting','durable_pairs':len(key_to_scores),'required_pairs':len(rows),'new_pairs':new,'session_seconds':time.perf_counter()-start,**memory_stats(device)})
    if set(key_to_scores)!={r['pair_sha256'] for r in rows}:raise ValueError('Final TEST ID audit failed')
    atomic_json(out/'prediction-resources.json',{'timing':timing,'new_pairs_this_session':new,'seconds':time.perf_counter()-start,
                'memory_scope':'Both encoders resident in one evaluation process; not separate per-model VRAM',**memory_stats(device)})
    # Free GPU before the CPU-only bootstrap and plotting stage.
    del encoder,head,frozen_critic,adapted_critic,adapted_encoder,adapted_head;torch.cuda.empty_cache()
    usable=[r for r in rows if r['score']];y=np.array([r['label'] for r in usable]);groups=np.array([r['group_sha256'] for r in usable]);weights,nreports=report_weights(usable)
    report={'test':receipt,'scored_official_reports':nreports,'positive_class':'material_error=1; synthetic weak and unverified',
            'warning':'Controlled synthetic stress-test discrimination, not natural generator error prevalence or clinical correctness probability',
            'report_weighting':'Each official report contributes equal total pair mass; duplicate reports remain within source-group bootstrap units',
            'models':{}}
    for column,name in enumerate(('mlp','adapted','lightgbm','xgboost')):
        z=np.array([key_to_scores[r['pair_sha256']][column] for r in usable]);t=temperatures[name]
        report['models'][name]={'temperature_from_train_calibration':t,
            'unique_pairs_uncalibrated':score_metrics(y,z),'unique_pairs_calibrated':score_metrics(y,z/t),
            'official_report_weighted_calibrated':score_metrics(y,z/t,weights),
            'source_group_bootstrap':bootstrap(y,z/t,groups,draws=args.bootstrap_draws),
            'diagnostics':diagnostics(usable,z,t)}
        atomic_json(out/'results.partial.json',report)
    atomic_json(out/'results.json',report);atomic_json(out/'status.json',{'phase':'complete','durable_pairs':len(key_to_scores),'scored_pairs':len(usable),'source_groups':len(set(groups)),'official_reports':nreports})

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('train','test','test-sha256','experiment','encoder','output','gpu-lease','locked-utc','selection-plan'):p.add_argument('--'+name,required=True)
    p.add_argument('--adapted-root',help='Training-validation-selected adapted run; original three-epoch run remains preserved')
    p.add_argument('--bootstrap-draws',type=int,default=500);return p

if __name__=='__main__':
    args=parser().parse_args()
    with gpu_lease(args.gpu_lease):main(args)
