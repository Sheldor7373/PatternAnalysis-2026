"""Post-training spectral variants of immutable, ordinary PEFT LoRA adapters.

Hard SVD minimises unweighted matrix reconstruction error, not task loss.
Soft shrinkage is the nuclear-norm proximal map (Cai/Candes/Shen, Thm2.1):
https://arxiv.org/abs/0810.3286 . This is not AdaLoRA training or SVD-LLM's
activation-aware whitening. All decomposition is on delta=(alpha/r)*B@A.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import uuid
import torch
from safetensors.torch import load_file,save_file
from checkpoint_storage import resolve_checkpoint,mark_complete,digest,file_digests


def factor_spectrum(u,values,vh,rank,scale):
    if rank<1 or rank>len(values) or not math.isfinite(scale) or scale<=0:
        raise ValueError('Invalid target rank or scale')
    roots=(values[:rank].clamp_min(0)/scale).sqrt()
    return (roots[:,None]*vh[:rank]).contiguous(),(u[:,:rank]*roots[None,:]).contiguous()


def spectral_factors(a,b,source_scale,method,rank,_decomposition=None):
    if a.ndim!=2 or b.ndim!=2 or b.shape[1]!=a.shape[0] or not 1<=rank<=a.shape[0] or not math.isfinite(source_scale) or source_scale<=0:
        raise ValueError('Invalid LoRA factors/rank/scaling')
    a,b=a.float(),b.float()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Nonfinite source adapter')
    delta=source_scale*(b@a)
    u,s,vh=torch.linalg.svd(delta,full_matrices=False) if _decomposition is None else _decomposition
    threshold=None
    if method in {'hard','full'}:
        values=s.clone();values[rank:]=0
    elif method=='soft':
        if rank>=len(s):raise ValueError('Soft threshold needs a following singular value')
        threshold=(float(s[rank-1])+float(s[rank]))/2
        values=(s-threshold).clamp_min(0)
        if int((values>0).sum())>rank:raise ValueError('Threshold rounding exceeded target rank')
    elif method=='zero':values=torch.zeros_like(s)
    elif method=='scale_half':values=s*.5
    else:raise ValueError('Unknown spectral method')
    # Keep alpha/r equal to source_scale. These factors represent delta/scale,
    # because PEFT multiplies their product by scale once during forward.
    if method=='zero':new_a,new_b=a.clone(),torch.zeros_like(b)
    elif method=='scale_half':new_a,new_b=a.clone(),b*.5
    else:new_a,new_b=factor_spectrum(u,values,vh,rank,source_scale)
    actual=source_scale*(new_b@new_a)
    target=(u*values[None,:])@vh
    if method=='zero':target=torch.zeros_like(delta)
    if method=='scale_half':target=delta*.5
    torch.testing.assert_close(actual,target,atol=2e-5,rtol=2e-4)
    original_energy=float(delta.double().square().sum())
    error=float((actual-delta).double().square().sum())
    stats={'shape':list(delta.shape),'source_rank':a.shape[0],'physical_rank':new_a.shape[0],
           'source_scale':source_scale,'new_scale':source_scale,'threshold':threshold,
           'threshold_tie':bool(method=='soft' and float(s[rank-1])==float(s[rank])),
           'positive_singular_values_after_shrinkage':int((values>0).sum()) if method=='soft' else None,
           'singular_values':s[:a.shape[0]].tolist(),
           'source_energy':original_energy,'update_energy':float(actual.double().square().sum()),
           'retained_energy_fraction':float(actual.double().square().sum())/original_energy if original_energy else 1.,
           'relative_frobenius_error':(error/original_energy)**.5 if original_energy else 0.,
           'factorization_max_error':float((actual-target).abs().max()),
           'hard_optimal_discarded_energy':float(s[rank:].double().square().sum()) if method=='hard' else None}
    if not torch.isfinite(actual).all():raise ValueError('Nonfinite derived update')
    return new_a,new_b,stats


def inspect_source(source):
    source=resolve_checkpoint(source).resolve()
    config=json.loads((source/'adapter_config.json').read_text())
    if (config.get('peft_type')!='LORA' or config.get('bias')!='none'
        or config.get('modules_to_save') or config.get('lora_bias')
        or config.get('use_dora') or config.get('use_rslora') or config.get('use_qalora')
        or config.get('rank_pattern') or config.get('alpha_pattern') or config.get('fan_in_fan_out')):
        raise ValueError('Study supports ordinary linear LoRA only; refuse hidden saved/bias/scaling changes')
    tensors=load_file(str(source/'adapter_model.safetensors'),device='cpu')
    names=sorted(n for n in tensors if n.endswith('.lora_A.weight'))
    expected=set(names)|{n.replace('.lora_A.weight','.lora_B.weight') for n in names}
    if not names or set(tensors)!=expected:raise ValueError('Unexpected adapter tensors: cannot establish zero equivalence')
    rank=config['r'];scale=config['lora_alpha']/rank
    if any(tensors[n].shape[0]!=rank or tensors[n.replace('.lora_A.weight','.lora_B.weight')].shape[1]!=rank for n in names):
        raise ValueError('Config rank disagrees with factor shapes')
    return source,config,tensors,names,scale


def build_variants(source,output):
    source,config,tensors,names,scale=inspect_source(source)
    output=Path(output).resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Choose a fresh output root separate from the source checkpoint')
    before=file_digests(source);marker=digest(source/'COMPLETE.json')
    output.mkdir(parents=True)
    source_rank=config['r']
    if source_rank%4:raise ValueError('This bounded study requires rank divisible by four')
    specs=[('zero','zero',source_rank),('hard_half','hard',source_rank//2),
           ('hard_quarter','hard',source_rank//4),('soft_half','soft',source_rank//2),
           ('scale_half','scale_half',source_rank),('full_reconstruction','full',source_rank)]
    report={'source':str(source),'source_completion_sha256':marker,'source_files_sha256':before,
            'builder_sha256':digest(Path(__file__)),'source_rank':source_rank,'source_scale':scale,
            'modules':len(names),'created_utc':datetime.now(timezone.utc).isoformat(),'variants':{}}
    derived={label:{'weights':{},'layers':{}} for label,_,_ in specs}
    # Decompose one scaled update at a time, once. Retain only small factors
    # for all variants, not 48 dense updates or repeated six-way SVD work.
    for name in names:
        other=name.replace('.lora_A.weight','.lora_B.weight')
        a,b=tensors[name].float(),tensors[other].float()
        decomposition=torch.linalg.svd(scale*(b@a),full_matrices=False)
        for label,method,rank in specs:
            new_a,new_b,stats=spectral_factors(a,b,scale,method,rank,decomposition)
            derived[label]['weights'][name]=new_a
            derived[label]['weights'][other]=new_b
            derived[label]['layers'][name]=stats
    for label,method,rank in specs:
        stage=output/('.stage-'+label+'-'+uuid.uuid4().hex);dest=output/label
        assert stage.resolve().is_relative_to(output) and dest.parent.resolve()==output
        stage.mkdir();weights=derived[label]['weights'];layers=derived[label]['layers']
        new_config=dict(config,r=rank,lora_alpha=scale*rank)
        if float(new_config['lora_alpha']).is_integer():new_config['lora_alpha']=int(new_config['lora_alpha'])
        assert new_config['lora_alpha']/rank==scale
        for path in source.iterdir():
            if path.name in {'COMPLETE.json','adapter_model.safetensors','adapter_config.json','experiment.json','README.md'}:continue
            if path.is_file():shutil.copy2(path,stage/path.name)
        save_file(weights,str(stage/'adapter_model.safetensors'),metadata={'format':'pt'})
        (stage/'adapter_config.json').write_text(json.dumps(new_config,indent=2),encoding='utf-8')
        settings=json.loads((source/'experiment.json').read_text())
        provenance={'label':label,'method':method,'source_completion_sha256':marker,
                    'source_rank':source_rank,'rank':rank,'alpha':new_config['lora_alpha'],'scale':scale,
                    'no_training':True,'no_norm_renormalisation':True,'decomposition':'float32 SVD of scaled B@A',
                    'builder_sha256':report['builder_sha256'],'layers':layers}
        settings.update(rank=rank,post_training_variant=provenance)
        (stage/'experiment.json').write_text(json.dumps(settings,indent=2),encoding='utf-8')
        mark_complete(stage);resolve_checkpoint(stage)
        stage.rename(dest)
        total=sum(v['source_energy'] for v in layers.values())
        report['variants'][label]={'path':str(dest),'completion_sha256':digest(dest/'COMPLETE.json'),
            'rank':rank,'alpha':new_config['lora_alpha'],'scale':scale,
            'global_update_energy_fraction':sum(v['update_energy'] for v in layers.values())/total,
            'global_relative_frobenius_error':(sum(v['source_energy']*v['relative_frobenius_error']**2 for v in layers.values())/total)**.5,
            'layers':layers}
    if file_digests(source)!=before or digest(source/'COMPLETE.json')!=marker:
        raise ValueError('Original source changed during derivation')
    report['source_unchanged']=True
    (output/'derivation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--cpu-threads',type=int,default=4);args=p.parse_args();torch.set_num_threads(args.cpu_threads)
    result=build_variants(args.source,args.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_files_sha256','variants']},indent=2))
