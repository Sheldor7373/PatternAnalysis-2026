"""Teacher-forced reference likelihood on existing reports, never correctness confidence."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from checkpoint_storage import resolve_checkpoint
from dataset import ReportDataset, read_jsonl
from gpu_lease import gpu_lease
from inference_identity import bind_source, verify_source, effective_runtime, artifact_manifest
from modules import RadiologySimplifier, load_checkpoint, choose_device
from runtime import set_seed, sha256, synchronise, memory_stats, environment
from status_io import atomic_json

def row_losses(logits, labels):
    """Sum reference-token NLL per row, ignoring padding labels (-100)."""
    if logits.shape[:2] != labels.shape or not torch.isfinite(logits).all():
        raise ValueError('Invalid/nonfinite reference logits')
    # Match the Transformers T5 flattened-vocabulary loss path. The spatial
    # cross-entropy kernel used by transpose(1,2) can accumulate differently on CUDA.
    losses=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),labels.reshape(-1),reduction='none',ignore_index=-100).reshape_as(labels)
    counts=labels.ne(-100).sum(1)
    if bool((counts==0).any()):
        raise ValueError('Empty reference labels')
    return losses.sum(1), counts

def run(args):
    set_seed(3710); torch.set_num_threads(4)
    device=choose_device(args.device)
    runtime=effective_runtime('fp32',device)
    source=None; checkpoint=None
    if args.checkpoint:
        checkpoint=resolve_checkpoint(args.checkpoint)
        configuration=json.loads((checkpoint/'experiment.json').read_text())
        if configuration['mode']=='lora':
            source=bind_source(configuration['model'],configuration['revision'])
        model_identity={'checkpoint_completion_sha256':sha256(checkpoint/'COMPLETE.json'),
                        'files':artifact_manifest(checkpoint)}
    else:
        source=bind_source(args.model,args.revision)
        model_identity={'base':source['identity']}
    if source is not None:
        model_identity['base']=source['identity']
    identity={'model':model_identity,'manifest_sha256':sha256(args.manifest),
              'runtime':runtime,'packages':environment(device)['packages'],'source_length':384,'target_length':256,'batch_size':8,
              'diagnostic_code_sha256':sha256(__file__),
              'dependency_sha256':{name:sha256(Path(__file__).parent/name) for name in
                 ('dataset.py','modules.py','runtime.py','inference_identity.py','checkpoint_storage.py','gpu_lease.py','status_io.py')}}
    output=Path(args.output)
    if output.exists():
        prior=json.loads(output.read_text())
        assert prior['identity']==identity, 'Completed diagnostic identity mismatch'
        print('Existing identical diagnostic retained'); return
    with gpu_lease(args.gpu_lock):
        if checkpoint:
            model,tokenizer,_=load_checkpoint(checkpoint,device,base_source=(source['source'],source['revision']) if source else None)
        else:
            tokenizer=AutoTokenizer.from_pretrained(source['source'],revision=source['revision'])
            model=RadiologySimplifier(source['source'],source['revision'],mode='frozen').to(device)
        if source:verify_source(source)
        model.eval()
        rows=read_jsonl(args.manifest)
        dataset=ReportDataset(rows,tokenizer,384,256)
        collate=DataCollatorForSeq2Seq(tokenizer,model=model.model,label_pad_token_id=-100)
        records=[]; start=time.perf_counter()
        with torch.inference_mode():
            for offset in range(0,len(rows),8):
                batch=collate([dataset[i] for i in range(offset,min(offset+8,len(rows)))])
                batch={k:v.to(device) for k,v in batch.items()}
                result=model(**batch)
                sums,counts=row_losses(result.logits,batch['labels'])
                recomputed=sums.sum()/counts.sum()
                if not torch.allclose(recomputed,result.loss.float(),atol=2e-5,rtol=2e-5):
                    raise RuntimeError(f'Reference loss mismatch: per-row={float(recomputed)}, model={float(result.loss)}')
                for row,total,count in zip(rows[offset:offset+8],sums.tolist(),counts.tolist()):
                    nll=total/count
                    records.append({'id':row['id'],'reference_tokens':count,'reference_nll_sum':total,
                                    'reference_nll_per_token':nll,'geometric_mean_reference_token_probability':float(torch.exp(torch.tensor(-nll))),
                                    'source':row['source']})
                print(f'{len(records)}/{len(rows)} reference diagnostics',flush=True)
        synchronise(device)
        assert sha256(args.manifest)==identity['manifest_sha256']
        if source:verify_source(source)
        if checkpoint:assert artifact_manifest(checkpoint)==model_identity['files']
        value={'identity':identity,'rows':records,'token_weighted_nll':sum(r['reference_nll_sum'] for r in records)/sum(r['reference_tokens'] for r in records),
               'wall_seconds':time.perf_counter()-start,'scope':'Teacher-forced likelihood of truncated existing references; not calibrated factual correctness',**memory_stats(device)}
        value['rows_sha256']=hashlib.sha256(json.dumps(records,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        atomic_json(output,value)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint');p.add_argument('--model');p.add_argument('--revision',default='main')
    p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--gpu-lock',required=True)
    run(p.parse_args())
