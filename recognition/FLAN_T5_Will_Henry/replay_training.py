"""Replay the recorded optimisation protocol from public configuration metadata.

Default is a command plan. GPU training requires explicit --execute. Historical
artifact hashes identify old files; fresh outputs have new metadata/hashes.
"""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path

STAGES={
 'lora':[('lora-b2',8750,17500),('lora-b4',77434,292233)],
 'full':[('full-initial',10181,20362),('full-recovery',40779,81558),('full-b4',93449,292233)],
 'extra':[('full-extra',117802,389644)]}

def command_plan(history,scenario,data,model,output_root,parent=None):
    plan=[]
    for index,(name,step,exposures) in enumerate(STAGES[scenario]):
        arguments=dict(history['runs'][name]['saved_arguments'])
        for key in ('resume','transition_from','max_minutes','stop_at_step'):arguments.pop(key,None)
        arguments.update(data=str(data),model=str(model),output=str(Path(output_root)/name))
        if index<len(STAGES[scenario])-1:arguments['stop_at_step']=step
        source=parent if index==0 else '${PREVIOUS_IMMUTABLE_CHECKPOINT}'
        if source is not None:arguments['transition_from']=str(source)
        command=[sys.executable,'-B',str(Path(__file__).with_name('train.py'))]
        for key,value in arguments.items():
            if value is False or value is None:continue
            command.append('--'+key.replace('_','-'))
            if value is not True:command.append(str(value))
        plan.append({'name':name,'expected_step':step,'expected_exposures':exposures,'command':command})
    return plan

def canonical_manifest(path):
    count=0;digest=hashlib.sha256()
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():continue
            row=json.loads(line);digest.update((json.dumps(row,sort_keys=True,ensure_ascii=False,separators=(',',':'))+'\n').encode());count+=1
    return count,digest.hexdigest()

def main(args):
    history=json.loads(Path(args.history).read_text(encoding='utf-8'))
    if args.scenario=='extra' and not args.source_checkpoint:
        if args.execute:raise ValueError('Extra-epoch execution requires a completed three-epoch source checkpoint')
        parent='${YOUR_COMPLETED_EPOCH3_LATEST}'
    else:parent=args.source_checkpoint
    plan=command_plan(history,args.scenario,args.data,args.model,args.output_root,parent)
    if not args.execute:
        print(json.dumps({'scope':'New protocol replay controls, not a claim these CLI strings were historically issued',
                          'execute':False,'phases':plan},indent=2));return
    import torch
    from checkpoint_storage import resolve_checkpoint
    for split in ('train','dev'):
        count,digest=canonical_manifest(Path(args.data)/(split+'.jsonl'))
        expected=history['data'][split]
        if (count,digest)!=(expected['rows'],expected['canonical_rows_sha256']):
            raise ValueError('Prepared data content/order differs from recorded protocol: '+split)
    root=Path(args.output_root).resolve();root.mkdir(parents=True,exist_ok=True)
    for phase in plan:
        command=list(phase['command']);output=root/phase['name']
        command[command.index('--output')+1]=str(output)
        if '--transition-from' in command:
            command[command.index('--transition-from')+1]=str(resolve_checkpoint(parent))
        done=False
        if (output/'latest.json').exists():
            saved=torch.load(resolve_checkpoint(output/'latest')/'trainer.pt',map_location='cpu',weights_only=False)
            if (saved['step'],saved['consumed_examples'])==(phase['expected_step'],phase['expected_exposures']):done=True
            elif saved['step']<phase['expected_step']:command.append('--resume')
            else:raise ValueError('Existing phase advanced beyond the recorded replay boundary')
        if not done:subprocess.run(command,check=True)
        parent=resolve_checkpoint(output/'latest')
        saved=torch.load(parent/'trainer.pt',map_location='cpu',weights_only=False)
        if (saved['step'],saved['consumed_examples'])!=(phase['expected_step'],phase['expected_exposures']):
            raise ValueError('Phase stopped before its required boundary; inspect and rerun to resume')
        print(json.dumps({'phase':phase['name'],'verified_step':saved['step'],'verified_exposures':saved['consumed_examples']}),flush=True)

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--history',default=str(Path(__file__).parent/'configs/historical_training.json'))
    p.add_argument('--scenario',choices=STAGES,required=True)
    p.add_argument('--data',default='data/prepared');p.add_argument('--model',required=True)
    p.add_argument('--output-root',required=True);p.add_argument('--source-checkpoint')
    group=p.add_mutually_exclusive_group();group.add_argument('--plan',action='store_true');group.add_argument('--execute',action='store_true')
    return p

if __name__=='__main__':main(parser().parse_args())
