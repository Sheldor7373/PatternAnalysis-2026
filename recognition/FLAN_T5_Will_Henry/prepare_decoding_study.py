"""Reserve disjoint development report/image groups before decoding selection.

This is a new development study, not a fresh test of the already-inspected
official validation set. Historical exclusions are only as complete as supplied
records; no patient IDs exist. All clinical text stays in ignored local files.
"""
import argparse
import hashlib
import random
from pathlib import Path
from dataset import read_jsonl, report_key, select_rows, write_json, write_jsonl


def components(rows):
    parent=list(range(len(rows))); previous={}
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,row in enumerate(rows):
        identities=[('report',report_key(row['input']))]
        if row.get('image'):identities.append(('image',row['image']))
        for key in identities:
            if key in previous:parent[find(i)]=find(previous[key])
            else:previous[key]=i
    groups={}
    for i,row in enumerate(rows):groups.setdefault(find(i),[]).append(row)
    return list(groups.values())


def reserve(rows, seen_rows, seed=3729, representative=192, confirmation=128,
            cutoff_representative=128, challenge=48, cutoff_challenge=32):
    seen_ids={r['id'] for r in seen_rows};seen_keys={report_key(r['input']) for r in seen_rows}
    seen_images={r.get('image') for r in seen_rows if r.get('image')}
    groups=components(rows)
    eligible=[g for g in groups if not any(r['id'] in seen_ids or report_key(r['input']) in seen_keys
                  or bool(r.get('image') and r['image'] in seen_images) for r in g)]
    rng=random.Random(seed);rng.shuffle(eligible)
    needed=representative+confirmation+cutoff_representative+challenge+cutoff_challenge
    if len(eligible)<needed:raise ValueError('Insufficient unused development groups')
    # Sampling unit is one random report from a randomly ordered component.
    # Draw representative samples BEFORE enriching by source properties.
    chosen=[rng.choice(g) for g in eligible]
    a=representative;b=a+confirmation;c=b+cutoff_representative
    samples={'decoding-representative':chosen[:a],'decoding-confirmation-reserved':chosen[a:b],
             'cutoff-representative-reserved':chosen[b:c]}
    remaining=chosen[c:]
    ranked=sorted(remaining,key=lambda r:(-len(r['input'].split()),r['id']))
    samples['cutoff-challenge-reserved']=ranked[:cutoff_challenge]
    used={r['id'] for r in ranked[:cutoff_challenge]};ranked=[r for r in ranked if r['id'] not in used]
    long=ranked[:challenge//2];used.update(r['id'] for r in long)
    import re
    cues=re.compile(r'\b(no|not|without|possible|suspected|questionable|cannot|could|unlikely)\b',re.I)
    cue_rank=sorted((r for r in ranked if r['id'] not in used),
                    key=lambda r:(-len(cues.findall(r['input'])),r['id']))
    samples['decoding-challenge']=long+cue_rank[:challenge-len(long)]
    return samples,{'development_groups':len(groups),'excluded_groups':len(groups)-len(eligible),
                    'eligible_groups':len(eligible),'seen_ids':len(seen_ids),'seed':seed,
                    'sampling_unit':'one random row per shuffled report/image component; group-balanced, not original row-weighted',
                    'challenge_selection':'input-only: longest sources and most negation/uncertainty cues; enriched, not prevalence',
                    'limits':'Recorded historical exclusions only; development informed checkpoint selection through other groups. No patient independence or fresh official-test claim.'}


def main(args):
    output=Path(args.output)
    if output.exists():raise ValueError('Choose a fresh output directory; reservations are immutable')
    rows=read_jsonl(args.development)
    seen=select_rows(rows,args.monitor_limit,args.monitor_seed)
    inventory=[]
    for name in args.seen:
        path=Path(name);values=read_jsonl(path)
        seen.extend(r for r in values if 'id' in r and 'input' in r)
        inventory.append({'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':len(values)})
    samples,summary=reserve(rows,seen)
    output.mkdir(parents=True)
    summary.update(source_sha256=hashlib.sha256(Path(args.development).read_bytes()).hexdigest(),
                   historical_manifests=inventory,monitor_limit=args.monitor_limit,monitor_seed=args.monitor_seed,
                   samples={})
    for name,values in samples.items():
        path=output/(name+'.jsonl');write_jsonl(path,values)
        summary['samples'][name]={'rows':len(values),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                                 'sources':{s:sum(r.get('source')==s for r in values) for s in sorted({r.get('source','unknown') for r in values})}}
    write_json(output/'reservation.json',summary)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--development',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seen',action='append',default=[])
    p.add_argument('--monitor-limit',type=int,default=256);p.add_argument('--monitor-seed',type=int,default=3710)
    import json
    print(json.dumps(main(p.parse_args()),indent=2))
