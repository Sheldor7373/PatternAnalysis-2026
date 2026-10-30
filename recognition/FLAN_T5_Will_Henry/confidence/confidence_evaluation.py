"""CPU scoring for a locked synthetic stress test; group-aware denominators."""
from collections import Counter
import itertools
import json
from pathlib import Path
import numpy as np
from confidence_core import sha,pair_id

THRESHOLDS=(.01,.05,.1,.2,.5,.8,.9,.95,1.)

def load_test(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    names=('features.jsonl','labels.jsonl','metadata.jsonl','scoring-mask.jsonl')
    for name in names:
        if sha(root/name)!=manifest['sha256'][name]:raise ValueError('TEST assembly identity changed')
    rows=[];keys=set();groups=set();official=set();report_group={};streams=[(root/n).open(encoding='utf-8-sig') for n in names]
    try:
        for lines in itertools.zip_longest(*streams):
            if any(x is None for x in lines):raise ValueError('Unaligned TEST files')
            feature,label,meta,mask=map(json.loads,lines)
            if set(feature)!={'source','candidate'}:raise ValueError('TEST feature metadata leakage')
            key=pair_id(feature['source'],feature['candidate'])
            if key!=meta['pair_sha256'] or key in keys:raise ValueError('Duplicated or changed TEST pair')
            keys.add(key);group={r['group_sha256'] for r in meta['lineage']}
            if len(group)!=1:raise ValueError('TEST pair crosses source components')
            group=next(iter(group));groups.add(group)
            if any(r['source_id'].startswith('train:') for r in meta['lineage']):raise ValueError('TRAIN source in TEST')
            mapped=sorted({i for r in meta['lineage'] for i in r['mapped_official_ids']});official.update(mapped)
            if not mapped:raise ValueError('TEST pair lacks official report mapping')
            for report in mapped:
                if report in report_group and report_group[report]!=group:raise ValueError('Official report crosses TEST groups')
                report_group[report]=group
            if mask['score'] and type(label['material_error']) is not bool:raise ValueError('Unresolved scored TEST label')
            rows.append(dict(feature,pair_sha256=key,group_sha256=group,mapped_ids=mapped,
                             label=int(label['material_error']) if mask['score'] else None,score=bool(mask['score']),
                             categories=sorted({r['slot'] for r in meta['lineage']})))
    finally:
        for stream in streams:stream.close()
    assert len(groups)==manifest['counts']['components'] and len(official)==manifest['counts']['official_mapped_rows']
    assert sum(r['score'] for r in rows)==manifest['counts']['usable_pairs']
    return rows,{'manifest_sha256':sha(root/'manifest.json'),'all_pairs':len(rows),'usable_pairs':sum(r['score'] for r in rows),
                 'held_pairs':sum(not r['score'] for r in rows),'source_groups':len(groups),'official_reports':len(official),
                 'feature_files_sha256':{n:manifest['sha256'][n] for n in names}}

def score_metrics(labels,logits,weights=None):
    from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
    y=np.asarray(labels);z=np.asarray(logits);p=1/(1+np.exp(-np.clip(z,-40,40)))
    w=np.ones(len(y),dtype=float) if weights is None else np.asarray(weights,dtype=float)
    if w.sum()<=0:raise ValueError('No metric mass')
    norm=w/w.sum();bins=np.minimum((p*10).astype(int),9)
    loss=np.logaddexp(0,z)-y*z
    ece=0.;reliability=[]
    for i in range(10):
        mask=bins==i;mass=w[mask].sum()
        if mass:
            prediction=np.average(p[mask],weights=w[mask]);truth=np.average(y[mask],weights=w[mask]);ece+=mass/w.sum()*abs(prediction-truth)
            reliability.append({'bin':i,'pairs':int(mask.sum()),'mass':float(mass),'mean_error_probability':float(prediction),'weak_error_rate':float(truth)})
    return {'auroc':float(roc_auc_score(y,p,sample_weight=w)),
            'error_auprc':float(average_precision_score(y,p,sample_weight=w)),
            'correct_auprc':float(average_precision_score(1-y,1-p,sample_weight=w)),
            'log_loss':float(np.sum(norm*loss)),'brier':float(np.sum(norm*(p-y)**2)),
            'balanced_accuracy_at_05':float(balanced_accuracy_score(y,p>=.5,sample_weight=w)),
            'weak_error_prevalence':float(np.sum(norm*y)),'ece_10_equal_width':float(ece),'reliability':reliability}

def report_weights(rows):
    counts=Counter(i for r in rows for i in r['mapped_ids'])
    return np.array([sum(1/counts[i] for i in r['mapped_ids']) for r in rows]),len(counts)

def bootstrap(labels,logits,groups,weights=None,draws=500,seed=3752):
    """Source components, not duplicated rows, are the resampling unit."""
    from sklearn.metrics import roc_auc_score,average_precision_score
    y=np.asarray(labels);z=np.asarray(logits);p=1/(1+np.exp(-np.clip(z,-40,40)))
    base=np.ones(len(y)) if weights is None else np.asarray(weights)
    unique,inverse=np.unique(groups,return_inverse=True);rng=np.random.default_rng(seed)
    values={k:[] for k in ('auroc','error_auprc','correct_auprc','log_loss','brier')}
    loss=np.logaddexp(0,z)-y*z
    for draw in range(draws):
        w=base*np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique))[inverse]
        values['auroc'].append(roc_auc_score(y,p,sample_weight=w))
        values['error_auprc'].append(average_precision_score(y,p,sample_weight=w))
        values['correct_auprc'].append(average_precision_score(1-y,1-p,sample_weight=w))
        values['log_loss'].append(np.average(loss,weights=w));values['brier'].append(np.average((p-y)**2,weights=w))
    return {'draws':draws,'seed':seed,'source_groups':len(unique),'unit':'report/image connected component; preserve all its pairs and mapped-report weights',
            'percentile_95':{k:np.quantile(v,[.025,.975]).tolist() for k,v in values.items()}}

def diagnostics(rows,logits,temperature=1.):
    y=np.array([r['label'] for r in rows]);p=1/(1+np.exp(-np.clip(np.asarray(logits)/temperature,-40,40)))
    risk=[]
    for threshold in THRESHOLDS:
        accepted=p<=threshold
        risk.append({'maximum_error_probability':threshold,'accepted_pairs':int(accepted.sum()),'pair_coverage':float(accepted.mean()),
                     'weak_error_risk':float(y[accepted].mean()) if accepted.any() else None,
                     'source_groups':len({r['group_sha256'] for r,yes in zip(rows,accepted) if yes}),
                     'official_reports':len({i for r,yes in zip(rows,accepted) if yes for i in r['mapped_ids']})})
    categories={}
    for category in sorted({c for r in rows for c in r['categories']}):
        mask=np.array([category in r['categories'] for r in rows]);error=mask&(y==1);correct=mask&(y==0)
        categories[category]={'pairs':int(mask.sum()),'weak_error_pairs':int(error.sum()),'weak_correct_pairs':int(correct.sum()),
                              'miss_rate_at_05':float(np.mean(p[error]<.5)) if error.any() else None,
                              'false_alarm_rate_at_05':float(np.mean(p[correct]>=.5)) if correct.any() else None}
    return {'threshold_policy':'predeclared, never selected on TEST; risk is synthetic weak-label error among accepted pairs',
            'risk_coverage':risk,'categories':categories,'category_policy':'multi-label lineage slices; a collapsed pair may appear in several slices'}
