"""Two-text learned confidence representations; metadata never enters the encoder."""
import hashlib
import itertools
import json
import os
from pathlib import Path
import time
import uuid
import numpy as np
import torch
from torch import nn

PARTS = ('fit', 'validation', 'calibration')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''): h.update(block)
    return h.hexdigest()

def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    atomic_replace(temporary, path)

def atomic_replace(temporary, path):
    for delay in (0,.05,.1,.2,.4,.8):
        if delay:time.sleep(delay)
        try:os.replace(temporary,path);return
        except OSError as error:
            if getattr(error,'winerror',None) not in {5,32,33} or delay==.8:raise

def split_group(group):
    n = int(hashlib.sha256(('confidence-split-v1:' + group).encode()).hexdigest()[:16], 16) % 10000
    return 'fit' if n < 8000 else 'validation' if n < 9000 else 'calibration'

def pair_id(source, candidate):
    return hashlib.sha256(json.dumps([source, candidate], ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()

def load_assembly(root, view='conservative'):
    """Verify immutable row-aligned files and group separation before using labels."""
    root = Path(root); manifest = json.loads((root/'manifest.json').read_text())
    recorded = manifest.get('sha256', manifest.get('output_sha256', {}))
    result = {}; all_groups = {}; all_ids = set(); files = {}
    for part in PARTS:
        paths = [root/(part+'-'+kind+'.jsonl') for kind in ('features','labels','metadata','masks')]
        for path in paths:
            digest = sha(path)
            if recorded.get(path.name) != digest: raise ValueError('Assembly file hash missing or changed: ' + path.name)
            files[path.name] = digest
        rows = []
        streams = [p.open(encoding='utf-8-sig') for p in paths]
        try:
            for lines in itertools.zip_longest(*streams):
                if any(x is None for x in lines): raise ValueError('Unaligned assembly files')
                feature, label, meta, mask = map(json.loads, lines)
                if set(feature) != {'source','candidate'}: raise ValueError('Feature metadata leakage or missing text')
                if any(not isinstance(feature[k],str) or not feature[k].strip() for k in feature): raise ValueError('Empty text')
                if type(label.get('material_error')) is not bool: raise ValueError('Expected binary label')
                groups = {x['group_sha256'] for x in meta['lineage']}
                if len(groups) != 1: raise ValueError('Pair crosses source components')
                group = next(iter(groups))
                if split_group(group) != part: raise ValueError('Changed fixed group split')
                if any(not x['source_id'].startswith('train:') for x in meta['lineage']): raise ValueError('TEST source in TRAIN assembly')
                if group in all_groups and all_groups[group] != part: raise ValueError('Group leakage')
                all_groups[group] = part
                key = pair_id(feature['source'],feature['candidate'])
                if key != meta['pair_sha256'] or key in all_ids: raise ValueError('Wrong or duplicated pair identity')
                all_ids.add(key)
                if view not in mask: raise ValueError('Missing selection mask')
                if mask[view]: rows.append(dict(feature, pair_sha256=key, group_sha256=group, label=int(label['material_error'])))
        finally:
            for stream in streams: stream.close()
        result[part] = rows
    return result, {'manifest_sha256':sha(root/'manifest.json'), 'files_sha256':files,
                    'input_groups':len(all_groups), 'view':view,
                    'rows':{p:len(v) for p,v in result.items()},
                    'groups':{p:len({x['group_sha256'] for x in v}) for p,v in result.items()},
                    'label_status':'synthetic weak and unverified; report/image grouping is not patient grouping'}

def encode_pair(tokenizer, source, candidate, source_cap=1024, candidate_cap=1024):
    if source_cap<1 or candidate_cap<1:raise ValueError('Both token caps must be positive')
    if not source.strip() or not candidate.strip(): raise ValueError('Both source and candidate required')
    a = tokenizer.encode(source, add_special_tokens=False)
    b = tokenizer.encode(candidate, add_special_tokens=False)
    prefix = tokenizer.encode('source report: ', add_special_tokens=False)
    separator = tokenizer.encode('\ncandidate rewrite: ', add_special_tokens=False)
    left = a[:source_cap]; right = b[:candidate_cap]
    ids = prefix + left + separator + right + [tokenizer.eos_token_id]
    start = len(prefix); middle = start + len(left) + len(separator)
    return {'input_ids':ids,
            'source_mask':[0]*start+[1]*len(left)+[0]*(len(ids)-start-len(left)),
            'candidate_mask':[0]*middle+[1]*len(right)+[0],
            'source_tokens':len(a),'candidate_tokens':len(b),
            'source_truncated':len(a)>source_cap,'candidate_truncated':len(b)>candidate_cap}

def collate(examples, pad_id, device):
    size = max(len(x['input_ids']) for x in examples)
    result = {}
    for key in ('input_ids','source_mask','candidate_mask'):
        fill = pad_id if key=='input_ids' else 0
        result[key] = torch.tensor([x[key]+[fill]*(size-len(x[key])) for x in examples], device=device)
    result['attention_mask'] = torch.tensor([[1]*len(x['input_ids'])+[0]*(size-len(x['input_ids'])) for x in examples],device=device)
    return result

def pooled_states(encoder, batch):
    hidden = encoder(input_ids=batch['input_ids'],attention_mask=batch['attention_mask']).last_hidden_state.float()
    def mean(mask):
        mask = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden*mask).sum(1)/mask.sum(1).clamp_min(1)
    # Both means are contextual: all source and candidate tokens attend jointly.
    return torch.cat([mean(batch['source_mask']),mean(batch['candidate_mask'])],dim=-1)

class ConfidenceMLP(nn.Module):
    def __init__(self, dimensions, width=128, dropout=.2):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dimensions,width),nn.GELU(),nn.Dropout(dropout),nn.Linear(width,1))
    def forward(self, x): return self.layers(x).squeeze(-1)

class JointCritic(nn.Module):
    """Stage B uses the same encoder/pooling/head, with fit-only scaling fixed."""
    def __init__(self, encoder, head, mean, scale):
        super().__init__(); self.encoder=encoder; self.head=head
        self.register_buffer('mean',torch.as_tensor(mean)); self.register_buffer('scale',torch.as_tensor(scale))
    def forward(self,batch): return self.head((pooled_states(self.encoder,batch)-self.mean)/self.scale)

def metrics(labels, logits):
    from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, balanced_accuracy_score
    y = np.asarray(labels); z = np.asarray(logits); p = 1/(1+np.exp(-np.clip(z,-40,40)))
    bins = np.minimum((p*10).astype(int),9)
    ece = sum(np.mean(bins==i)*abs(np.mean(p[bins==i])-np.mean(y[bins==i])) for i in range(10) if np.any(bins==i))
    return {'rows':len(y),'error_fraction':float(y.mean()), 'log_loss':float(log_loss(y,p,labels=[0,1])),
            'auroc':float(roc_auc_score(y,p)) if len(set(y))==2 else None,
            'error_auprc':float(average_precision_score(y,p)),
            'correct_auprc':float(average_precision_score(1-y,1-p)),
            'balanced_accuracy_at_05':float(balanced_accuracy_score(y,p>=.5)),
            'brier':float(np.mean((p-y)**2)),'ece_10_equal_width':float(ece)}
