import pathlib,json,hashlib

def sha(b): return hashlib.sha256(b).hexdigest()

def read(p): return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))

def split(group):
 n=int(sha(('confidence-split-v1:'+group).encode())[:16],16)%10000
 return 'fit' if n<8000 else 'validation' if n<9000 else 'calibration'

def pair_disposition(labels, groups):
 if len(labels)>1: return 'conflict'
 if len(groups)>1: return 'cross_component'
 return 'retain'

def verify_resume(directory):
 m=read(directory/'manifest.json')
 return all(sha((directory/n).read_bytes())==h for n,h in m['output_sha256'].items())

def repair(original,side,allowed):
 if sha(original)!=side['original_checkpoint_sha256']: raise ValueError('repair checkpoint mismatch')
 obj=json.loads(original); expected={x['slot'] for x in allowed['flagged_slots']}; updates=side['updates']
 if {x['slot'] for x in updates}!=expected or len(updates)!=len(expected): raise ValueError('repair slot mismatch')
 old={r['slot']:r for r in obj['records']}
 if not expected<=old.keys(): raise ValueError('missing original repair slots')
 for r in updates: old[r['slot']]=r
 obj['records']=[old[r['slot']] for r in obj['records']]
 return obj
