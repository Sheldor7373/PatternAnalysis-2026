"""Read preserved histories and plot the checkpoint lineage without joining pauses."""
import argparse
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--runs-root',default='runs',help='Directory containing the recorded October run segments')
parser.add_argument('--continuation',required=True,help='Run root containing training/ for the final full continuation')
parser.add_argument('--output',default='runs/training-figure',help='Derived figures and source inventory; keep outside input runs')
args=parser.parse_args()
RUNS = Path(args.runs_root).resolve()
OUT = Path(args.output).resolve()
if OUT.is_relative_to(RUNS) and OUT == RUNS:
    raise ValueError('Figure output cannot be the input run root')
specs = [('LoRA', RUNS/'local-20261019-lora-main-01', 8750),
         ('LoRA', RUNS/'local-20261019-lora-main-03-b4-continuation', None),
         ('Full', RUNS/'proposed-full-baseline', 10181),
         ('Full', RUNS/'proposed-full-baseline-recovery-01', 40779),
         ('Full', Path(args.continuation), None),
         ('Superseded LoRA trial', RUNS/'local-20261019-lora-main-02-b2', 0)]
for _, source, _ in specs:
    if OUT.is_relative_to(source.resolve()):
        raise ValueError('Figure output must be outside every input run directory')
OUT.mkdir(parents=True,exist_ok=True)

def exposures(settings, step):
    size = settings['train_examples']
    args = settings['arguments']
    policy = settings.get('scheduler_policy')
    if policy:
        epoch, offset, updates = policy['anchor_epoch'], policy['anchor_offset'], step-policy['anchor_step']
    else:
        epoch, offset, updates = 0, 0, step
    count = epoch*size+offset
    for current in range(epoch, args['epochs']):
        remaining = size-(offset if current == epoch else 0)
        n = math.ceil(math.ceil(remaining/args['batch_size'])/args['accumulate'])
        taken = min(updates, n)
        count += min(remaining, taken*args['batch_size']*args['accumulate'])
        updates -= taken
        if updates <= 0:
            break
    return count

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
inventory = []
seen_labels = set()
colours = {'LoRA':'#2375ad', 'Full':'#cc5635'}
for model, run, cutoff in specs:
    training = run/'training'
    history = training/'history.jsonl'
    if not history.exists():
        continue
    rows = [json.loads(line) for line in history.read_text().splitlines() if line.strip()]
    settings = json.loads((training/'run.json').read_text())
    dev = [r for r in rows if 'dev_loss' in r]
    windows = [r for r in rows if r.get('event') == 'train_window']
    inventory.append(dict(model=model, run=str(run), history=str(history), history_rows=len(rows),
                          dates_utc=[rows[0].get('timestamp_utc'), rows[-1].get('timestamp_utc')],
                          steps=[min(r['step'] for r in rows), max(r['step'] for r in rows)],
                          train_windows=len(windows), development_measurements=len(dev),
                          first_dev_loss=dev[0]['dev_loss'] if dev else None,
                          last_dev_loss=dev[-1]['dev_loss'] if dev else None,
                          dev_examples=settings['dev_examples'],
                          dev_selected_ids_sha256=settings['dev_selected_ids_sha256'],
                          settings=str(training/'run.json'),
                          latest_pointer=str(training/'latest.json'),
                          metrics=str(training/'training_metrics.json') if (training/'training_metrics.json').exists() else None,
                          gpu_log=str(run/'gpu.jsonl') if (run/'gpu.jsonl').exists() else None,
                          gpu_samples=sum(1 for _ in (run/'gpu.jsonl').open()) if (run/'gpu.jsonl').exists() else 0,
                          lineage_max_step=cutoff, source_batch_size=settings['arguments']['batch_size'],
                          source_accumulate=settings['arguments']['accumulate'],
                          sessions=[{k:r.get(k) for k in ('timestamp_utc','pid','step','elapsed_seconds')}
                                    for r in rows if r.get('event') == 'session_start']))
    if cutoff == 0:
        continue
    # Remove replayed/uncheckpointed tail when a later session resumes earlier.
    canonical = []
    for row in rows:
        if row.get('event') == 'session_start':
            canonical = [r for r in canonical if r['step'] <= row['step']]
        if cutoff is None or row['step'] <= cutoff:
            canonical.append(row)
    # Keep each process/session separate: do not connect a line through pauses.
    sessions = {}
    for r in canonical:
        if r.get('event') not in ('train_window','dev_baseline','dev_diagnostic','session_evaluation','epoch_evaluation'):
            continue
        sessions.setdefault(r.get('pid'), []).append(r)
    for session in sessions.values():
        for axis, field in zip(axes, ('window_train_loss', 'dev_loss')):
            selected = [r for r in session if field in r]
            x = [r.get('consumed_examples', exposures(settings, r['step']))/settings['train_examples'] for r in selected]
            y = [r[field] for r in selected]
            if y:
                key = (model, field)
                axis.plot(x, y, color=colours[model], alpha=.8,
                          marker='.' if field == 'dev_loss' else None,
                          linewidth=1, label=model if key not in seen_labels else None)
                seen_labels.add(key)
for axis in axes:
    axis.set(xlabel='Training examples consumed / 97,411 (epoch equivalents)', ylabel='Token cross-entropy')
    axis.grid(alpha=.25)
    axis.legend()
axes[0].set_title('Token-weighted training-window loss\n(training mode; unsmoothed recorded windows)')
axes[1].set_title('Fixed development subset loss\n256 reports; evaluation mode; diagnostic and epoch-end checks')
fig.suptitle('Preserved training evidence — segments stop at valid checkpoint lineage')
fig.savefig(OUT/'loss-by-examples.png', dpi=180)
fig.savefig(OUT/'loss-by-examples.pdf')
(OUT/'inventory.json').write_text(json.dumps(inventory, indent=2))
print(json.dumps(inventory, indent=2))
