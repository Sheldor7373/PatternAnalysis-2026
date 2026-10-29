"""Choose a small development-only timing set outside the dev256 monitor.

This performs CPU data preparation only; it never loads a model, uses the GPU,
or reads the official validation manifest.
"""
import argparse
import hashlib
from pathlib import Path

from dataset import read_jsonl, select_rows, write_json, write_jsonl


def prepare(data, output, limit=64, monitor_limit=256, seed=3710):
    data, output = Path(data), Path(output)
    if limit < 1 or monitor_limit < 1:
        raise ValueError('Sample and monitor sizes must be positive')
    if output.resolve().is_relative_to(data.resolve()):
        raise ValueError('Pilot output must be outside the prepared data directory')
    if output.exists() or output.with_suffix('.selection.json').exists():
        raise ValueError('Choose a fresh pilot output; existing files are preserved')
    source = data/'dev.jsonl'
    development = read_jsonl(source)
    monitor_ids = {row['id'] for row in select_rows(development, monitor_limit, seed)}
    candidates = [row for row in development if row['id'] not in monitor_ids]
    if len(candidates) < limit:
        raise ValueError('Not enough development rows outside the monitoring subset')
    selected = select_rows(candidates, limit, seed+1)
    if len({row['id'] for row in selected}) != len(selected):
        raise ValueError('Pilot IDs must be unique')
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, selected)
    summary = {'scope':'training-derived development timing pilot; not final validation',
               'source_manifest_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
               'selected_manifest_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
               'selected_ids_sha256':hashlib.sha256('\n'.join(row['id'] for row in selected).encode()).hexdigest(),
               'examples':len(selected),'selection_seed':seed+1,
               'excluded_monitor_rows':len(monitor_ids),'monitor_seed':seed,
               'selection':'Deterministic sample from dev rows after excluding the fixed monitoring IDs'}
    write_json(output.with_suffix('.selection.json'),summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='data/prepared')
    parser.add_argument('--output',required=True)
    parser.add_argument('--limit',type=int,default=64)
    args=parser.parse_args()
    print(prepare(args.data,args.output,args.limit))
