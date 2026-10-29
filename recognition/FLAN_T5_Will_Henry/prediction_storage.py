"""Durable generation batches bound to a fixed model, manifest and decoder.

Batch files are authoritative. A crash before publishing a batch may repeat that
uncommitted batch; published batches are verified and never generated again.
"""
import hashlib
import json
import os
from pathlib import Path
import uuid

from status_io import atomic_json


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode('utf-8')


class PredictionRun:
    def __init__(self, output, resume=False):
        self.output = Path(output)
        self.resume = resume
        self.predictions = []
        self.batches = []

    def __enter__(self):
        if self.output.exists() and any(self.output.iterdir()) and not self.resume:
            raise ValueError('Output directory is not empty; use --resume or a fresh path')
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = (self.output / '.prediction.lock').open('a+b')
        if self.lock.tell() == 0:
            self.lock.write(b'0')
            self.lock.flush()
        self.lock.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except Exception:
            self.lock.close()
            raise
        return self

    def __exit__(self, *unused):
        self.lock.close()

    def configure(self, identity, metadata, rows):
        self.rows = rows
        path = self.output / 'generation_run.json'
        fingerprint = hashlib.sha256(canonical(identity)).hexdigest()
        if path.exists():
            document = json.loads(path.read_text(encoding='utf-8'))
            if document.get('schema_version') != 1 or document.get('identity') != identity or document.get('identity_sha256') != fingerprint:
                raise ValueError('Resume identity changed: model, manifest, code or generation settings differ')
            self.metadata = document['metadata']
        else:
            unknown = [p for p in self.output.iterdir() if p.name != '.prediction.lock']
            if unknown:
                raise ValueError('No compatible resumable generation record; preserving existing output')
            self.metadata = metadata
            atomic_json(path, {'schema_version': 1, 'identity': identity,
                               'identity_sha256': fingerprint, 'metadata': metadata})
        self.identity = identity
        directory = self.output / 'batches'
        directory.mkdir(exist_ok=True)
        for path in sorted(directory.glob('*.json')):
            document = json.loads(path.read_text(encoding='utf-8'))
            payload = document.get('payload')
            if document.get('schema_version') != 1 or document.get('sha256') != hashlib.sha256(canonical(payload)).hexdigest():
                raise ValueError(f'Generation batch checksum failed: {path.name}')
            self._validate(payload)
            if path.name != f'{payload["start"]:08d}.json':
                raise ValueError('Generation batch filename disagrees with its cursor')
            self.predictions.extend(payload['rows'])
            self.batches.append(payload)
        self.progress('ready' if len(self.predictions) < len(rows) else 'generated')

    def _validate(self, batch):
        if batch['start'] != len(self.predictions) or not batch['rows']:
            raise ValueError('Generation batches are not a contiguous nonempty prefix')
        start = batch['start']
        expected = self.rows[start:start + len(batch['rows'])]
        if len(expected) != len(batch['rows']):
            raise ValueError('Generation batch extends beyond the selected manifest')
        for source, produced in zip(expected, batch['rows']):
            if any(produced.get(key) != value for key, value in source.items()) or not isinstance(produced.get('prediction'), str):
                raise ValueError('Generation batch rows differ from the selected manifest/order')
        for key in ('generation_seconds', 'wall_seconds'):
            if not isinstance(batch[key], (int, float)) or not 0 < batch[key] < float('inf'):
                raise ValueError(f'Invalid generation timing: {key}')

    def commit(self, batch):
        self._validate(batch)
        destination = self.output / 'batches' / f'{batch["start"]:08d}.json'
        if destination.exists():
            raise ValueError('Refusing to replace a committed generation batch')
        document = {'schema_version': 1, 'payload': batch,
                    'sha256': hashlib.sha256(canonical(batch)).hexdigest()}
        atomic_json(destination, document)
        self.predictions.extend(batch['rows'])
        self.batches.append(batch)
        self.progress('generating' if len(self.predictions) < len(self.rows) else 'generated')

    def progress(self, state):
        atomic_json(self.output / 'generation_status.json',
                    {'state': state, 'committed_examples': len(self.predictions),
                     'total_examples': len(self.rows), 'committed_batches': len(self.batches),
                     'note': 'Only checksum-verified published batches count as committed.'})

    def export_predictions(self):
        if len(self.predictions) != len(self.rows):
            raise ValueError('Cannot export an incomplete prediction set as final')
        destination = self.output / 'predictions.jsonl'
        temporary = destination.with_name('.predictions.' + uuid.uuid4().hex + '.tmp')
        with temporary.open('x', encoding='utf-8') as stream:
            for row in self.predictions:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
        return destination
