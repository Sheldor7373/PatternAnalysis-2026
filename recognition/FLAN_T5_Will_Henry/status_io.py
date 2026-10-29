"""Atomic monitoring JSON with bounded Windows sharing-error retries.

Checkpoint publication is deliberately outside this helper. Exhaustion raises;
the previous destination and the new uniquely named temporary file are retained.
"""
import json
import os
from pathlib import Path
import time
import uuid

RETRY_DELAYS = (0.05, 0.1, 0.2, 0.4, 0.8)  # six attempts, at most 1.55s sleep
RETRYABLE_WINERRORS = {5, 32, 33}  # access denied, sharing violation, lock violation


def atomic_json(path, value):
    path = Path(path)
    text = json.dumps(value, indent=2)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            os.replace(temporary, path)
            return
        except OSError as error:
            if getattr(error, 'winerror', None) not in RETRYABLE_WINERRORS:
                raise
            if attempt == len(RETRY_DELAYS):
                error.add_note(f'Status publication exhausted {attempt + 1} attempts; retained {temporary}')
                raise
            time.sleep(RETRY_DELAYS[attempt])
