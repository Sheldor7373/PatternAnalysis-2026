"""Publish immutable checkpoints using a small atomic pointer file."""

import hashlib
import json
from pathlib import Path
import uuid

MARKER = "COMPLETE.json"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def file_digests(directory):
    directory = Path(directory)
    return {path.relative_to(directory).as_posix(): digest(path)
            for path in sorted(directory.rglob("*")) if path.is_file() and path.name != MARKER}


def mark_complete(directory):
    directory = Path(directory)
    files = file_digests(directory)
    if not files:
        raise ValueError("Cannot complete an empty checkpoint")
    (directory / MARKER).write_text(json.dumps({"schema_version": 1, "files": files}, sort_keys=True), encoding="utf-8")


def resolve_checkpoint(path):
    path = Path(path)
    if path.is_dir():
        directory = path
    else:
        pointer = path.with_suffix(".json")
        metadata = json.loads(pointer.read_text(encoding="utf-8"))
        directory = (pointer.parent / metadata["directory"]).resolve()
        if not directory.is_relative_to(pointer.parent.resolve()):
            raise ValueError("Checkpoint pointer leaves its run directory")
        if digest(directory / MARKER) != metadata["completion_sha256"]:
            raise ValueError("Checkpoint completion marker changed")
    try:
        marker = json.loads((directory / MARKER).read_text(encoding="utf-8"))
        if marker.get("schema_version") != 1 or not marker.get("files") or marker["files"] != file_digests(directory):
            raise ValueError("Checkpoint files are incomplete or changed")
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Checkpoint has no valid completion marker") from error
    return directory


def publish_checkpoint(alias, writer):
    """Writer fills a fresh directory; publish only after every file is complete.

    Failed/unpublished stages are retained for diagnosis. Existing published
    checkpoints are never edited. Pointer replacement is on the same filesystem.
    This protects interruption during saves; it is not a power-loss durability claim.
    """
    alias = Path(alias)
    alias.parent.mkdir(parents=True, exist_ok=True)
    stage = alias.parent / "checkpoints" / f"{alias.name}-{uuid.uuid4().hex}"
    stage.mkdir(parents=True)
    writer(stage)
    mark_complete(stage)
    pointer = alias.with_suffix(".json")
    temporary = pointer.with_name(f".{pointer.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps({"directory": stage.relative_to(alias.parent).as_posix(),
                                     "completion_sha256": digest(stage / MARKER)}), encoding="utf-8")
    temporary.replace(pointer)
    return stage
