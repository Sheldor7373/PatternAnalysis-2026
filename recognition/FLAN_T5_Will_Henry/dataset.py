"""Download and audit LaymanRRG; keep report duplicates out of held-out data."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

DATASET = "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track"
DATA_REVISION = "6783ab38f9de28c4372799346f51d3927ae7eb9a"
PROMPT = "Rewrite this radiology report in plain English. Preserve all findings, negation and uncertainty:\n"


def normalise(text):
    text = unicodedata.normalize("NFKC", str(text))
    return re.sub(r"\s+", " ", text).strip().casefold()


def report_key(text):
    return hashlib.sha256(normalise(text).encode("utf-8")).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_rows(rows, split):
    clean, missing = [], 0
    for index, row in enumerate(rows):
        source, target = row.get("radiology_report"), row.get("layman_report")
        if not isinstance(source, str) or not source.strip() or not isinstance(target, str) or not target.strip():
            missing += 1
            continue
        clean.append({"id": f"{split}:{index}", "source": row.get("source", "unknown"),
                      "image": row.get("images_path", ""), "input": source.strip(),
                      "target": target.strip(), "key": report_key(source)})
    return clean, missing


def partition_rows(train, validation, seed=3710, dev_fraction=0.1):
    """Derive dev from train; preserve every usable official validation row.

    Connected report/image groups stay together. This is report/image hygiene,
    not proof of patient independence: the schema does not supply patient IDs.
    """
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be between zero and one")
    heldout_keys = {r["key"] for r in validation}
    heldout_images = {r["image"] for r in validation if r["image"]}
    kept, overlap = [], 0
    for row in train:
        if row["key"] in heldout_keys or (row["image"] and row["image"] in heldout_images):
            overlap += 1
        else:
            kept.append(row)
    # Union-find also handles two different reports attached to the same image.
    parent = list(range(len(kept)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    previous = {}
    for index, row in enumerate(kept):
        identities = [("report", row["key"])]
        if row["image"]:
            identities.append(("image", row["image"]))
        for identity in identities:
            if identity in previous:
                parent[find(index)] = find(previous[identity])
            else:
                previous[identity] = index
    groups = {}
    for index, row in enumerate(kept):
        groups.setdefault(find(index), []).append(row)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    if len(keys) < 2:
        raise ValueError("Need at least two independent training groups")
    count = min(len(keys) - 1, max(1, round(len(keys) * dev_fraction)))
    dev_groups = set(keys[:count])
    training = [r for key in sorted(groups) if key not in dev_groups for r in groups[key]]
    dev = [r for key in sorted(dev_groups) for r in groups[key]]
    evaluation = list(validation)
    if not training or not dev or not evaluation:
        raise ValueError("Audit left an empty split")
    val_targets = {}
    for row in validation:
        val_targets.setdefault(row["key"], set()).add(normalise(row["target"]))
    stats = {"train_overlap_removed": overlap, "training_groups": len(groups),
             "dev_group_fraction": dev_fraction,
             "validation_duplicate_reports": len(validation) - len(val_targets),
             "conflicting_validation_targets": sum(len(v) > 1 for v in val_targets.values()),
             "counts": {"train": len(training), "dev": len(dev), "evaluation": len(evaluation)}}
    return {"train": training, "dev": dev, "evaluation": evaluation}, stats


def select_rows(rows, limit=None, seed=3710):
    if limit is None:
        return rows
    if limit < 1:
        raise ValueError("limit must be positive")
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    return [rows[i] for i in indices[:limit]]


class ReportDataset:
    def __init__(self, rows, tokenizer, max_source_length=384, max_target_length=256):
        self.rows, self.tokenizer = rows, tokenizer
        self.max_source_length, self.max_target_length = max_source_length, max_target_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        tokens = self.tokenizer(PROMPT + row["input"], truncation=True, max_length=self.max_source_length)
        tokens["labels"] = self.tokenizer(text_target=row["target"], truncation=True,
                                         max_length=self.max_target_length)["input_ids"]
        return tokens


def prepare(output, revision=DATA_REVISION, seed=3710, raw_dir=None):
    import pyarrow.parquet as pq
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw, files, missing = {}, {}, {}
    # The challenge test split is deliberately left untouched. Section 2.6
    # specifies held-out validation for quantitative evaluation.
    for split in ("train", "validation"):
        filename = f"{split}-00000-of-00001.parquet"
        if raw_dir:
            path = Path(raw_dir) / filename
        else:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(DATASET, f"data/{filename}", repo_type="dataset", revision=revision)
        files[split] = {"sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        table = pq.read_table(path)
        files[split].update(raw_rows=table.num_rows, fields={f.name: str(f.type) for f in table.schema})
        raw[split], missing[split] = clean_rows(table.to_pylist(), split)
    splits, stats = partition_rows(raw["train"], raw["validation"], seed)
    for name, rows in splits.items():
        write_jsonl(output / f"{name}.jsonl", rows)
        files[name] = {**files.get(name, {}),
                       "manifest_sha256": hashlib.sha256((output / f"{name}.jsonl").read_bytes()).hexdigest()}
    audit = {"dataset": DATASET, "revision": revision, "seed": seed, "files": files,
             "missing_pairs_removed": missing, "official_usable_counts": {k: len(v) for k, v in raw.items()},
             **stats, "source_counts": {k: dict(Counter(r["source"] for r in v)) for k, v in splits.items()},
             "lengths_characters": {k: {field: length_summary([len(r[field]) for r in v])
                                          for field in ("input", "target")} for k, v in splits.items()},
             "limitation": "No patient IDs supplied; patient-independent evaluation cannot be certified.",
             "test_split": "Not downloaded or used. Primary evaluation preserves all usable official validation rows.",
             "raw_file_provenance": "Caller-supplied raw files; verify hashes against pinned download" if raw_dir
             else "Downloaded via Hugging Face Hub at pinned revision"}
    write_json(output / "audit.json", audit)
    print(json.dumps(audit, indent=2))


def length_summary(values):
    values = sorted(values)
    return {"min": values[0], "median": values[len(values) // 2],
            "p95": values[min(len(values) - 1, int(len(values) * 0.95))],
            "max": values[-1], "mean": sum(values) / len(values)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/prepared")
    parser.add_argument("--revision", default=DATA_REVISION)
    parser.add_argument("--seed", type=int, default=3710)
    parser.add_argument("--raw-dir", help="Use already-downloaded parquet files without network access")
    args = parser.parse_args()
    prepare(args.output, args.revision, args.seed, args.raw_dir)
