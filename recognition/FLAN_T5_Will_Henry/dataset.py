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


def partition_rows(train, validation, seed=3710, dev_fraction=0.2):
    """Keep official validation examples held out; remove their duplicates from train.

    The official validation split is grouped by normalised report before splitting
    into development and final evaluation. Patient identifiers are not supplied:
    this is report-level hygiene, not evidence of patient independence.
    """
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be between zero and one")
    groups = {}
    for row in validation:
        groups.setdefault(row["key"], []).append(row)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    if len(keys) < 2:
        raise ValueError("Need at least two distinct validation reports")
    count = min(len(keys) - 1, max(1, round(len(keys) * dev_fraction)))
    dev_keys = set(keys[:count])
    heldout_keys = set(groups)
    heldout_images = {r["image"] for r in validation if r["image"]}
    seen, kept, overlap, duplicates = set(), [], 0, 0
    for row in train:
        if row["key"] in heldout_keys or (row["image"] and row["image"] in heldout_images):
            overlap += 1
        elif row["key"] in seen:
            duplicates += 1
        else:
            seen.add(row["key"])
            kept.append(row)
    # One row per unique report gives boilerplate reports one vote. Keep the
    # official row ids so results can also be mapped back to the supplied data.
    dev = [groups[key][0] for key in sorted(dev_keys)]
    evaluation = [groups[key][0] for key in sorted(heldout_keys - dev_keys)]
    # An image can have two different report strings. Group these by image too.
    eval_images = {r["image"] for r in evaluation if r["image"]}
    dev = [r for r in dev if not r["image"] or r["image"] not in eval_images]
    if not kept or not dev or not evaluation:
        raise ValueError("Audit left an empty split")
    stats = {"train_overlap_removed": overlap, "train_duplicate_reports_removed": duplicates,
             "validation_duplicate_reports": len(validation) - len(groups),
             "conflicting_validation_targets": sum(len({normalise(r['target']) for r in g}) > 1
                                                   for g in groups.values()),
             "counts": {"train": len(kept), "dev": len(dev), "evaluation": len(evaluation)}}
    return {"train": kept, "dev": dev, "evaluation": evaluation}, stats


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


def prepare(output, revision=DATA_REVISION, seed=3710):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw, files, missing = {}, {}, {}
    # The challenge test split is deliberately left untouched. Section 2.6
    # specifies held-out validation for quantitative evaluation.
    for split in ("train", "validation"):
        path = hf_hub_download(DATASET, f"data/{split}-00000-of-00001.parquet",
                              repo_type="dataset", revision=revision)
        files[split] = {"sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        raw[split], missing[split] = clean_rows(pq.read_table(path).to_pylist(), split)
    splits, stats = partition_rows(raw["train"], raw["validation"], seed)
    for name, rows in splits.items():
        write_jsonl(output / f"{name}.jsonl", rows)
        files[name] = {**files.get(name, {}),
                       "manifest_sha256": hashlib.sha256((output / f"{name}.jsonl").read_bytes()).hexdigest()}
    audit = {"dataset": DATASET, "revision": revision, "seed": seed, "files": files,
             "missing_pairs_removed": missing, "official_usable_counts": {k: len(v) for k, v in raw.items()},
             **stats, "source_counts": {k: dict(Counter(r["source"] for r in v)) for k, v in splits.items()},
             "limitation": "No patient IDs supplied; patient-independent evaluation cannot be certified.",
             "test_split": "Not downloaded or used. Final evaluation is a reserved part of official validation."}
    write_json(output / "audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/prepared")
    parser.add_argument("--revision", default=DATA_REVISION)
    parser.add_argument("--seed", type=int, default=3710)
    args = parser.parse_args()
    prepare(args.output, args.revision, args.seed)
