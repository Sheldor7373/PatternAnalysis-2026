"""Cache one pinned FLAN checkpoint and report token lengths before GPU work."""

import argparse
from pathlib import Path
from huggingface_hub import model_info, snapshot_download
from transformers import AutoTokenizer
from dataset import PROMPT, length_summary, read_jsonl, write_json
from modules import MODEL


def main(args):
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    info = model_info(args.model, revision=args.revision)
    model_path = snapshot_download(args.model, revision=info.sha, cache_dir=cache,
                                   allow_patterns=["*.json", "*.model", "*.safetensors"])
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    result = {"model": args.model, "revision": info.sha, "local_path": model_path,
              "source_length": args.source_length, "target_length": args.target_length, "splits": {}}
    for split in ("train", "dev", "evaluation"):
        rows = read_jsonl(Path(args.data) / f"{split}.jsonl")
        stats = {}
        for field in ("input", "target"):
            lengths = []
            for start in range(0, len(rows), 512):
                texts = [(PROMPT if field == "input" else "") + r[field] for r in rows[start:start + 512]]
                lengths.extend(map(len, tokenizer(texts, truncation=False, verbose=False)["input_ids"]))
            limit = args.source_length if field == "input" else args.target_length
            stats[field] = {**length_summary(lengths), "truncated_rows": sum(n > limit for n in lengths),
                            "truncated_fraction": sum(n > limit for n in lengths) / len(lengths)}
        result["splits"][split] = stats
    write_json(Path(args.data) / "token_audit.json", result)
    print(f"Cached {args.model}@{info.sha} at {model_path}")
    print(f"Token audit: {Path(args.data) / 'token_audit.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--revision", default="main")
    p.add_argument("--data", default="data/prepared")
    p.add_argument("--cache", default="cache/models")
    p.add_argument("--source-length", type=int, default=384)
    p.add_argument("--target-length", type=int, default=256)
    main(p.parse_args())
