"""Read-only checks for prepared JSONL splits; repeated references are not leakage.

Run: python verify_manifests.py --data data/prepared --expected-revision <SHA>
To check dataset.py's stored keys, also pass
--stored-key-contract nfkc-whitespace-casefold-sha256.
The default record fields match dataset.py. Override with --field input=report,
for example. Origins describe the official split encoded in row IDs, not the
prepared filename: train/dev default to train:<index>, evaluation to validation.
No patient independence or clinical correctness can be inferred by these checks.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys
import unicodedata


FIELDS = {"id": "id", "input": "input", "target": "target", "key": "key", "image": "image"}
ORIGINS = {"train": "train", "dev": "train", "evaluation": "validation"}
HASH_CONTRACT = "nfkc-whitespace-casefold-sha256"


def normalise(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def report_hash(text):
    return hashlib.sha256(normalise(text).encode("utf-8")).hexdigest()


def summary_hash(values):
    """Stable set summary, separate from the order-sensitive file digest."""
    payload = json.dumps(sorted(values), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_manifests(manifests, audit_path=None, *, fields=None, origins=None,
                     expected_dataset=None, expected_revision=None, stored_key_contract=None):
    """Return deterministic JSON-safe findings without writing any input file.

    Missing image identity is counted but never turned into a shared identity.
    Canonical hashes are independently computed for overlap checks. Stored keys
    are checked only when their normalization contract is explicitly supplied;
    remapping a field name does not imply a producer's hash algorithm.
    Repeated reports/references inside a split are descriptive, not failures.
    An audit is checked against actual file digests/counts, not trusted blindly.
    Audit/origin consistency does not authenticate original source provenance.
    """
    names = dict(FIELDS)
    if fields:
        if set(fields) - set(FIELDS):
            raise ValueError("Unknown logical record field")
        names.update(fields)
    if any(not isinstance(v, str) or not v for v in names.values()):
        raise ValueError("Field names must be nonempty strings")
    if stored_key_contract not in (None, HASH_CONTRACT):
        raise ValueError("Unsupported stored-key hash contract")
    expected_origins = dict(ORIGINS if origins is None else origins)
    checks, stats, identities = [], {}, {}

    def check(name, passed, **details):
        checks.append({"check": name, "passed": bool(passed), **details})

    check("required_splits", set(ORIGINS).issubset(manifests))
    for split in sorted(manifests):
        ids, reports, images = Counter(), Counter(), Counter()
        targets = defaultdict(set)
        problems = Counter()
        examples = defaultdict(list)
        digest = hashlib.sha256()
        rows, blank_lines, missing_images, stored_keys = 0, 0, 0, 0

        def problem(kind, line):
            problems[kind] += 1
            if len(examples[kind]) < 5:
                examples[kind].append(line)

        try:
            with Path(manifests[split]).open("rb") as stream:
                for line_number, raw in enumerate(stream, 1):
                    digest.update(raw)
                    if not raw.strip():
                        blank_lines += 1
                        continue
                    rows += 1
                    try:
                        row = json.loads(raw.decode("utf-8"))
                    except (ValueError, UnicodeError):
                        problem("invalid_json", line_number)
                        continue
                    if not isinstance(row, dict):
                        problem("non_object_record", line_number)
                        continue
                    values = {key: row.get(field) for key, field in names.items()}
                    for key in ("id", "input", "target"):
                        if not isinstance(values[key], str) or not values[key].strip():
                            problem("missing_or_invalid_" + key, line_number)
                    record_id = values["id"]
                    if isinstance(record_id, str) and record_id.strip():
                        ids[record_id] += 1
                        if split in expected_origins:
                            match = re.fullmatch(r"([^:]+):([0-9]+)", record_id)
                            if not match or match.group(1) != expected_origins[split]:
                                problem("wrong_origin_id", line_number)
                    text, target = values["input"], values["target"]
                    if isinstance(text, str) and text.strip():
                        key = report_hash(text)
                        reports[key] += 1
                        if names["key"] in row:
                            stored_keys += 1
                            if stored_key_contract is not None and values["key"] != key:
                                problem("report_hash_mismatch", line_number)
                        if isinstance(target, str) and target.strip():
                            targets[key].add(report_hash(target))
                    image = values["image"]
                    if image is None or image == "" or (isinstance(image, str) and not image.strip()):
                        missing_images += 1
                    elif not isinstance(image, str):
                        problem("invalid_image_identity", line_number)
                    else:
                        # Exact path identity; case and path components may be meaningful.
                        images[image.strip()] += 1
        except OSError:
            check(split + ".readable", False)
            continue
        check(split + ".readable", True)
        check(split + ".nonempty", rows > 0)
        check(split + ".records", not problems,
              issues={key: {"count": problems[key], "lines": examples[key]}
                      for key in sorted(problems)})
        check(split + ".unique_record_ids", all(n == 1 for n in ids.values()),
              repeated_rows=sum(n - 1 for n in ids.values()))
        stats[split] = {
            "rows": rows, "blank_lines": blank_lines, "manifest_sha256": digest.hexdigest(),
            "unique_ids": len(ids), "ids_sha256": summary_hash(ids),
            "unique_reports": len(reports), "report_hashes_sha256": summary_hash(reports),
            "stored_key_rows": stored_keys,
            "stored_key_rows_checked": stored_keys if stored_key_contract else 0,
            "repeated_report_rows": sum(n - 1 for n in reports.values()),
            "reports_with_multiple_targets": sum(len(v) > 1 for v in targets.values()),
            "unique_images": len(images), "image_ids_sha256": summary_hash(images),
            "missing_image_rows": missing_images,
        }
        identities[split] = {"id": set(ids), "report": set(reports), "image": set(images)}

    overlaps = []
    for left, right in itertools.combinations(sorted(identities), 2):
        counts = {key: len(identities[left][key] & identities[right][key])
                  for key in ("id", "report", "image")}
        overlaps.append({"left": left, "right": right, **counts})
        check(left + "/" + right + ".disjoint", not any(counts.values()), counts=counts)

    provenance = {"verified": False, "dataset": None, "revision": None,
                  "verification_scope": "Audit metadata and origin-label consistency only.",
                  "expected_origins": dict(sorted(expected_origins.items()))}
    if audit_path is None:
        if expected_dataset is not None or expected_revision is not None:
            check("audit.required_for_expected_provenance", False)
    else:
        try:
            audit_raw = Path(audit_path).read_bytes()
            audit = json.loads(audit_raw.decode("utf-8"))
            if not isinstance(audit, dict):
                raise ValueError("Audit must be an object")
        except (OSError, ValueError, UnicodeError):
            check("audit.readable_object", False)
        else:
            check("audit.readable_object", True)
            provenance.update(dataset=audit.get("dataset"), revision=audit.get("revision"),
                              audit_sha256=hashlib.sha256(audit_raw).hexdigest())
            for key, expected in (("dataset", expected_dataset), ("revision", expected_revision)):
                value = audit.get(key)
                check("audit." + key, isinstance(value, str) and bool(value.strip())
                      and (expected is None or value == expected))
            counts, files = audit.get("counts"), audit.get("files")
            for split in sorted(manifests):
                actual = stats.get(split, {})
                count = counts.get(split) if isinstance(counts, dict) else None
                file_info = files.get(split) if isinstance(files, dict) else None
                sha = file_info.get("manifest_sha256") if isinstance(file_info, dict) else None
                check("audit." + split + ".count", type(count) is int
                      and count == actual.get("rows"))
                check("audit." + split + ".manifest_sha256", isinstance(sha, str)
                      and sha == actual.get("manifest_sha256"))
            # When available, independently compare prepared evaluation size with
            # the audit's usable official split count; do not silently collapse references.
            official_counts = audit.get("official_usable_counts")
            official_eval = expected_origins.get("evaluation")
            if isinstance(official_counts, dict) and official_eval in official_counts:
                count = official_counts[official_eval]
                check("audit.evaluation.official_usable_count", type(count) is int
                      and count == stats.get("evaluation", {}).get("rows"))
            provenance["verified"] = all(c["passed"] for c in checks)

    return {"schema_version": 1, "ok": all(c["passed"] for c in checks),
            "canonical_hash_contract": HASH_CONTRACT, "stored_key_contract": stored_key_contract,
            "fields": names, "splits": stats, "overlaps": overlaps, "checks": checks,
            "provenance": provenance,
            "limitations": ["Image overlap uses only available exact image identities.",
                            "Repeated reports within evaluation retain their original weighting.",
                            "These checks cannot establish patient independence or clinical safety.",
                            "Audit metadata is checked for consistency, not independently authenticated."]}


def assignments(items, allowed=None):
    result = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator or not key or not value or (allowed is not None and key not in allowed):
            raise ValueError("Expected a supported NAME=VALUE assignment")
        if key in result:
            raise ValueError("Duplicate assignment: " + key)
        result[key] = value
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/prepared"))
    parser.add_argument("--split", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--field", action="append", default=[], metavar="LOGICAL=FIELD")
    parser.add_argument("--origin", action="append", default=[], metavar="SPLIT=OFFICIAL_SPLIT")
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--skip-audit", action="store_true",
                        help="Only verify records/leakage; provenance remains unverified")
    parser.add_argument("--expected-dataset")
    parser.add_argument("--expected-revision")
    parser.add_argument("--stored-key-contract", choices=[HASH_CONTRACT],
                        help="Explicit producer key contract; omitted keys remain unchecked")
    args = parser.parse_args(argv)
    if args.skip_audit and args.audit:
        parser.error("--skip-audit and --audit cannot be combined")
    try:
        manifests = {split: args.data / (split + ".jsonl") for split in ORIGINS}
        manifests.update(assignments(args.split))
        fields = assignments(args.field, FIELDS)
        origins = dict(ORIGINS)
        origins.update(assignments(args.origin))
        result = verify_manifests(
            manifests, None if args.skip_audit else (args.audit or args.data / "audit.json"),
            fields=fields, origins=origins, expected_dataset=args.expected_dataset,
            expected_revision=args.expected_revision, stored_key_contract=args.stored_key_contract)
    except ValueError as error:
        parser.error(str(error))
    json.dump(result, sys.stdout, sort_keys=True, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
