"""Read-only, offline tokenizer/span preflight; no model imports or forwards.

Pure helpers use stdlib and a tokenizer interface. Transformers is imported
only by load_cached_tokenizer. A failed boundary/cap check stops at that case;
neither the manifest nor its fixed spans/candidates/caps are rewritten.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from negation_mapping import POLICY_VERSION, MappingFailure as PreflightFailure, map_source_span

PINNED_REVISION = "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"
MODELS = ("frozen", "lora_merged_copy")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def canonical(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def validate_contract(manifest):
    sources = {row["id"]: row for row in manifest["sources"]}
    if len(sources) != len(manifest["sources"]) or not sources:
        raise ValueError("Source identities must be nonempty and unique")
    if len(sources) != manifest["counts"]["sources"]:
        raise ValueError("Source count differs from the fixed manifest")
    if manifest["input_contract"]["tokenizer_revision"] != PINNED_REVISION:
        raise ValueError("Unexpected tokenizer revision")
    split_families = manifest["family_splits"]
    seen, text_splits = set(), defaultdict(set)
    for stage, families in split_families.items():
        if seen.intersection(families):
            raise ValueError("A family crosses splits")
        seen.update(families)
    for row in sources.values():
        if row["family_id"] not in split_families.get(row["split"], []):
            raise ValueError("Source family/split mismatch")
        if row["split"] != "heldout" and row["finding_id"] not in manifest["fit_selection_findings"]:
            raise ValueError("Heldout finding entered fit/selection")
        if row["finding"] != manifest["finding_terms"].get(row["finding_id"]):
            raise ValueError("Finding term and identity differ")
        heldout_term = manifest["finding_terms"][manifest["heldout_recipient_finding"]]
        if row["split"] != "heldout" and heldout_term in canonical(row["source"] + json.dumps(row["candidates"])):
            raise ValueError("Heldout term entered fit/selection text or candidates")
        if row["split"] == "heldout" and "recipient" in row["roles"] and row["finding_id"] != manifest["heldout_recipient_finding"]:
            raise ValueError("Unexpected heldout recipient finding")
        text_splits[canonical(row["source"])].add(row["split"])
        span = row["source_char_span"]
        if len(span) != 2 or any(type(index) is not int for index in span):
            raise ValueError("Character span must contain two integer offsets")
        start, stop = span
        if not 0 <= start < stop <= len(row["source"]) or row["source"][start:stop] != row["source_span_text"]:
            raise ValueError("Source character span differs from fixed text")
        for candidate in row["candidates"].values():
            if candidate["eos_included"] is not True:
                raise ValueError("The fixed candidates require one included EOS")
            if canonical(row["finding"]) not in canonical(candidate["text"]):
                raise ValueError("Candidate does not name its recipient finding")
    if any(len(parts) != 1 for parts in text_splits.values()):
        raise ValueError("Canonical source text crosses splits")
    cases = {row["id"]: row for row in manifest["intervention_cases"]}
    if len(cases) != len(manifest["intervention_cases"]):
        raise ValueError("Duplicate intervention identity")
    for stage in split_families:
        if sum(row["split"] == stage for row in cases.values()) != manifest["counts"][stage + "_interventions"]:
            raise ValueError("Intervention count differs from fixed stage")
    for case in cases.values():
        rec, donor = sources[case["recipient_id"]], sources[case["donor_id"]]
        if (rec["split"] != donor["split"] or rec["split"] != case["split"]
                or rec["family_id"] != donor["family_id"] or rec["finding_id"] == donor["finding_id"]
                or "recipient" not in rec["roles"] or "donor" not in donor["roles"]
                or case["candidate_owner"] != rec["id"]):
            raise ValueError("Intervention violates split/family/donor/candidate isolation")
        expected = "same_polarity" if rec["assertion"] == donor["assertion"] else "cross_polarity"
        if case["control"] != expected:
            raise ValueError("Incorrect donor-polarity relationship")
    self_ids = []
    for case in manifest["self_patch_cases"]:
        if case["recipient_id"] != case["donor_id"] or case["recipient_id"] not in sources:
            raise ValueError("Invalid self-patch identity")
        self_ids.append(case["recipient_id"])
    expected_self = {row["id"] for row in sources.values() if "recipient" in row["roles"]}
    if len(self_ids) != len(set(self_ids)) or set(self_ids) != expected_self:
        raise ValueError("Self patches must cover each recipient exactly once")
    return sources, cases


def token_record(ids, mask, eligible=None):
    if (len(ids) != len(mask) or not ids or any(type(x) is not int or x < 0 for x in ids)
            or any(type(x) is not bool for x in mask)):
        raise ValueError("Require token IDs and explicit Boolean mask of equal nonzero length")
    result = {"input_ids": ids, "attention_mask": mask, "token_count": sum(mask),
              "token_ids_sha256": digest(ids), "attention_mask_sha256": digest(mask),
              "active_tokens_sha256": digest([token for token, valid in zip(ids, mask) if valid])}
    if eligible is not None:
        if (len(eligible) != len(ids) or any(type(x) is not bool for x in eligible)
                or any(e and not valid for e, valid in zip(eligible, mask))):
            raise ValueError("Eligibility must be Boolean and a subset of attention")
        result.update(eligible_mask=eligible, eligible_mask_sha256=digest(eligible))
    result["forward_input_sha256"] = digest({k: result[k] for k in ("input_ids", "attention_mask", "eligible_mask") if k in result})
    return result


def bare_encoding(tokenizer, text):
    encoded = tokenizer(text, add_special_tokens=False, truncation=False,
                        return_offsets_mapping=True, return_attention_mask=True)
    ids, offsets = list(encoded["input_ids"]), [list(pair) for pair in encoded["offset_mapping"]]
    if not ids or len(ids) != len(offsets) or encoded.get("attention_mask") != [1] * len(ids):
        raise PreflightFailure("encoding", "Expected complete unpadded bare encoding and explicit all-one mask")
    if any(type(token) is not int or token < 0 for token in ids):
        raise PreflightFailure("encoding", "Invalid token IDs")
    if tokenizer.eos_token_id in ids:
        raise PreflightFailure("eos", "Bare encoding already contains EOS; no silent EOS removal")
    covered = set()
    for pair in offsets:
        if len(pair) != 2 or any(type(x) is not int for x in pair) or not 0 <= pair[0] < pair[1] <= len(text):
            raise PreflightFailure("offsets", "Invalid or zero-width bare-token offset")
        covered.update(range(*pair))
    if any(not char.isspace() and index not in covered for index, char in enumerate(text)):
        raise PreflightFailure("coverage", "Offsets omit non-whitespace source/candidate characters")
    return ids, offsets


def encode_source(tokenizer, row, contract):
    prompt, source = contract["prompt"], row["source"]
    text = prompt + source
    ids, offsets = bare_encoding(tokenizer, text)
    ids.append(tokenizer.eos_token_id)
    offsets.append([0, 0])
    if len(ids) > contract["max_source_tokens"]:
        raise PreflightFailure("source_cap", f"Complete source has {len(ids)} tokens including EOS; cap={contract['max_source_tokens']}")
    start, stop = [len(prompt) + value for value in row["source_char_span"]]
    mapped = map_source_span(text, len(prompt), row["source_char_span"], ids, offsets,
                             [True] * len(ids), tokenizer.all_special_ids)
    result = token_record(ids, [True] * len(ids), mapped["eligible_mask"])
    result.update(offset_mapping=offsets, token_span=mapped["token_span"], span_mapping=mapped["mapping"],
                  prompt_characters=len(prompt), combined_char_span=[start, stop],
                  source_span_text=row["source_span_text"], eos_included=True)
    return result


def encode_candidate(tokenizer, candidate, cap):
    ids, _ = bare_encoding(tokenizer, candidate["text"])
    ids.append(tokenizer.eos_token_id)
    if len(ids) > cap:
        raise PreflightFailure("candidate_cap", f"Complete candidate has {len(ids)} tokens including EOS; cap={cap}")
    if ids.count(tokenizer.eos_token_id) != 1 or ids[-1] != tokenizer.eos_token_id:
        raise PreflightFailure("eos", "Candidate must contain exactly one final EOS")
    result = token_record(ids, [True] * len(ids))
    result.update(target_mask=result.pop("attention_mask"), target_mask_sha256=result.pop("attention_mask_sha256"),
                  eos_included=True)
    return result


def preflight(manifest, tokenizer, identity=None):
    sources, _ = validate_contract(manifest)
    if (not getattr(tokenizer, "is_fast", False) or type(tokenizer.eos_token_id) is not int
            or tokenizer.eos_token_id == tokenizer.pad_token_id):
        raise ValueError("Require a fast tokenizer with distinct configured EOS/pad IDs")
    report = {"schema_version": 1, "status": "PASS", "protocol_sha256": digest(manifest),
              "tokenizer": identity or {"kind": "synthetic tokenizer fixture"},
              "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id,
              "special_token_ids": sorted(set(tokenizer.all_special_ids)), "mapping_policy_version": POLICY_VERSION,
              "records": [], "failure": None, "manifest_modified": False,
              "scope": "tokenization/structural checks only; not label approval or model evidence"}
    before = digest(manifest)
    for row in sources.values():
        component = "source"
        try:
            encoded = encode_source(tokenizer, row, manifest["input_contract"])
            candidates = {}
            for component, candidate in row["candidates"].items():
                candidates[component] = encode_candidate(tokenizer, candidate, manifest["input_contract"]["max_candidate_tokens"])
            report["records"].append({"id": row["id"], "split": row["split"], "family_id": row["family_id"],
                                      "source": encoded, "candidates": candidates})
        except PreflightFailure as error:
            report.update(status="FAIL", failure={"id": row["id"], "component": component,
                                                   "code": error.code, "detail": str(error)})
            break
    if digest(manifest) != before:
        raise RuntimeError("Preflight unexpectedly modified its manifest")
    report["completed_sources"] = len(report["records"])
    return report


def load_cached_tokenizer(directory, revision):
    directory = Path(directory).resolve()
    if revision != PINNED_REVISION or not directory.is_dir() or not directory.name.endswith(revision):
        raise ValueError("Require the explicit local cache directory labelled with the pinned revision")
    names = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "spiece.model", "config.json")
    files = {}
    for name in names:
        path = directory / name
        data = path.read_bytes()
        files[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    from transformers import AutoTokenizer  # tokenizer only; never AutoModel
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True,
                                              use_fast=True, trust_remote_code=False)
    if tokenizer.eos_token_id != config["eos_token_id"] or tokenizer.pad_token_id != config["pad_token_id"]:
        raise ValueError("Tokenizer special IDs differ from cached configuration metadata")
    return tokenizer, {"kind": "offline cached tokenizer", "path": str(directory), "revision": revision,
                       "class": type(tokenizer).__module__ + "." + type(tokenizer).__name__, "is_fast": tokenizer.is_fast,
                       "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id,
                       "load_options": {"local_files_only": True, "use_fast": True, "trust_remote_code": False},
                       "files": files, "verification": "Local revision-labelled directory and file hashes; no remote lookup or weight-file read"}


def main(argv=None, loader=load_cached_tokenizer):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    raw = args.manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.expected_manifest_sha256:
        parser.error("Manifest bytes differ from the approved expected SHA256")
    output = args.output.resolve()
    if (output.exists() or output.is_relative_to(args.manifest.resolve().parent)
            or output.is_relative_to(args.tokenizer.resolve())):
        parser.error("Choose a new output outside protocol and tokenizer directories")
    manifest = json.loads(raw)
    validate_contract(manifest)
    tokenizer, identity = loader(args.tokenizer, manifest["input_contract"]["tokenizer_revision"])
    report = preflight(manifest, tokenizer, identity)
    report["manifest_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "status": report["status"], "completed_sources": report["completed_sources"],
                      "failure": report["failure"]}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
