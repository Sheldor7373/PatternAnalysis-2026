"""Deterministic arithmetic over saved scores; no tokenizer/model execution.

Input receipts establish internal consistency, not authenticity of checkpoint,
logit or parameter measurements. The failed real preflight cannot be consumed.
Synthetic fixtures exercise the calculations without authorising any fitting.
"""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re

from negation_preflight import MODELS, digest, token_record, validate_contract
from negation_mapping import POLICY_VERSION, map_source_span

AGGREGATION = "sum; padding excluded; no length normalisation"
CONDITIONS = {"learned_true", "permuted_3720", "random_3711", "random_3712", "random_3713"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value), "Require a finite number, not a Boolean")
    return value


def sha(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "Require a lowercase SHA256 receipt")
    return value


def mean(values):
    values = list(values)
    require(bool(values), "Cannot average an empty cell")
    return math.fsum(values) / len(values)


def indexed(rows, key, expected):
    result = {}
    for row in rows:
        identity = key(row)
        require(identity not in result, "Duplicate saved score/record identity")
        result[identity] = row
    require(set(result) == set(expected), "Missing, extra or mixed-stage saved score/record identities")
    return result


def validate_preflight(manifest, report):
    sources, _ = validate_contract(manifest)
    require(report["status"] == "PASS" and report["failure"] is None, "A failed/partial preflight cannot be scored")
    require(report["protocol_sha256"] == digest(manifest), "Preflight protocol hash mismatch")
    require(report["completed_sources"] == len(sources) and report["manifest_modified"] is False,
            "Require complete, unmodified preflight")
    records = indexed(report["records"], lambda r: r["id"], sources)
    eos = report["eos_token_id"]
    require(type(eos) is int and eos >= 0 and type(report["pad_token_id"]) is int
            and report["pad_token_id"] >= 0 and eos != report["pad_token_id"], "Invalid EOS/pad receipt")
    contract = manifest["input_contract"]
    require(report["mapping_policy_version"] == POLICY_VERSION, "Unsupported span-mapping policy")
    specials = report["special_token_ids"]
    require(isinstance(specials, list) and all(type(n) is int and n >= 0 for n in specials)
            and len(specials) == len(set(specials)) and eos in specials and report["pad_token_id"] in specials,
            "Invalid special-token identity receipt")
    for identity in sorted(records):
        row, saved = sources[identity], records[identity]
        require((saved["split"], saved["family_id"]) == (row["split"], row["family_id"]), "Preflight split/family mismatch")
        require(set(saved["candidates"]) == {"positive", "negative"}, "Only the fixed recipient candidate pair is allowed")
        for name, record in [("source", saved["source"])] + sorted(saved["candidates"].items()):
            is_source = name == "source"
            mask = record["attention_mask"] if is_source else record["target_mask"]
            computed = token_record(record["input_ids"], mask, record["eligible_mask"] if is_source else None)
            for field, value in computed.items():
                saved_field = ({"attention_mask": "target_mask", "attention_mask_sha256": "target_mask_sha256"}.get(field, field)
                               if not is_source else field)
                require(record[saved_field] == value, "Mask-aware token hash/count receipt mismatch")
            # This preflight is explicitly unpadded; no hidden masked EOS is accepted.
            require(all(mask) and record["eos_included"] is True and record["input_ids"].count(eos) == 1
                    and record["input_ids"][-1] == eos, "Require complete unpadded tokens and exactly one final EOS")
            require(len(mask) <= contract["max_source_tokens" if is_source else "max_candidate_tokens"], "Token cap exceeded")
        source = saved["source"]
        offsets, eligible = source["offset_mapping"], source["eligible_mask"]
        text = contract["prompt"] + row["source"]
        start, stop = [len(contract["prompt"]) + n for n in row["source_char_span"]]
        require(source["prompt_characters"] == len(contract["prompt"])
                and source["combined_char_span"] == [start, stop] and source["source_span_text"] == row["source_span_text"],
                "Prompt-offset/source-span receipt mismatch")
        require(len(offsets) == len(eligible) and offsets[-1] == [0, 0] and not eligible[-1], "Invalid source offsets/EOS eligibility")
        mapped = map_source_span(text, len(contract["prompt"]), row["source_char_span"], source["input_ids"],
                                 offsets, source["attention_mask"], specials)
        require(source["token_span"] == mapped["token_span"] and eligible == mapped["eligible_mask"]
                and source["span_mapping"] == mapped["mapping"], "Shared span/eligibility policy receipt mismatch")
    return records


def score_margin(row, candidate_records):
    for name in ("positive", "negative"):
        score, candidate = row[name], candidate_records[name]
        require(finite(score["summed_logprob"]) <= 0, "A summed log probability cannot be positive")
        require(type(score["token_count"]) is int and score["token_count"] == candidate["token_count"]
                and score["eos_included"] is True and score["aggregation"] == AGGREGATION,
                "Score count/EOS/summed-mask semantics mismatch")
        for field in ("token_ids_sha256", "target_mask_sha256"):
            require(score[field] == candidate[field], "Score candidate token/mask hash mismatch")
    margin = row["positive"]["summed_logprob"] - row["negative"]["summed_logprob"]
    finite(margin)
    if "margin" in row:
        require(finite(row["margin"]) == margin, "Provided margin differs from raw candidate sums")
    return margin


def equal_family(rows, value):
    families = defaultdict(list)
    for row in rows:
        families[row["family_id"]].append(row[value])
    family_means = {family: mean(families[family]) for family in sorted(families)}
    return {"family_means": family_means, "equal_family_mean": mean(family_means.values())}


def preference(margin):
    return "present" if margin > 0 else "absent" if margin < 0 else "tie"


def substantive_gate(input_effects):
    pairs = [{**row, "success": row["I"] >= math.log(2) and row["M_present"] > 0 and row["M_absent"] < 0}
             for row in input_effects]
    findings = sorted({row["finding_id"] for row in pairs})
    require(len(pairs) == 8 and len(findings) == 2 and all(sum(r["finding_id"] == f for r in pairs) == 4 for f in findings),
            "Substantive gate requires all eight fixed pairs")
    by_finding = {finding: sum(r["success"] for r in pairs if r["finding_id"] == finding) for finding in findings}
    family_means = equal_family(pairs, "I")["family_means"]
    passed = sum(r["success"] for r in pairs) >= 6 and min(by_finding.values()) >= 3 and all(v > 0 for v in family_means.values())
    return {"status": "PASS" if passed else "FAIL", "threshold_nats": math.log(2), "successful_pairs": sum(r["success"] for r in pairs),
            "total_pairs": 8, "successes_by_finding": by_finding, "family_mean_I": family_means, "pairs": pairs}


def mechanical_gate(evidence, self_rows, pilot_ids, baseline):
    """Consume explicitly declared measurement receipts; do not claim to recheck tensors."""
    missing, failures = [], []
    evidence = evidence or {}
    for model in MODELS:
        item = evidence.get(model)
        if item is None:
            missing.append(model)
            continue
        before, after = sha(item["parameters_before_sha256"]), sha(item["parameters_after_sha256"])
        if before != after:
            failures.append(model + ":parameters_changed")
        errors = indexed(item["self_logits"], lambda r: r["source_id"], pilot_ids)
        for source_id, error in errors.items():
            ratio = finite(error["max_normalised_error"])
            require(ratio >= 0 and error["atol"] == 1e-6 and error["rtol"] == 1e-6, "Incorrect self-logit tolerance receipt")
            if ratio > 1:
                failures.append(model + ":self_logits:" + source_id)
        if model == "lora_merged_copy":
            merged = indexed(item["merge_equivalence"], lambda r: r["source_id"], pilot_ids)
            for source_id, error in merged.items():
                ratio = finite(error["max_normalised_logit_error"])
                require(ratio >= 0 and error["atol"] == 1e-5 and error["rtol"] == 1e-5, "Incorrect merge tolerance receipt")
                reference = finite(error["reference_margin"])
                actual = baseline[(model, source_id)]
                if ratio > 1 or abs(actual - reference) > 1e-5 + 1e-5 * abs(reference):
                    failures.append(model + ":merge_equivalence:" + source_id)
    for row in self_rows:
        if row["absolute_effect"] > 1e-6:
            failures.append(row["model"] + ":self_margin:" + row["source_id"])
    return {"status": "FAIL" if failures else "UNAVAILABLE" if missing else "PASS", "missing": missing,
            "failures": sorted(failures), "evidence": evidence,
            "receipt_semantics": "Declared tensor measurements, not independently verified here. max_normalised_error=max(abs(actual-reference)/(atol+rtol*abs(reference))) over all compared valid logits."}


def summarise(manifest, preflight, data):
    sources, cases = validate_contract(manifest)
    records = validate_preflight(manifest, preflight)
    require(data["protocol_sha256"] == digest(manifest) and data["preflight_sha256"] == digest(preflight), "Saved score provenance hash mismatch")
    require(data["data_kind"] in {"synthetic_test_scores", "saved_model_scores"}, "Declare score provenance kind")
    stage, condition = data["stage"], data["basis_condition"]
    require(set(data["checkpoint_ids"]) == set(MODELS), "Require both fixed checkpoints")
    for identity in data["checkpoint_ids"].values():
        sha(identity)
    if stage == "pilot":
        require(condition == "coordinate_e0", "Pilot must use the fixed unlearned coordinate")
        recipient_ids = set(manifest["screened_pilot"]["recipient_ids"])
    elif stage == "substantive":
        require(condition == "none", "Substantive gate is baseline-only")
        recipient_ids = set(manifest["substantive_signal_before_fitting"]["panel_source_ids"])
    else:
        require(stage in {"selection", "heldout"} and condition in CONDITIONS, "Unknown stage or prespecified basis condition")
        recipient_ids = {r["id"] for r in sources.values() if r["split"] == stage and "recipient" in r["roles"]}
    if stage != "substantive":
        require(set(data["basis_ids"]) == set(MODELS), "Require one fixed basis identity per checkpoint")
        for identity in data["basis_ids"].values():
            sha(identity)
    if stage == "selection" and condition == "learned_true":
        require(type(data["snapshot_step"]) is int and data["snapshot_step"] in (32, 64), "Only selection snapshots32/64 are allowed")
    if stage == "heldout":
        lock = data["selection_lock"]
        for field in ("protocol_sha256", "preflight_sha256", "checkpoint_ids", "basis_ids", "basis_condition"):
            require(lock[field] == data[field], "Heldout selection-lock mismatch")
        sha(lock["selection_summary_sha256"])
    baseline_rows = indexed(data["baselines"], lambda r: (r["model"], r["source_id"]),
                            {(model, identity) for model in MODELS for identity in recipient_ids})
    baseline = {key: score_margin(row, records[key[1]]["candidates"]) for key, row in baseline_rows.items()}
    active_cases = {} if stage == "substantive" else {identity: c for identity, c in cases.items() if c["recipient_id"] in recipient_ids}
    self_cases = {} if stage == "substantive" else {"self:" + identity: {"recipient_id": identity, "donor_id": identity} for identity in recipient_ids}
    all_cases = {**active_cases, **self_cases}
    patches = indexed(data["patches"], lambda r: (r["model"], r["case_id"]),
                      {(model, identity) for model in MODELS for identity in all_cases})
    patched_margins = {key: score_margin(row, records[all_cases[key[1]]["recipient_id"]]["candidates"]) for key, row in patches.items()}
    report = {"schema_version": 1, "data_kind": data["data_kind"], "stage": stage, "basis_condition": condition,
              "protocol_sha256": digest(manifest), "preflight_sha256": digest(preflight), "scores_sha256": digest(data),
              "checkpoint_ids": data["checkpoint_ids"], "basis_ids": data.get("basis_ids"), "snapshot_step": data.get("snapshot_step"),
              "selection_lock": data.get("selection_lock"), "models": {}, "fitting_authorised": False,
              "scope": manifest["claim_scope"], "optional_readouts_enabled": False,
              "limitations": ["No model execution, tensor authenticity, human-label approval, clinical confidence or safety is established.",
                              "A positive P need not cross zero in either direction. Shared cardiac prefixes cancel; donor-finding substitution is unmeasured.",
                              "Fixed small family panel; no uncertainty interval, null-distribution p-value or population-generalisation claim.",
                              "Hash-bound lock receipts establish consistency, not independent evidence of selection timing or authorisation."],
              "raw_baselines": [baseline_rows[key] for key in sorted(baseline_rows)],
              "raw_patches": [patches[key] for key in sorted(patches)]}
    for model in MODELS:
        pairs = {}
        for identity in sorted(recipient_ids):
            row = sources[identity]
            pairs.setdefault((row["family_id"], row["finding_id"]), {})[row["assertion"]] = baseline[(model, identity)]
        input_effects = []
        for (family, finding), margins in sorted(pairs.items()):
            require(set(margins) == {"absent", "present"}, "Input contrast lacks an assertion pair")
            input_effects.append({"family_id": family, "finding_id": finding, "M_absent": margins["absent"],
                                  "M_present": margins["present"], "I": finite(margins["present"] - margins["absent"])})
        directional, self_rows, donor_pairs = [], [], defaultdict(dict)
        for case_id, case in sorted(all_cases.items()):
            recipient, donor = sources[case["recipient_id"]], sources[case["donor_id"]]
            original, patched = baseline[(model, recipient["id"])], patched_margins[(model, case_id)]
            effect = finite(patched - original)
            if case_id in self_cases:
                self_rows.append({"model": model, "source_id": recipient["id"], "effect": effect, "absolute_effect": abs(effect)})
                continue
            direction = ("A" if recipient["assertion"] == "absent" else "P") + ("A" if donor["assertion"] == "absent" else "P")
            donor_preference = preference(patched) == donor["assertion"]
            baseline_correct = preference(original) == recipient["assertion"]
            cross = direction in ("AP", "PA")
            directional.append({"case_id": case_id, "family_id": recipient["family_id"], "recipient_id": recipient["id"],
                                "donor_finding_id": donor["finding_id"], "direction": direction, "baseline_margin": original,
                                "patched_margin": patched, "effect": effect, "absolute_effect": abs(effect),
                                "baseline_preference": preference(original), "baseline_correct": baseline_correct,
                                "post_preference": preference(patched), "post_tie": patched == 0,
                                "donor_consistent": donor_preference, "flip": cross and baseline_correct and donor_preference})
            donor_pairs[(recipient["id"], donor["finding_id"])][donor["assertion"]] = patched
        contrasts = []
        for (recipient_id, donor_finding), margins in sorted(donor_pairs.items()):
            require(set(margins) == {"absent", "present"}, "Primary P lacks a fixed donor-polarity pair")
            recipient = sources[recipient_id]
            contrasts.append({"recipient_id": recipient_id, "donor_finding_id": donor_finding, "family_id": recipient["family_id"],
                              "recipient_finding_id": recipient["finding_id"], "recipient_assertion": recipient["assertion"],
                              "M_from_absent": margins["absent"], "M_from_present": margins["present"],
                              "P": finite(margins["present"] - margins["absent"])})
        direction_groups = []
        for family in sorted({r["family_id"] for r in directional}):
            for direction in ("AP", "PA", "AA", "PP"):
                rows = [r for r in directional if r["family_id"] == family and r["direction"] == direction]
                require(bool(rows), "Incomplete directional family cell")
                direction_groups.append({"family_id": family, "direction": direction, "count": len(rows),
                                         "mean_effect": mean(r["effect"] for r in rows),
                                         "mean_absolute_effect": mean(r["absolute_effect"] for r in rows),
                                         **{field + "_count": sum(r[field] for r in rows)
                                            for field in ("baseline_correct", "donor_consistent", "post_tie", "flip")}})
        direction_aggregates = {direction: {"signed": equal_family([r for r in directional if r["direction"] == direction], "effect"),
                                            "absolute": equal_family([r for r in directional if r["direction"] == direction], "absolute_effect")}
                                for direction in ("AP", "PA", "AA", "PP")} if directional else {}
        cross_pairs = defaultdict(list)
        for row in directional:
            if row["direction"] in ("AP", "PA"):
                recipient = sources[row["recipient_id"]]
                cross_pairs[(row["family_id"], recipient["finding_id"], row["donor_finding_id"])].append(row)
        joint = []
        for (family, finding, donor_finding), rows in sorted(cross_pairs.items()):
            require(len(rows) == 2 and {r["direction"] for r in rows} == {"AP", "PA"}, "Incomplete bidirectional comparison")
            joint.append({"family_id": family, "recipient_finding_id": finding, "donor_finding_id": donor_finding,
                          "both_cross_directions_flip": all(r["flip"] for r in rows)})
        result = {"input_effects": input_effects, "I": equal_family(input_effects, "I"),
                  "primary_contrasts": contrasts, "P": equal_family(contrasts, "P") if contrasts else None,
                  "directional_cases": directional, "directional_families": direction_groups,
                  "directional_aggregates": direction_aggregates, "bidirectional_flip_pairs": joint,
                  "self_effects": self_rows}
        if stage == "substantive":
            result["substantive_gate"] = substantive_gate(input_effects)
        if stage == "pilot":
            result["numerical_input_gate"] = {"status": "PASS" if all(r["I"] > 1e-6 for r in input_effects) else "FAIL",
                                              "all_I": [r["I"] for r in input_effects], "threshold_strictly_greater_than": 1e-6}
        report["models"][model] = result
    frozen, lora = [report["models"][model] for model in MODELS]
    report["paired_lora_minus_frozen"] = {}
    for table, field, keys in (("input_effects", "I", ("family_id", "finding_id")),
                               ("primary_contrasts", "P", ("recipient_id", "donor_finding_id"))):
        comparisons = [{**{k: b[k] for k in keys}, "family_id": b["family_id"], "difference": b[field] - a[field]}
                       for a, b in zip(frozen[table], lora[table])]
        report["paired_lora_minus_frozen"][field] = {"cases": comparisons,
                "aggregate": equal_family(comparisons, "difference") if comparisons else None}
    if stage == "substantive":
        report["gate"] = {"status": "PASS" if all(report["models"][m]["substantive_gate"]["status"] == "PASS" for m in MODELS) else "FAIL",
                          "meaning": "Small fixed development readout only; failure diagnoses this comparison, not absence of a negation representation."}
    elif stage == "pilot":
        all_self = [row for model in MODELS for row in report["models"][model]["self_effects"]]
        mechanical = mechanical_gate(data.get("mechanical_evidence"), all_self, recipient_ids, baseline)
        numerical = [report["models"][m]["numerical_input_gate"]["status"] for m in MODELS]
        status = "FAIL" if "FAIL" in numerical or mechanical["status"] == "FAIL" else mechanical["status"]
        report["gate"] = {"status": status, "mechanical": mechanical,
                          "meaning": "Numerical/mechanical feasibility only; no positive intervention effect is required for coordinate e0."}
    else:
        report["gate"] = {"status": "NOT_APPLICABLE", "meaning": "Descriptive saved-score arithmetic; not an execution or fitting approval."}
    return report


def select_snapshots(report32, report64):
    """Selection arithmetic only; returns a receipt, does not load/save a basis."""
    for report, step in ((report32, 32), (report64, 64)):
        require(report["stage"] == "selection" and report["basis_condition"] == "learned_true"
                and report["snapshot_step"] == step, "Selection may use only learned-true snapshots32/64")
    for field in ("protocol_sha256", "preflight_sha256", "checkpoint_ids", "data_kind"):
        require(report32[field] == report64[field], "Selection provenance mismatch")
    choices = {}
    for model in MODELS:
        p32, p64 = [finite(r["models"][model]["P"]["equal_family_mean"]) for r in (report32, report64)]
        step = 64 if p64 - p32 > 1e-6 else 32
        chosen = report64 if step == 64 else report32
        choices[model] = {"step": step, "basis_id": chosen["basis_ids"][model], "P32": p32, "P64": p64}
    return {"choices": choices, "rule": "Greater equal-family P; ties within1e-6 select32",
            "selection_report_sha256": {"32": digest(report32), "64": digest(report64)}, "fitting_authorised": False}


def prefit_gate(pilot, substantive):
    require(pilot["stage"] == "pilot" and substantive["stage"] == "substantive", "Require pilot and substantive reports")
    for field in ("protocol_sha256", "preflight_sha256", "checkpoint_ids", "data_kind"):
        require(pilot[field] == substantive[field], "Pre-fit gate provenance mismatch")
    states = [r["gate"]["status"] for r in (pilot, substantive)]
    return {"status": "FAIL" if "FAIL" in states else "PASS" if states == ["PASS", "PASS"] else "UNAVAILABLE",
            "pilot_sha256": digest(pilot), "substantive_sha256": digest(substantive), "fitting_authorised": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "preflight", "scores", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists() and not args.output.resolve().is_relative_to(args.manifest.resolve().parent)
            and args.output.resolve() not in {args.preflight.resolve(), args.scores.resolve()}, "Choose a new output outside protocol/input files")
    inputs = [json.loads(path.read_text(encoding="utf-8")) for path in (args.manifest, args.preflight, args.scores)]
    report = summarise(*inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "stage": report["stage"], "data_kind": report["data_kind"], "gate": report["gate"]["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
