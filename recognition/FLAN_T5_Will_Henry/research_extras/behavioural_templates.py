"""Prepare synthetic CheckList-inspired fixtures, never model predictions.

Forty hand-authored template families; all slot/contrast/invariance variants stay
in their family's fixed partition. Expected propositions require human review.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import unicodedata

PAPER = "https://aclanthology.org/2020.acl-main.442/"
FINDINGS = ("pleural effusion", "pneumothorax", "focal opacity")

# Each pair is a targeted semantic contrast, not an exact-output reference.
# The first seven families per capability are development; the final three
# are held out. Splits are frozen explicitly, never assigned per generated row.
NEGATION = (
    ("No {finding}.", "{finding} is present."),
    ("There is no evidence of {finding}.", "There is evidence of {finding}."),
    ("{finding} is absent.", "{finding} is visible."),
    ("The examination is negative for {finding}.", "The examination is positive for {finding}."),
    ("Imaging shows no {finding}.", "Imaging shows {finding}."),
    ("No {finding} is identified on this examination.", "{finding} is identified on this examination."),
    ("There is neither {finding} nor cardiac enlargement.", "There is {finding} but no cardiac enlargement."),
    ("Without evidence of {finding} on the current image.", "With evidence of {finding} on the current image."),
    ("Cardiac enlargement is present, but {finding} is not seen.", "Cardiac enlargement is present, and {finding} is seen."),
    ("{finding}: not detected.", "{finding}: detected."),
)
UNCERTAINTY = (
    ("Possible {finding}.", "{finding} is present."),
    ("{finding} may be present.", "{finding} is present."),
    ("Cannot exclude {finding}.", "{finding} is identified."),
    ("Findings are suggestive of {finding}.", "Findings demonstrate {finding}."),
    ("Suspected {finding}.", "Definite {finding}."),
    ("There could be {finding}.", "There is {finding}."),
    ("{finding} is considered likely.", "{finding} is confirmed on this examination."),
    ("It is unclear whether {finding} is present.", "It is clear that {finding} is present."),
    ("The appearance raises the possibility of {finding}.", "The appearance establishes {finding}."),
    ("{finding} cannot be ruled out on this image.", "{finding} is seen on this image."),
)
LATERALITY = (
    "{side} {finding} is present.",
    "There is {finding} on the {side} side.",
    "The {side} side shows {finding}.",
    "{finding} is confined to the {side} side.",
    "A finding of {finding} is identified on the {side}.",
    "On the {side}, {finding} is visible.",
    "{finding} involves only the {side} side.",
    "{finding} affects the {side} side; the opposite side is clear.",
    "Site: {side} side. Finding: {finding}.",
    "Unilateral {finding}, located on the {side}.",
)
TEMPORAL = (
    # label: (current assertion, prior assertion, direction)
    (("{finding} has increased since the previous image.", "{finding} has decreased since the previous image."),
     (("present", "present", "increased"), ("present", "present", "decreased"))),
    (("{finding} is unchanged from the previous study.", "{finding} has increased from the previous study."),
     (("present", "present", "stable"), ("present", "present", "increased"))),
    (("Previously seen {finding} has resolved.", "Previously seen {finding} persists."),
     (("absent", "present", "resolved"), ("present", "present", "persistent"))),
    (("New {finding} compared with the prior examination.", "{finding} was present previously and remains present."),
     (("present", "absent", "new"), ("present", "present", "persistent"))),
    (("{finding} is improving but remains visible.", "{finding} is worsening and remains visible."),
     (("present", "present", "decreased"), ("present", "present", "increased"))),
    (("Compared with yesterday, {finding} is smaller.", "Compared with yesterday, {finding} is larger."),
     (("present", "present", "decreased"), ("present", "present", "increased"))),
    (("{finding} is no longer visible after being seen previously.", "{finding} is still visible after being seen previously."),
     (("absent", "present", "resolved"), ("present", "present", "persistent"))),
    (("Earlier imaging showed no {finding}; it is present now.", "Earlier imaging showed {finding}; it is absent now."),
     (("present", "absent", "new"), ("absent", "present", "resolved"))),
    (("The amount of {finding} is stable relative to the earlier image.", "The amount of {finding} has reduced relative to the earlier image."),
     (("present", "present", "stable"), ("present", "present", "decreased"))),
    (("{finding} has recurred after an interval when it was absent.", "{finding} remains absent after previously resolving."),
     (("present", "absent", "recurrent"), ("absent", "absent", "remains_absent"))),
)


def families():
    result = []
    for capability, templates in (("negation", NEGATION), ("uncertainty", UNCERTAINTY),
                                  ("laterality", LATERALITY), ("temporal", TEMPORAL)):
        for index, entry in enumerate(templates, 1):
            if capability == "negation":
                pair, labels = entry, ({"assertion": "absent"}, {"assertion": "present"})
            elif capability == "uncertainty":
                pair, labels = entry, ({"assertion": "uncertain"}, {"assertion": "present"})
            elif capability == "laterality":
                pair = (entry.replace("{side}", "left"), entry.replace("{side}", "right"))
                labels = ({"assertion": "present", "laterality": "left"},
                          {"assertion": "present", "laterality": "right"})
            else:
                pair, states = entry
                labels = tuple({"assertion": current, "prior_assertion": prior, "change": change}
                               for current, prior, change in states)
            # Scope controls are preserved as secondary propositions too.
            labels = [dict(label) for label in labels]
            if capability == "negation" and index == 7:
                for label in labels:
                    label["secondary"] = {"concept": "cardiac enlargement", "assertion": "absent"}
            if capability == "negation" and index == 9:
                for label in labels:
                    label["secondary"] = {"concept": "cardiac enlargement", "assertion": "present"}
            if capability == "laterality" and index in (4, 7, 8, 10):
                for label in labels:
                    label["opposite_laterality"] = "right" if label["laterality"] == "left" else "left"
                    label["opposite_finding_assertion"] = "absent"
            result.append({"family_id": f"{capability}-{index:02d}", "capability": capability,
                           "split": "development" if index <= 7 else "heldout",
                           "templates": list(pair), "expected": labels,
                           "label_status": "author_proposed; human review pending"})
    return result


def make_cases(specs=None):
    cases = []
    for family in families() if specs is None else specs:
        for finding_index, finding in enumerate(FINDINGS, 1):
            pair_id = f"{family['family_id']}-f{finding_index}"
            for contrast, (template, expected) in enumerate(zip(family["templates"], family["expected"])):
                text = template.format(finding=finding)
                text = text[0].upper() + text[1:]
                base_id = f"{pair_id}-c{contrast}"
                for variant, source in (("base", text), ("whitespace", "  " + text.replace(" ", "  ") + "\n")):
                    # Whitespace perturbation is the only automatic invariance;
                    # no synonym or clinical-equivalence assumption is made.
                    cases.append({"id": f"{base_id}-{variant}", "family_id": family["family_id"],
                                  "capability": family["capability"], "split": family["split"],
                                  "source_kind": "synthetic; no patient records", "input": source,
                                  "finding": finding, "contrast": contrast,
                                  "expected_proposition": {"concept": finding, **expected},
                                  "label_status": family["label_status"],
                                  "minimum_functionality": "Preserve the proposed proposition in a plain-English rewrite.",
                                  "directional_pair_id": pair_id,
                                  "directional_expectation": f"Changing contrast must preserve the changed {family['capability']} meaning, not identical wording.",
                                  "invariance_group_id": base_id, "variant": variant,
                                  "invariance_expectation": "Whitespace changes must preserve meaning; exact output text may differ.",
                                  "human_review": {"label_approved": "", "reviewer": "", "notes": ""}})
    return cases


def canonical_source(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def validate(specs, cases):
    if len(specs) != 40 or len({f["family_id"] for f in specs}) != 40:
        raise ValueError("Expected exactly forty distinct template families")
    if Counter(f["capability"] for f in specs) != {key: 10 for key in ("negation", "uncertainty", "laterality", "temporal")}:
        raise ValueError("Expected ten families per capability")
    mapping = {f["family_id"]: f for f in specs}
    if len({row["id"] for row in cases}) != len(cases) or len(cases) != 480:
        raise ValueError("Expected 480 uniquely identified fixture cases")
    text_splits = {}
    for row in cases:
        family = mapping[row["family_id"]]
        if row["split"] != family["split"]:
            raise ValueError("Family variants cannot cross partitions")
        if row["source_kind"] != "synthetic; no patient records" or "{" in row["input"]:
            raise ValueError("Non-synthetic or unresolved fixture")
        text_splits.setdefault(canonical_source(row["input"]), set()).add(row["split"])
    if any(len(parts) != 1 for parts in text_splits.values()):
        raise ValueError("Canonical source text cannot cross partitions")
    for key in ("family_id", "directional_pair_id", "invariance_group_id"):
        groups = {}
        for row in cases:
            groups.setdefault(row[key], set()).add(row["split"])
        if any(len(parts) != 1 for parts in groups.values()):
            raise ValueError(f"Partition leakage through {key}")


def write_suite(output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Choose a new output directory; existing suites are immutable")
    specs, cases = families(), make_cases()
    validate(specs, cases)
    output.mkdir(parents=True)
    files = {}
    for name, rows in (("families.jsonl", specs), ("development.jsonl", [r for r in cases if r["split"] == "development"]),
                       ("heldout.jsonl", [r for r in cases if r["split"] == "heldout"])):
        data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode()
        (output / name).write_bytes(data)
        files[name] = {"sha256": hashlib.sha256(data).hexdigest(), "rows": len(rows)}
    # Blank output judgments avoid laundering author labels into model results.
    with (output / "human-review.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("id", "family_id", "split", "capability", "input",
                                                   "expected_proposition", "label_approved", "reviewer", "notes"))
        writer.writeheader()
        for row in cases:
            writer.writerow({**{key: row[key] for key in ("id", "family_id", "split", "capability", "input")},
                             "expected_proposition": json.dumps(row["expected_proposition"], sort_keys=True),
                             "label_approved": "", "reviewer": "", "notes": ""})
    review_data = (output / "human-review.csv").read_bytes()
    files["human-review.csv"] = {"sha256": hashlib.sha256(review_data).hexdigest(), "rows": len(cases)}
    manifest = {"schema_version": 1, "source": PAPER, "scope": "synthetic fixtures only; no model outputs",
                "family_count": 40, "family_split": {"development": 28, "heldout": 12},
                "case_count": len(cases), "case_split": dict(Counter(r["split"] for r in cases)),
                "test_types": ["MFT: proposition preservation", "INV: whitespace", "DIR: semantic contrast"],
                "label_status": "author_proposed; human review pending", "files": files,
                "limitations": ["Heldout means withheld from tuning, not absent from pretraining.",
                                "Shared concepts and related linguistic constructions remain across families.",
                                "Forty families are a small designed test suite, not a clinical population sample.",
                                "Human labels require approval before interpreting model failures.",
                                "Do not score clinical correctness with exact text equality or the project's limited regex flags.",
                                "Official validation remains unopened; no inference or training is performed."]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = write_suite(args.output)
    print(json.dumps({key: manifest[key] for key in ("scope", "family_count", "family_split", "case_count", "case_split")}))


if __name__ == "__main__":
    main()
