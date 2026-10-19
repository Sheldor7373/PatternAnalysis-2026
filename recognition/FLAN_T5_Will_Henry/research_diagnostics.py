"""Read saved research artefacts without importing torch or changing a run.

Example: python research_diagnostics.py --run runs/lora --result runs/lora-dev
    --output runs/review-001
The output is a point-in-time review, not a training controller or clinical score.
Use --supervisor <directory> to attach saved supervisor.json/gpu.jsonl to a run.
Health collection does not load a model, generate predictions or score text.
Optional --analyse-predictions scores complete saved prediction files on CPU.
Loss windows are consumed as recorded by train.py, never recovered by taking
differences of cumulative means. Memory fields use MiB, independently of rates.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata

ROUGE = ("rouge1", "rouge2", "rougeL", "rougeLsum")


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class Reader:
    def __init__(self):
        self.sources = {}
        self.warnings = []

    def read(self, path, lines=False):
        path = Path(path).resolve()
        if not path.exists():
            return [] if lines else {}
        data = path.read_bytes()
        self.sources[str(path)] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        if not lines:
            value = json.loads(data)
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object: {path}")
            return value
        records = []
        chunks = data.splitlines(keepends=True)
        for index, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            try:
                record = json.loads(chunk)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if index == len(chunks) - 1 and not chunk.endswith((b"\n", b"\r")):
                    self.warnings.append(f"Ignored incomplete final JSONL record: {path}")
                    break
                raise ValueError(f"Invalid JSONL record {index + 1}: {path}")
            if not isinstance(record, dict):
                raise ValueError(f"Expected object at {path}:{index + 1}")
            records.append(record)
        return records


def checkpoint_metadata(run, alias, reader):
    pointer = reader.read(run / f"{alias}.json")
    if not pointer:
        return {}
    directory = (run / pointer["directory"]).resolve()
    if not directory.is_relative_to(run):
        raise ValueError("Checkpoint pointer leaves its run directory")
    # Read only small metadata. Training's resolver validates all weight hashes.
    marker = reader.read(directory / "COMPLETE.json")
    metadata = reader.read(directory / "experiment.json")
    return {"directory": str(directory), "completion_marker_present": bool(marker),
            "weights_verified_by_diagnostics": False, "metadata": metadata}


def run_summary(path, reader, supervisor=None):
    path = Path(path).resolve()
    settings = reader.read(path / "run.json")
    history = reader.read(path / "history.jsonl", lines=True)
    supervisor_path = Path(supervisor).resolve() if supervisor is not None else None
    telemetry = reader.read(supervisor_path / "gpu.jsonl" if supervisor_path else path / "gpu_telemetry.jsonl", lines=True)
    supervision = reader.read(supervisor_path / "supervisor.json") if supervisor_path else {}
    final = reader.read(path / "training_metrics.json")
    status = reader.read(path / "status.json")
    if not settings:
        reader.warnings.append(f"Missing run.json: {path}")
    last = history[-1] if history else {}
    latest = {key: next((row[key] for row in reversed(history) if finite(row.get(key))), None)
              for key in ("window_train_loss", "window_target_tokens", "window_optimizer_steps", "window_seconds",
                          "learning_rate", "next_learning_rate", "grad_norm_preclip", "window_max_grad_norm_preclip",
                          "window_target_tokens_per_second", "peak_allocated_mib", "peak_reserved_mib")}
    for key in ("peak_allocated_mib", "peak_reserved_mib"):
        if finite(final.get(key)):
            latest[key] = final[key]
    # Diagnostic dev losses are plotted, but never treated as selection decisions.
    selected = [row for row in history if row.get("selection_decision") is True and finite(row.get("dev_loss"))]
    device = [row["used_mib"] for row in telemetry if finite(row.get("used_mib"))]
    return {"path": str(path), "settings": settings, "status": status, "final_metrics": final,
            "supervisor_path": str(supervisor_path) if supervisor_path else None, "supervision": supervision,
            "history": history, "gpu_telemetry": telemetry,
            "checkpoints": {name: checkpoint_metadata(path, name, reader) for name in ("latest", "best")},
            "summary": {"mode": settings.get("mode"), "seed": settings.get("arguments", {}).get("seed"),
                        "model_revision": settings.get("revision"),
                        "git_commit": settings.get("environment", {}).get("git_commit"),
                        "step": last.get("step"), "epoch": last.get("epoch"),
                        "elapsed_seconds": last.get("elapsed_seconds"), "train_loss": last.get("train_loss"),
                        "train_loss_scope": last.get("train_loss_scope"), "epoch_progress": last.get("epoch_progress"),
                        "reported_nonfinite_loss_or_gradients": any(row.get("finite_loss") is False
                                                                   or row.get("finite_gradients") is False for row in history),
                        "latest_dev_loss": next((row["dev_loss"] for row in reversed(history)
                                                 if finite(row.get("dev_loss"))), None),
                        "best_selection_dev_loss": min((row["dev_loss"] for row in selected), default=None),
                        "sampled_device_peak_used_mib": max(device, default=None),
                        "stop_reason": final.get("stop_reason"), "history_records": len(history), **latest,
                        "supervisor_state": supervision.get("state"),
                        "checkpoint_age_seconds": supervision.get("checkpoint_age_seconds"),
                        **{key: final.get(key, settings.get(key)) for key in
                           ("total_parameters", "trainable_parameters")}}}


def result_summary(path, reader):
    path = Path(path).resolve()
    if path.is_dir():
        path = path / "metrics.json"
    metrics = reader.read(path)
    if not metrics:
        raise ValueError(f"Missing or empty prediction metrics: {path}")
    return {"path": str(path), "metrics": metrics}


def compare_results(results):
    """Require explicit matching evaluation identity/settings before comparison.

    A complete manifest may be development data; its label is never promoted to
    final official evaluation. Missing identity does not count as a match.
    """
    issues = []
    required = ("manifest_sha256", "selected_ids_sha256", "examples", "generation", "evaluation_scope")
    if len(results) < 2:
        issues.append("Need at least two result files for a comparison")
    modes = [r["metrics"].get("model", {}).get("mode") for r in results]
    if any(mode not in ("frozen", "full", "lora") for mode in modes) or len(set(modes)) != len(modes):
        issues.append("Modes must be distinct frozen/full/lora results")
    for key in required:
        values = [r["metrics"].get(key) for r in results]
        if any(value is None or value == {} for value in values):
            issues.append(f"Missing comparison identity/settings: {key}")
        elif any(value != values[0] for value in values[1:]):
            issues.append(f"Different {key}")
    for key in ("model", "revision"):
        values = [r["metrics"].get("model", {}).get(key) for r in results]
        if any(value is None for value in values) or any(value != values[0] for value in values[1:]):
            issues.append(f"Missing or different base model {key}")
    cost_issues = []
    for key in ("batch_size", "precision"):
        values = [r["metrics"].get(key) for r in results]
        if any(value is None for value in values) or any(value != values[0] for value in values[1:]):
            cost_issues.append(f"Missing or different {key}")
    devices = [r["metrics"].get("environment", {}).get("gpu") for r in results]
    if any(value is None for value in devices) or any(value != devices[0] for value in devices[1:]):
        cost_issues.append("Missing or different GPU identity; timing comparability unverified")
    rows = []
    for result in results:
        metrics = result["metrics"]
        row = {"path": result["path"], "mode": metrics.get("model", {}).get("mode"),
               "evaluation_scope": metrics.get("evaluation_scope"), "examples": metrics.get("examples")}
        row.update({key: metrics.get("rouge", {}).get(key) for key in ("rouge1", "rouge2", "rougeL", "rougeLsum")})
        row.update({key: metrics.get(key) for key in ("total_parameters", "trainable_parameters", "tokens_per_second",
                                                    "peak_allocated_mib", "peak_reserved_mib")})
        rows.append(row)
    return {"quality_settings_match": not issues, "comparison_issues": issues,
            "inference_cost_settings_match": not issues and not cost_issues, "cost_issues": cost_issues,
            "all_three_modes_present": set(modes) == {"frozen", "full", "lora"}, "rows": rows}


def build_report(runs=(), results=(), supervisors=()):
    reader = Reader()
    runs = list(runs)
    if supervisors and len(supervisors) != len(runs):
        raise ValueError("Supply one supervisor directory per run, in the same order")
    run_data = [run_summary(path, reader, supervisor)
                for path, supervisor in zip(runs, supervisors or [None] * len(runs))]
    result_data = [result_summary(path, reader) for path in results]
    return {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "runs": run_data, "results": result_data, "comparison": compare_results(result_data),
            "sources": reader.sources, "warnings": reader.warnings,
            "limitations": ["Files are read independently; an active run may advance during collection.",
                            "Sampled device memory includes other processes and can miss peaks.",
                            "Missing measurements remain null; no patient-independence or clinical-safety claim."]}


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def canonical(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def align_predictions(predictions):
    """Align by original row ID, retaining repeated sources/references as rows."""
    aligned = {}
    for mode, rows in predictions.items():
        mapping = {}
        for row in rows:
            if not all(isinstance(row.get(key), str) for key in ("id", "input", "target", "prediction")):
                raise ValueError(f"Missing prediction fields in {mode}")
            if row["id"] in mapping:
                raise ValueError(f"Duplicate prediction ID in {mode}: {row['id']}")
            mapping[row["id"]] = row
        aligned[mode] = mapping
    if len(aligned) < 2:
        raise ValueError("Paired analysis needs at least two modes")
    modes = list(aligned)
    first = aligned[modes[0]]
    if not first or any(set(mapping) != set(first) for mapping in aligned.values()):
        raise ValueError("Prediction identities differ or are empty")
    for row_id, row in first.items():
        for mode in modes[1:]:
            other = aligned[mode][row_id]
            if any(row.get(key) != other.get(key) for key in ("input", "target", "image", "source")):
                raise ValueError(f"Source/reference/exposure differs for {row_id}")
        visible = [aligned[mode][row_id]["visible_input"] for mode in modes
                   if aligned[mode][row_id].get("visible_input") is not None]
        if visible and any(value != visible[0] for value in visible[1:]):
            raise ValueError(f"Recorded model-visible source differs for {row_id}")
    return aligned


def report_groups(rows):
    """Report/image components; no claim that these are independent patients."""
    parent = list(range(len(rows)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen = {}
    for index, row in enumerate(rows):
        keys = [("report", canonical(row["input"]))]
        if isinstance(row.get("image"), str) and row["image"].strip():
            keys.append(("image", row["image"]))
        for key in keys:
            if key in seen:
                parent[find(index)] = find(seen[key])
            else:
                seen[key] = index
    groups = {}
    for index in range(len(rows)):
        groups.setdefault(find(index), []).append(index)
    return list(groups.values())


def bootstrap_differences(scores, groups, draws=1000, seed=3710):
    """Paired cluster bootstrap; shared draws for every mode/metric difference."""
    import numpy as np
    if draws < 2 or not groups or any(not group for group in groups):
        raise ValueError("Need at least two bootstrap draws and nonempty groups")
    modes = list(scores)
    lengths = {len(values) for values in scores.values()}
    covered = [index for group in groups for index in group]
    if len(lengths) != 1 or sorted(covered) != list(range(next(iter(lengths)))):
        raise ValueError("Groups must partition paired rows exactly once")
    pairs = [(a, b) for i, a in enumerate(modes) for b in modes[i + 1:]]
    columns = [(a, b, metric) for a, b in pairs for metric in ROUGE]
    differences = np.array([[scores[b][i][metric] - scores[a][i][metric]
                             for a, b, metric in columns] for i in range(len(covered))])
    sums = np.array([differences[group].sum(axis=0) for group in groups])
    sizes = np.array([len(group) for group in groups])
    available = len(groups) >= 2
    if available:
        rng = np.random.default_rng(seed)
        estimates = []
        for _ in range(draws):
            counts = np.bincount(rng.integers(len(groups), size=len(groups)), minlength=len(groups))
            estimates.append((counts @ sums) / (counts @ sizes))
        low, high = np.quantile(estimates, [0.025, 0.975], axis=0)
    else:
        low = high = [None] * len(columns)
    means = differences.mean(axis=0)
    return {"method": "paired report/image-component percentile bootstrap, row-weighted means",
            "uncertainty": "Evaluation-sample uncertainty only; not training/random-seed uncertainty",
            "seed": seed, "draws": draws if available else 0, "requested_draws": draws,
            "groups": len(groups), "rows": len(covered), "uncertainty_available": available,
            "uncertainty_unavailable_reason": None if available else "Fewer than two observable report/image components",
            "limitation": "Components are observable report/image clusters, not verified independent patients.",
            "intervals": [{"difference": f"{b} minus {a}", "metric": key, "mean": float(means[i]),
                           "lower_95": float(low[i]) if available else None,
                           "upper_95": float(high[i]) if available else None}
                          for i, (a, b, key) in enumerate(columns)]}


def length_bin(words):
    return "0-50 words" if words <= 50 else ("51-100 words" if words <= 100 else ">100 words")


def analyse_predictions(predictions, score=None, draws=1000, seed=3710):
    from collections import Counter
    from metrics import safety_flags, sentences
    aligned = align_predictions(predictions)
    modes = list(aligned)
    ids = sorted(aligned[modes[0]])
    base = [aligned[modes[0]][row_id] for row_id in ids]
    if score is None:
        from rouge_score.rouge_scorer import RougeScorer
        scorer = RougeScorer(list(ROUGE), use_stemmer=True, split_summaries=False)

        def score(row):
            values = scorer.score("\n".join(sentences(row["target"])),
                                  "\n".join(sentences(row["prediction"])))
            return {key: values[key].fmeasure for key in ROUGE}
    scores = {mode: [score(aligned[mode][row_id]) for row_id in ids] for mode in modes}
    if any(not finite(row.get(key)) or not 0 <= row[key] <= 1
           for values in scores.values() for row in values for key in ROUGE):
        raise ValueError("Per-example ROUGE must contain four finite F1 scores in [0, 1]")
    per_example, audit = [], []
    for mode in modes:
        repeats = Counter(canonical(row["prediction"]) for row in aligned[mode].values() if row["prediction"].strip())
        for index, row_id in enumerate(ids):
            row = aligned[mode][row_id]
            flags = safety_flags(row["input"], row["prediction"])
            entry = {"id": row_id, "mode": mode, "source": row.get("source"), **scores[mode][index],
                     "input_characters": len(row["input"]), "target_characters": len(row["target"]),
                     "generated_characters": len(row["prediction"]),
                     "input_words": len(row["input"].split()), "target_words": len(row["target"].split()),
                     "generated_words": len(row["prediction"].split()),
                     "empty_output": not row["prediction"].strip(),
                     "repeated_output_count": repeats[canonical(row["prediction"])] if row["prediction"].strip() else 0,
                     "regex_candidate": any(flags.values())}
            for field in ("source_tokens", "target_tokens", "generated_tokens", "input_truncated", "output_has_eos"):
                entry[field] = row.get(field)
            entry["source_length_bin"] = length_bin(entry["input_words"])
            entry["target_length_bin"] = length_bin(entry["target_words"])
            per_example.append(entry)
    # Candidate selection is reproducible and explicitly failure-selected, never
    # a representative error-rate estimate or an automatically confirmed failure.
    baseline = "frozen" if "frozen" in modes else modes[0]
    for index, row_id in enumerate(ids):
        row = base[index]
        flags = {mode: safety_flags(row["input"], aligned[mode][row_id]["prediction"]) for mode in modes}
        regression = max(scores[baseline][index]["rougeL"] - scores[mode][index]["rougeL"] for mode in modes)
        candidate = any(any(value.values()) for value in flags.values())
        audit.append({"id": row_id, "selection": "failure-selected candidate; human confirmation required",
                      "selection_regex_candidate": candidate, "selection_max_rougeL_regression": regression,
                      "full_source": row["input"], "full_reference": row["target"],
                      "visible_source": {mode: aligned[mode][row_id].get("visible_input") for mode in modes},
                      "visible_source_note": "Null means exact model-visible text was not recorded; prompt inclusion follows producer metadata.",
                      "outputs": {mode: aligned[mode][row_id]["prediction"] for mode in modes},
                      "regex_flags": flags,
                      "manual_review": {mode: {key: "" for key in ("unsupported_claims", "omissions", "negation",
                                                                   "uncertainty", "readability", "confirmed_failure")}
                                        for mode in modes}})
    audit.sort(key=lambda row: (-row["selection_regex_candidate"], -row["selection_max_rougeL_regression"], row["id"]))
    strata = []
    for mode in modes:
        rows = [row for row in per_example if row["mode"] == mode]
        for field in ("source", "source_length_bin", "target_length_bin"):
            for value in sorted({str(row[field]) for row in rows}):
                group = [row for row in rows if str(row[field]) == value]
                strata.append({"mode": mode, "stratum": field, "value": value, "rows": len(group),
                               **{key: sum(row[key] for row in group) / len(group) for key in ROUGE}})
    return {"per_example": per_example, "strata": strata, "manual_audit_candidates": audit[:5],
            "all_model_visible_sources_recorded": all(isinstance(row.get("visible_input"), str)
                                                       for mapping in aligned.values() for row in mapping.values()),
            "selection_criteria": "Up to five: regex candidates first, then largest ROUGE-L regression versus frozen; ID breaks ties. Failure-selected, not representative.",
            "bootstrap": bootstrap_differences(scores, report_groups(base), draws, seed),
            "length_units": "Characters and whitespace words; token counts/visibility/EOS only when explicitly recorded by producer.",
            "regex_limitation": "Regex flags are manual-review aids, not clinical accuracy or confirmed errors."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=[])
    parser.add_argument("--result", action="append", default=[])
    parser.add_argument("--supervisor", action="append", default=[],
                        help="Optional supervisor directory per --run, in matching order (gpu.jsonl/supervisor.json)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--analyse-predictions", action="store_true",
                        help="Optional CPU analysis of predictions.jsonl beside matching result metrics; never generates outputs")
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=3710)
    args = parser.parse_args()
    if not args.run and not args.result:
        parser.error("Supply at least one --run or --result")
    output = Path(args.output).resolve()
    for source in args.run + args.result + args.supervisor:
        source = Path(source).resolve()
        root = source if source.is_dir() else source.parent
        if output.is_relative_to(root):
            parser.error("Review output must be outside source run/result directories")
    report = build_report(args.run, args.result, args.supervisor)
    if args.analyse_predictions:
        if not report["comparison"]["quality_settings_match"]:
            parser.error("Paired prediction analysis requires matching result identities and generation settings")
        reader = Reader()
        predictions = {}
        for result in report["results"]:
            metrics = result["metrics"]
            rows = reader.read(Path(result["path"]).parent / "predictions.jsonl", lines=True)
            ids_digest = hashlib.sha256("\n".join(row["id"] for row in rows).encode()).hexdigest()
            if len(rows) != metrics["examples"] or ids_digest != metrics["selected_ids_sha256"]:
                parser.error("Prediction rows do not match saved metrics identity/count")
            predictions[metrics["model"]["mode"]] = rows
        if reader.warnings:
            parser.error("Prediction analysis requires complete JSONL files")
        report["prediction_analysis"] = analyse_predictions(predictions, draws=args.bootstrap_draws, seed=args.bootstrap_seed)
        for result in report["results"]:
            metrics = result["metrics"]
            mode_rows = [row for row in report["prediction_analysis"]["per_example"]
                         if row["mode"] == metrics["model"]["mode"]]
            for row in mode_rows:
                row.update({"metrics_path": result["path"],
                            "prediction_path": str(Path(result["path"]).parent / "predictions.jsonl"),
                            "evaluation_scope": metrics["evaluation_scope"],
                            "model_revision": metrics["model"].get("revision"),
                            "generation_seed": metrics["generation"].get("seed")})
            for key in ROUGE:
                recorded = metrics.get("rouge", {}).get(key)
                actual = sum(row[key] for row in mode_rows) / len(mode_rows)
                if not finite(recorded) or not math.isclose(actual, recorded, abs_tol=1e-8):
                    parser.error(f"Recomputed {key} disagrees with saved metrics: {result['path']}")
        report["sources"].update(reader.sources)
    output.mkdir(parents=True, exist_ok=True)
    (output / "review.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    write_csv(output / "status.csv", [{"path": row["path"], **row["summary"]} for row in report["runs"]])
    write_csv(output / "comparison.csv", report["comparison"]["rows"])
    if "prediction_analysis" in report:
        analysis = report["prediction_analysis"]
        write_csv(output / "per-example.csv", analysis["per_example"])
        write_csv(output / "strata.csv", analysis["strata"])
        (output / "manual-audit.jsonl").write_text("".join(json.dumps(row) + "\n" for row in analysis["manual_audit_candidates"]), encoding="utf-8")
    print(json.dumps({"output": str(output), "warnings": report["warnings"],
                      "comparison_issues": report["comparison"]["comparison_issues"]}, indent=2))


if __name__ == "__main__":
    main()
