"""Render saved review.json with the existing matplotlib dependency; no GPU use."""

import argparse
from datetime import datetime
import json
from pathlib import Path

from research_diagnostics import finite


def telemetry_series(samples, field):
    rows = [row for row in samples if finite(row.get(field))]
    if any(finite(row.get("elapsed_seconds")) for row in rows):
        return [(row["elapsed_seconds"] / 60, row[field]) for row in rows if finite(row.get("elapsed_seconds"))]
    timed = []
    for row in rows:
        timestamp = row.get("time")
        if not finite(timestamp) and isinstance(row.get("timestamp_utc"), str):
            try:
                parsed = datetime.fromisoformat(row["timestamp_utc"].replace("Z", "+00:00"))
                timestamp = parsed.timestamp() if parsed.tzinfo is not None else None
            except ValueError:
                timestamp = None
        if finite(timestamp):
            timed.append((timestamp, row[field]))
    origin = min((point[0] for point in timed), default=0)
    return [((time - origin) / 60, value) for time, value in timed]


def render(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    roots = [Path(source["path"]).resolve() for source in report["runs"]]
    roots += [Path(source["path"]).resolve().parent for source in report.get("results", [])]
    roots += [Path(source["supervisor_path"]).resolve() for source in report["runs"] if source.get("supervisor_path")]
    for root in roots:
        if output.resolve().is_relative_to(root):
            raise ValueError("Plot output must be outside source run/result directories")
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, run in enumerate(report["runs"]):
        fig, axes = plt.subplots(5, 2, figsize=(11, 14), constrained_layout=True)
        panels = [(axes[0, 0], [("window_train_loss", "Nonoverlapping token-weighted window")], "Window training loss"),
                  (axes[0, 1], [("dev_loss", "Development loss (diagnostic or selection)")], "Development loss"),
                  (axes[1, 0], [("train_loss", "Source cumulative/epoch training mean")], "Cumulative training loss"),
                  (axes[1, 1], [("learning_rate", "Used LR"), ("next_learning_rate", "Next LR")], "Learning rate"),
                  (axes[2, 0], [("window_target_tokens_per_second", "Recorded window tokens/s")], "Target tokens/s"),
                  (axes[2, 1], [("window_seconds", "Window duration")], "Window seconds"),
                  (axes[3, 0], [("grad_norm_preclip", "Last step before clipping"),
                               ("window_max_grad_norm_preclip", "Window maximum before clipping")], "Gradient norm"),
                  (axes[3, 1], [("peak_allocated_mib", "Torch allocated peak"),
                               ("peak_reserved_mib", "Torch reserved peak")], "Allocator memory (MiB)")]
        for axis, fields, label in panels:
            found = False
            for field, title in fields:
                points = [(r["elapsed_seconds"] / 60, r[field]) for r in run["history"]
                          if finite(r.get("elapsed_seconds")) and finite(r.get(field))
                          and (not field.startswith("window_") or r.get("event") == "train_window")]
                if points:
                    axis.plot(*zip(*points), label=title, marker=".")
                    found = True
            axis.set(xlabel="Elapsed minutes", ylabel=label)
            axis.grid(alpha=0.25)
            if found:
                axis.legend(fontsize=8)
            else:
                axis.text(0.5, 0.5, "No recorded measurements", ha="center", transform=axis.transAxes)
        samples = run["gpu_telemetry"]
        for axis, field, label in [(axes[4, 0], "used_mib", "Sampled whole-device memory (MiB)"),
                                   (axes[4, 1], "temperature_c", "Sampled temperature (C)")]:
            # Separate telemetry axes avoid pretending its first sample is the
            # training-session origin when only wall-clock timestamps exist.
            points = telemetry_series(samples, field)
            if points:
                axis.plot(*zip(*points), marker=".")
            else:
                axis.text(0.5, 0.5, "No recorded measurements", ha="center", transform=axis.transAxes)
            axis.set(xlabel="Minutes (recorded elapsed or first telemetry sample)", ylabel=label)
            axis.grid(alpha=0.25)
        fig.suptitle(f"{run['path']}\nSaved review; loss/throughput definitions follow source telemetry", fontsize=9)
        path = output / f"run-{index + 1:02d}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path))
    comparison = report["comparison"]
    if comparison["quality_settings_match"]:
        fig, axes = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)
        rows = comparison["rows"]
        for axis, metric in zip(axes.flat, ("rouge1", "rouge2", "rougeL", "rougeLsum")):
            valid = [row for row in rows if finite(row.get(metric))]
            axis.bar([row["mode"] for row in valid], [row[metric] for row in valid])
            axis.set(ylabel=f"{metric} mean F1 (0–1)", ylim=(0, 1))
        fig.suptitle("Matched saved evaluation results — " + str(rows[0]["evaluation_scope"]), fontsize=9)
        path = output / "rouge-comparison.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path))
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = json.loads(Path(args.review).read_text(encoding="utf-8"))
    print(json.dumps(render(report, args.output), indent=2))
