"""CPU-only linear CKA of saved, already pooled report representations.

No model loading, tokenisation, pooling, PCA fitting or inference. Following
Kornblith et al. (2019), columns are centred over the SAME ordered examples:
CKA = ||Xc.T Yc||_F^2 / (||Xc.T Xc||_F ||Yc.T Yc||_F).
This is the conventional biased linear estimator, not unbiased HSIC. The code
uses its equivalent centred linear Gram form to bound memory for a small panel.
https://proceedings.mlr.press/v97/kornblith19a.html
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

MAX_ROWS, MAX_FEATURES, MAX_BYTES = 256, 2048, 8_000_000
MATCH_FIELDS = ("data_split", "source_manifest_sha256", "tokenizer_revision",
                "prompt_sha256", "source_length", "pooling", "site", "visible_tokens_sha256")
VERSION_FIELDS = ("model_label", "model_revision", "checkpoint_sha256")


def real_array(value, dimensions, name):
    array = np.asarray(value)
    if array.ndim != dimensions or array.dtype.kind not in "fiu" or 0 in array.shape:
        raise ValueError(f"{name} must be a nonempty {dimensions}D real numeric array")
    array = array.astype(np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def report_ids(values, count):
    ids = np.asarray(values)
    if ids.ndim != 1 or ids.dtype.kind != "U" or len(ids) != count:
        raise ValueError("report_ids must be a 1D Unicode string array matching rows")
    result = ids.tolist()
    if any(not value.strip() or "\n" in value or "\r" in value for value in result) or len(set(result)) != count:
        raise ValueError("report_ids must be nonempty and unique")
    return result


def centred_scaled(array):
    # Positive isotropic rescaling before/after centring preserves CKA and avoids
    # overflow/underflow from finite but extreme input magnitudes.
    # Remove exact constants first so large offsets cannot erase varying features.
    constant_columns = np.all(array == array[0], axis=0)
    array = np.where(constant_columns, 0.0, array)
    magnitude = np.max(np.abs(array))
    if magnitude == 0:
        return None
    array = array / magnitude
    array -= array.mean(axis=0, keepdims=True)
    # Repeated decimal constants can acquire a mean-rounding residual. Such
    # columns have exactly zero variance, not a tiny signal to renormalise.
    array[:, constant_columns] = 0
    magnitude = np.max(np.abs(array))
    return None if magnitude == 0 else array / magnitude


def linear_cka(x, y, x_ids, y_ids):
    """Require identical IDs/order; never silently reorder, subset or deduplicate."""
    x, y = real_array(x, 2, "X"), real_array(y, 2, "Y")
    left_ids, right_ids = report_ids(x_ids, len(x)), report_ids(y_ids, len(y))
    if left_ids != right_ids:
        raise ValueError("Report IDs and order must match exactly")
    if len(x) > MAX_ROWS or max(x.shape[1], y.shape[1]) > MAX_FEATURES:
        raise ValueError("Saved-array panel exceeds the restrained CPU limits")
    result = {"estimator": "biased centred linear CKA", "cka": None,
              "available": False, "unavailable_reason": None,
              "rows": len(x), "left_features": x.shape[1], "right_features": y.shape[1],
              "formula": "||Xc.T Yc||_F^2 / (||Xc.T Xc||_F * ||Yc.T Yc||_F)",
              "report_ids_sha256": hashlib.sha256("\n".join(left_ids).encode()).hexdigest(),
              "limitations": ["Descriptive representation similarity; not model quality, confidence or a causal mechanism.",
                              "No sample-uncertainty interval or patient-independence claim.",
                              "Small panels and high feature/sample ratios can inflate similarity."]}
    if len(x) < 3:
        result["unavailable_reason"] = "Need at least three reports; two-row centred CKA is degenerate"
        return result
    x, y = centred_scaled(x), centred_scaled(y)
    if x is None or y is None:
        result["unavailable_reason"] = "Zero centred variance in one or both representations"
        return result
    k, l = x @ x.T, y @ y.T
    value = float(np.sum(k * l) / (np.linalg.norm(k) * np.linalg.norm(l)))
    if not np.isfinite(value) or not -1e-12 <= value <= 1 + 1e-12:
        raise ValueError("Numerically invalid CKA result")
    result.update(cka=min(1.0, max(0.0, value)), available=True)
    return result


def validate_metadata(metadata, rows):
    if not isinstance(metadata, dict):
        raise ValueError("Metadata must be a JSON object")
    for key in MATCH_FIELDS + VERSION_FIELDS:
        value = metadata.get(key)
        if value is None or value == "" or value == [] or value == {}:
            raise ValueError(f"Missing saved provenance: {key}")
    for key in VERSION_FIELDS + ("source_manifest_sha256", "tokenizer_revision", "prompt_sha256", "site", "pooling"):
        if not isinstance(metadata[key], str) or not metadata[key].strip():
            raise ValueError(f"Saved provenance must be a nonempty string: {key}")
    if metadata["data_split"] not in ("synthetic", "train", "dev"):
        raise ValueError("Only synthetic/train/dev saved arrays are accepted; official validation is reserved")
    if type(metadata["source_length"]) is not int or metadata["source_length"] <= 0:
        raise ValueError("source_length must be a positive integer")
    visible = metadata["visible_tokens_sha256"]
    if not isinstance(visible, list) or len(visible) != rows or any(
            not isinstance(item, str) or len(item) != 64 or any(c not in "0123456789abcdef" for c in item)
            for item in visible):
        raise ValueError("Require one lowercase SHA256 of exact visible token IDs/mask per report")


def load_saved(path):
    """NPZ contains activations[n,d], report_ids[n], metadata (scalar JSON text).

    Metadata declares pooling and provenance; this reader does not verify that
    upstream extraction obeyed them. Pickled/object arrays are never accepted.
    """
    path = Path(path).resolve()
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Saved array file exceeds the 8 MB limit")
    data = path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        if (sorted(item.filename for item in members) != ["activations.npy", "metadata.npy", "report_ids.npy"]
                or sum(item.file_size for item in members) > MAX_BYTES):
            raise ValueError("Expected only three small NPZ arrays (8 MB expanded limit)")
    with np.load(io.BytesIO(data), allow_pickle=False) as saved:
        values = real_array(saved["activations"], 2, "activations")
        ids = report_ids(saved["report_ids"], len(values))
        text = saved["metadata"]
        if text.ndim != 0 or text.dtype.kind != "U":
            raise ValueError("metadata must be a scalar Unicode JSON string")
        metadata = json.loads(text.item())
    validate_metadata(metadata, len(values))
    return {"values": values, "ids": ids, "metadata": metadata,
            "source": {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}}


def compare_saved(left, right):
    for batch in (left, right):
        validate_metadata(batch["metadata"], len(batch["values"]))
    for key in MATCH_FIELDS:
        if left["metadata"][key] != right["metadata"][key]:
            raise ValueError(f"Saved representation exposure/settings differ: {key}")
    result = linear_cka(left["values"], right["values"], left["ids"], right["ids"])
    result.update(left_metadata=left["metadata"], right_metadata=right["metadata"],
                  sources=[left["source"], right["source"]],
                  provenance_note="Recorded upstream metadata, not independent verification of extraction or weights")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output in (args.left.resolve(), args.right.resolve()):
        parser.error("Choose a new output file; never overwrite source arrays or an existing result")
    result = compare_saved(load_saved(args.left), load_saved(args.right))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "cka": result["cka"], "available": result["available"]}))


if __name__ == "__main__":
    main()
