"""End-to-end audit consistency using synthetic parquet files, not course data."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from dataset import DATASET, DATA_REVISION, prepare, read_jsonl
from verify_manifests import HASH_CONTRACT, verify_manifests


class PreparationTests(unittest.TestCase):
    def test_preparation_preserves_official_evaluation_and_passes_independent_check(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for split, texts in {"train": ["a", "b", "c", "d", "same as eval", "a"],
                                 "validation": ["same as eval", "x", "x"]}.items():
                rows = [{"source": "synthetic", "images_path": f"{split}/{i}",
                         "radiology_report": text, "layman_report": "plain " + text}
                        for i, text in enumerate(texts)]
                pq.write_table(pa.Table.from_pylist(rows), root / f"{split}-00000-of-00001.parquet")
            prepared = root / "prepared"
            with redirect_stdout(io.StringIO()):
                prepare(prepared, raw_dir=root)
            manifests = {name: prepared / f"{name}.jsonl" for name in ("train", "dev", "evaluation")}
            report = verify_manifests(manifests, prepared / "audit.json", expected_dataset=DATASET,
                                      expected_revision=DATA_REVISION, stored_key_contract=HASH_CONTRACT)
            self.assertTrue(report["ok"], report)
            self.assertEqual(len(read_jsonl(manifests["evaluation"])), 3)
            audit = json.loads((prepared / "audit.json").read_text())
            self.assertEqual(audit["files"]["train"]["raw_rows"], 6)
            self.assertEqual(audit["train_overlap_removed"], 1)
            self.assertEqual(audit["official_usable_counts"], {"train": 6, "validation": 3})


if __name__ == "__main__":
    unittest.main()
