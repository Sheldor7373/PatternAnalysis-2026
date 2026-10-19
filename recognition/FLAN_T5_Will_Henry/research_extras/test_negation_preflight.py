"""Synthetic tokenizer fixtures only; no pretrained tokenizer/model in tests."""

import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from negation_preflight import (PINNED_REVISION, digest, validate_contract, token_record,
                                preflight, load_cached_tokenizer, main)

MANIFEST = Path(__file__).with_name("negation_experiment") / "negation_manifest.json"


def fixture_manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


class FixtureTokenizer:
    is_fast, eos_token_id, pad_token_id, all_special_ids = True, 1, 0, [0, 1, 2]

    def __init__(self, manifest):
        texts = [manifest["input_contract"]["prompt"]]
        for row in manifest["sources"]:
            texts += [row["source"]] + [c["text"] for c in row["candidates"].values()]
        vocabulary = sorted({piece.group() for text in texts for piece in re.finditer(r"\w+|[^\w\s]", text)})
        self.vocabulary = {piece: i + 3 for i, piece in enumerate(vocabulary)}
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        assert kwargs == {"add_special_tokens": False, "truncation": False,
                          "return_offsets_mapping": True, "return_attention_mask": True}
        pieces = list(re.finditer(r"\w+|[^\w\s]", text))
        return {"input_ids": [self.vocabulary[piece.group()] for piece in pieces],
                "offset_mapping": [list(piece.span()) for piece in pieces], "attention_mask": [1] * len(pieces)}


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.manifest = fixture_manifest()

    def test_complete_fixed_sources_offsets_masks_eos_counts_and_repeatability(self):
        before = copy.deepcopy(self.manifest)
        tokenizer = FixtureTokenizer(self.manifest)
        result = preflight(self.manifest, tokenizer)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["completed_sources"], 46)
        self.assertEqual(self.manifest, before)
        self.assertEqual(digest(result), digest(preflight(self.manifest, FixtureTokenizer(self.manifest))))
        first = result["records"][0]
        source = first["source"]
        start, stop = source["token_span"]
        self.assertEqual(source["offset_mapping"][start][0], len(self.manifest["input_contract"]["prompt"]))
        self.assertTrue(all(source["eligible_mask"][start:stop]))
        self.assertFalse(any(source["eligible_mask"][:start]))
        self.assertFalse(source["eligible_mask"][-1])
        self.assertEqual(source["input_ids"].count(1), 1)
        for row in result["records"]:
            for candidate in row["candidates"].values():
                self.assertEqual(candidate["input_ids"].count(1), 1)
                self.assertEqual(candidate["input_ids"][-1], 1)
                self.assertEqual(candidate["token_count"], sum(candidate["target_mask"]))
                self.assertTrue(candidate["eos_included"])
        self.assertEqual(len(tokenizer.calls), 46 * 3)

    def test_mask_aware_counts_and_active_hash_ignore_only_masked_values(self):
        first = token_record([0, 7, 1, 0], [True, True, True, False])
        other = token_record([0, 7, 1, 99], [True, True, True, False])
        self.assertEqual(first["token_count"], 3)  # valid token ID0 is counted
        self.assertEqual(first["active_tokens_sha256"], other["active_tokens_sha256"])
        self.assertNotEqual(first["forward_input_sha256"], other["forward_input_sha256"])
        self.assertNotEqual(first["active_tokens_sha256"], token_record([0, 8, 1, 0], [True, True, True, False])["active_tokens_sha256"])
        with self.assertRaises(ValueError):
            token_record([1], [1])

    def test_boundary_failure_stops_first_case_without_adjustment(self):
        prompt_end = len(self.manifest["input_contract"]["prompt"])
        class Straddling(FixtureTokenizer):
            def __call__(self, text, **kwargs):
                result = super().__call__(text, **kwargs)
                for pair in result["offset_mapping"]:
                    if pair[0] == prompt_end:
                        pair[0] -= 1
                        break
                return result
        tokenizer = Straddling(self.manifest)
        before = digest(self.manifest)
        result = preflight(self.manifest, tokenizer)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["failure"]["code"], "span_boundary")
        self.assertEqual(result["completed_sources"], 0)
        self.assertEqual(len(tokenizer.calls), 1)
        self.assertEqual(digest(self.manifest), before)

    def test_source_and_candidate_caps_stop_without_truncation(self):
        for key, component in (("max_source_tokens", "source"), ("max_candidate_tokens", "positive")):
            manifest = copy.deepcopy(self.manifest)
            manifest["input_contract"][key] = 1
            result = preflight(manifest, FixtureTokenizer(manifest))
            self.assertEqual(result["status"], "FAIL")
            self.assertEqual(result["failure"]["component"], component)
            self.assertTrue(result["failure"]["code"].endswith("cap"))
            self.assertEqual(result["completed_sources"], 0)

    def test_duplicate_eos_and_incomplete_offsets_are_failures(self):
        class ExtraEOS(FixtureTokenizer):
            def __call__(self, text, **kwargs):
                value = super().__call__(text, **kwargs)
                value["input_ids"][0] = 1
                return value
        self.assertEqual(preflight(self.manifest, ExtraEOS(self.manifest))["failure"]["code"], "eos")
        class MissingOffsets(FixtureTokenizer):
            def __call__(self, text, **kwargs):
                value = super().__call__(text, **kwargs)
                for key in value:
                    value[key] = value[key][1:]
                return value
        self.assertEqual(preflight(self.manifest, MissingOffsets(self.manifest))["failure"]["code"], "coverage")

    def test_split_and_identity_corruption_rejected_before_tokenization(self):
        for change in ("family", "heldout_term", "donor", "duplicate", "missing_case"):
            manifest = copy.deepcopy(self.manifest)
            if change == "family": manifest["family_splits"]["fit"].append("negation-08")
            if change == "heldout_term": manifest["sources"][0]["source"] += " atelectasis"
            if change == "donor": manifest["intervention_cases"][0]["donor_id"] = manifest["intervention_cases"][0]["recipient_id"]
            if change == "duplicate": manifest["sources"].append(copy.deepcopy(manifest["sources"][0]))
            if change == "missing_case": manifest["intervention_cases"].pop()
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_contract(manifest)

    def test_cached_loader_is_local_only_and_hashes_no_weight_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / ("flan-t5-small-" + PINNED_REVISION)
            path.mkdir()
            for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "spiece.model"):
                (path / name).write_bytes(b"synthetic asset")
            (path / "config.json").write_text(json.dumps({"eos_token_id": 1, "pad_token_id": 0}))
            (path / "model.safetensors").write_bytes(b"not read")
            calls = []
            def local_only(*args, **kwargs):
                calls.append((args, kwargs))
                return FixtureTokenizer(self.manifest)
            with patch.dict("sys.modules", {"transformers": SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=local_only))}):
                _, identity = load_cached_tokenizer(path, PINNED_REVISION)
            self.assertEqual(calls[0][1], {"local_files_only": True, "use_fast": True, "trust_remote_code": False})
            self.assertNotIn("model.safetensors", identity["files"])
            self.assertEqual(len(identity["files"]), 5)

    def test_cli_writes_new_report_and_preserves_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "inputs"; inputs.mkdir()
            path = inputs / "manifest.json"
            path.write_text(json.dumps(self.manifest))
            raw = path.read_bytes()
            output = root / "result.json"
            argv = ["--manifest", str(path), "--expected-manifest-sha256", hashlib.sha256(raw).hexdigest(),
                    "--tokenizer", str(root / "cache"), "--output", str(output)]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(argv, loader=lambda *args: (FixtureTokenizer(self.manifest), {"kind": "synthetic"})), 0)
            self.assertEqual(path.read_bytes(), raw)
            with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()):
                main(argv, loader=lambda *args: (FixtureTokenizer(self.manifest), {}))


if __name__ == "__main__":
    unittest.main()
