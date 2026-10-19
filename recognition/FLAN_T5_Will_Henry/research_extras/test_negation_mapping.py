"""Exact saved first-case trace and adversarial synthetic offset fixtures."""
import copy
import unittest

from negation_mapping import POLICY_VERSION, MappingFailure, map_source_span
from negation_preflight import preflight
from negation_summary import validate_preflight
from test_negation_preflight import FixtureTokenizer, fixture_manifest

IDS = [419,17504,48,2252,6427,934,16,6080,1566,5,1266,3473,15,66,7469,6,
       14261,257,11,14068,10,465,3,4788,9709,13577,11733,5]
OFFSETS = [[0,2],[2,7],[7,12],[12,18],[18,22],[22,29],[29,32],[32,38],
           [38,46],[46,47],[47,51],[51,55],[55,56],[56,60],[60,69],[69,70],
           [70,74],[74,79],[79,83],[83,95],[95,96],[96,99],[99,100],
           [100,103],[103,107],[107,111],[111,116],[116,117]]


class MappingTests(unittest.TestCase):
    def setUp(self):
        self.manifest = fixture_manifest()
        self.row = self.manifest["sources"][0]
        self.prompt = self.manifest["input_contract"]["prompt"]
        self.text = self.prompt + self.row["source"]

    def mapped(self, offsets=None):
        return map_source_span(self.text, len(self.prompt), self.row["source_char_span"],
                               IDS + [1], (offsets or copy.deepcopy(OFFSETS)) + [[0,0]], [True] * 29, [0,1,2])

    def test_exact_first_trace_maps_21_27_with_newline_exception_and_unchanged_offsets(self):
        before = copy.deepcopy(OFFSETS)
        result = self.mapped(OFFSETS)
        self.assertEqual(result["token_span"], [21,27])
        self.assertEqual(OFFSETS, before)
        self.assertEqual(result["mapping"]["policy_version"], POLICY_VERSION)
        exception = result["mapping"]["boundary_exceptions"]
        self.assertEqual(len(exception), 1)
        self.assertEqual(exception[0]["outside_text"], "\n")
        self.assertEqual(exception[0]["raw_offset"], [96,99])
        self.assertEqual(exception[0]["inside_span_text"], "No")
        self.assertFalse(result["mapping"]["raw_offsets_modified"])

    def test_internal_whitespace_preserved_prompt_and_eos_ineligible(self):
        result = self.mapped()
        self.assertEqual(IDS[22], 3)
        self.assertEqual(self.text[slice(*OFFSETS[22])], " ")
        self.assertTrue(all(result["eligible_mask"][21:27]))
        self.assertFalse(any(result["eligible_mask"][:21]))
        self.assertFalse(result["eligible_mask"][-1])

    def test_successful_trailing_whitespace_overhang_preserves_raw_offsets(self):
        offsets = [[0,2],[2,5],[5,6],[0,0]]
        before = copy.deepcopy(offsets)
        result = map_source_span("X\nNo .", 2, [0,2], [7,8,9,1], offsets, [True]*4, [0,1,2])
        self.assertEqual(result["token_span"], [1,2])
        self.assertEqual(offsets, before)
        self.assertEqual(result["mapping"]["boundary_exceptions"], [{"token_index": 1, "token_id": 8,
            "raw_offset": [2,5], "boundary": "stop", "outside_char_span": [4,5],
            "outside_text": " ", "inside_span_text": "No"}])

    def test_prompt_letter_or_punctuation_and_report_punctuation_overhang_rejected(self):
        for start in (94,95):  # prompt y/colon as well as newline
            offsets = copy.deepcopy(OFFSETS); offsets[21][0] = start
            with self.subTest(start=start), self.assertRaises(MappingFailure): self.mapped(offsets)
        offsets = copy.deepcopy(OFFSETS); offsets[26][1] = 117  # includes final period
        with self.assertRaises(MappingFailure): self.mapped(offsets)

    def test_prompt_only_whitespace_is_ineligible_and_whitespace_only_overhang_rejected(self):
        result = map_source_span("X\nNo", 2, [0,2], [7,8,9,1], [[0,1],[1,2],[2,4],[0,0]],
                                 [True]*4, [0,1,2])
        self.assertEqual(result["token_span"], [2,3])
        self.assertFalse(result["eligible_mask"][1])
        with self.assertRaises(MappingFailure):
            map_source_span("X\n No", 2, [0,3], [7,8,9,1], [[0,1],[1,3],[3,5],[0,0]],
                            [True]*4, [0,1,2])

    def test_nonwhitespace_coverage_contiguity_specials_and_attention_required(self):
        offsets = copy.deepcopy(OFFSETS); offsets[23][0] = 101  # omits p
        with self.assertRaises(MappingFailure): self.mapped(offsets)
        ids, offsets = IDS[:23] + [99] + IDS[23:] + [1], OFFSETS[:23] + [[0,1]] + OFFSETS[23:] + [[0,0]]
        with self.assertRaises(MappingFailure):
            map_source_span(self.text, 97, [0,19], ids, offsets, [True]*len(ids), [0,1,2])
        for special, masked in (([0,1,2,465], False), ([0,1,2], True)):
            attention = [True]*29
            if masked: attention[22] = False
            with self.assertRaises(MappingFailure):
                map_source_span(self.text, 97, [0,19], IDS+[1], OFFSETS+[[0,0]], attention, special)

    def test_preflight_and_summary_share_exact_mapping_and_exception_validation(self):
        first_text = self.text
        class ExactFirstTrace(FixtureTokenizer):
            def __call__(self, text, **kwargs):
                if text == first_text:
                    self.calls.append((text, kwargs))
                    return {"input_ids": IDS.copy(), "offset_mapping": copy.deepcopy(OFFSETS), "attention_mask": [1]*28}
                return super().__call__(text, **kwargs)
        result = preflight(self.manifest, ExactFirstTrace(self.manifest))
        self.assertEqual(result["status"], "PASS")
        source = result["records"][0]["source"]
        self.assertEqual(source["token_span"], [21,27])
        self.assertEqual(source["offset_mapping"], OFFSETS + [[0,0]])
        self.assertTrue(source["eligible_mask"][22])
        self.assertEqual(validate_preflight(self.manifest, result)[self.row["id"]]["source"], source)
        result["records"][0]["source"]["span_mapping"]["boundary_exceptions"][0]["outside_text"] = "\t"
        with self.assertRaises(ValueError): validate_preflight(self.manifest, result)


if __name__ == "__main__":
    unittest.main()
