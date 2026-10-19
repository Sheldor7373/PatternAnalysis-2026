import unittest
from metrics import concept_states, safety_flags, sentences


class SafetyTests(unittest.TestCase):
    def test_negation_reversal_is_flagged(self):
        flags = safety_flags("No pneumothorax.", "There is a pneumothorax.")
        self.assertIn("pneumothorax", flags["changed_states"])

    def test_uncertainty_removed_is_flagged(self):
        flags = safety_flags("Possible pneumonia.", "There is pneumonia.")
        self.assertIn("pneumonia", flags["changed_states"])

    def test_recognised_plain_english_does_not_add_a_diagnosis(self):
        flags = safety_flags("Cardiomegaly.", "An enlarged heart.")
        self.assertFalse(any(flags.values()))

    def test_missing_finding_is_a_review_candidate(self):
        self.assertEqual(safety_flags("A fracture.", "Everything is normal.")["missing_concepts"], ["fracture"])

    def test_clauses_and_newlines(self):
        self.assertEqual(sentences("One.\nTwo! Three?"), ["One.", "Two!", "Three?"])
        states = concept_states("No pneumonia but an effusion.")
        self.assertEqual(states["pneumonia"], ["negated"])
        self.assertEqual(states["effusion"], ["affirmed"])


if __name__ == "__main__":
    unittest.main()
