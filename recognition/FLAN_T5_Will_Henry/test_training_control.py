import unittest
from training_control import selection_point, update_selection


class SelectionTests(unittest.TestCase):
    def test_diagnostic_partial_epoch_does_not_change_best_or_patience(self):
        self.assertEqual(update_selection(1.0, 1, 0.1, False, 0.001), (1.0, 1, False))
        self.assertEqual(update_selection(1.0, 1, 5.0, False, 0.001), (1.0, 1, False))

    def test_interruption_does_not_add_selection_decisions(self):
        def run(events):
            best, stale, decisions = float("inf"), 0, []
            for step, finished, loss in events:
                eligible = selection_point(finished, step, 7)
                best, stale, improved = update_selection(best, stale, loss, eligible, 0.001)
                if eligible:
                    decisions.append((step, best, stale, improved))
            return decisions

        uninterrupted = [(3, True, 1.0), (6, True, 1.1), (7, False, 0.9)]
        interrupted = [(1, False, 0.5), (3, True, 1.0), (4, False, 2.0), (6, True, 1.1), (7, False, 0.9)]
        self.assertEqual(run(uninterrupted), run(interrupted))
        self.assertEqual([row[0] for row in run(interrupted)], [3, 6, 7])
