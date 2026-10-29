"""The operational same-batch recovery must not prevent a safe batch change."""
import unittest
from training_transition import transition_policy, lr_factor

class RecoveryBatchPolicyTests(unittest.TestCase):
    def test_reanchor_after_same_batch_recovery_keeps_next_lr_and_exposure(self):
        old = dict(epochs=3, batch_size=1, accumulate=2, max_steps=None)
        step = 40779
        lr = 1e-4 * (146118-step)/(146118-7305)
        state = dict(epoch=0, next_sample=81558, next_batch=81558, step=step,
                     optimizer={'param_groups':[{'lr':lr}]},
                     scheduler={'last_epoch':step, '_last_lr':[lr], 'base_lrs':[1e-4]})
        recovery = transition_policy(state, old, old, 97411)
        state['scheduler_policy'] = recovery
        for batch in (2, 4):
            policy = transition_policy(state, old, {**old, 'batch_size':batch, 'accumulate':1}, 97411)
            self.assertEqual(lr_factor(policy, step)*1e-4, lr)
            self.assertEqual(policy['anchor_examples'], 81558)
            self.assertEqual(policy['final_examples'], 292233)
            self.assertEqual(lr_factor(policy, policy['planned_steps']), 0)
            if batch == 2:
                self.assertEqual(policy['kind'], 'original_updates')
                self.assertEqual(policy['planned_steps'], 146118)
            else:
                self.assertEqual(policy['kind'], 'anchored_examples')
                self.assertEqual(policy['planned_steps'], 93449)
        bad = {**state, 'scheduler_policy':{**recovery, 'planned_steps':1}}
        with self.assertRaisesRegex(ValueError, 'does not match'):
            transition_policy(bad, old, old, 97411)
        bad['scheduler_policy']['kind'] = 'anchored_examples'
        with self.assertRaisesRegex(ValueError, 'second effective-batch'):
            transition_policy(bad, old, old, 97411)

if __name__ == '__main__':
    unittest.main()
