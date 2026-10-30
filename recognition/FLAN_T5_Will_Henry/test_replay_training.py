import json,unittest
from pathlib import Path
from replay_training import command_plan,STAGES
import train

class ReplayPlanTests(unittest.TestCase):
    def test_plans_have_exact_boundaries_and_do_not_truncate_schedule(self):
        history=json.loads((Path(__file__).parent/'configs/historical_training.json').read_text())
        for scenario in ('full','lora','extra'):
            plan=command_plan(history,scenario,'data folder','model folder','output folder','parent checkpoint' if scenario=='extra' else None)
            self.assertEqual(len(plan),len(STAGES[scenario]))
            for i,phase in enumerate(plan):
                args=train.parser().parse_args(phase['command'][3:])
                self.assertEqual(args.data,'data folder');self.assertEqual(args.model,'model folder')
                self.assertIsNone(args.max_steps);self.assertIsNone(args.max_minutes)
                self.assertEqual(args.stop_at_step,phase['expected_step'] if i<len(plan)-1 else None)
                self.assertEqual(args.epochs,4 if scenario=='extra' else 3)
                if i:self.assertEqual(args.transition_from,'${PREVIOUS_IMMUTABLE_CHECKPOINT}')
        self.assertEqual([p['expected_exposures'] for p in command_plan(history,'full','d','m','r')],[20362,81558,292233])

if __name__=='__main__':unittest.main()
