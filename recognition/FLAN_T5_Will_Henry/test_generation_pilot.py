"""CPU-only safeguards around development timing-set selection."""
from pathlib import Path
import tempfile
import unittest

from dataset import read_jsonl, select_rows, write_jsonl
from prepare_generation_pilot import prepare


class GenerationPilotTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.data=self.root/'prepared'
        self.data.mkdir()
        self.rows=[{'id':str(i),'input':'source','target':'reference'} for i in range(20)]
        write_jsonl(self.data/'dev.jsonl',self.rows)

    def test_selection_is_repeatable_and_excludes_monitor_without_validation(self):
        # There is deliberately no evaluation.jsonl to read.
        before=(self.data/'dev.jsonl').read_bytes()
        a=self.root/'pilot-a/dev.jsonl'
        b=self.root/'pilot-b/dev.jsonl'
        first=prepare(self.data,a,5,4)
        second=prepare(self.data,b,5,4)
        self.assertEqual(a.read_bytes(),b.read_bytes())
        self.assertEqual(first,second)
        selected={row['id'] for row in read_jsonl(a)}
        monitored={row['id'] for row in select_rows(self.rows,4,3710)}
        self.assertFalse(selected & monitored)
        self.assertEqual((self.data/'dev.jsonl').read_bytes(),before)

    def test_prepared_inputs_and_existing_pilot_are_protected(self):
        with self.assertRaisesRegex(ValueError,'outside'):
            prepare(self.data,self.data/'pilot.jsonl',5,4)
        output=self.root/'pilot.jsonl'
        prepare(self.data,output,5,4)
        before=output.read_bytes()
        with self.assertRaisesRegex(ValueError,'fresh'):
            prepare(self.data,output,5,4)
        self.assertEqual(output.read_bytes(),before)

    def test_insufficient_disjoint_rows_fail_without_creating_output(self):
        output=self.root/'pilot.jsonl'
        with self.assertRaisesRegex(ValueError,'Not enough'):
            prepare(self.data,output,19,4)
        self.assertFalse(output.exists())


if __name__=='__main__':
    unittest.main()
