import unittest,json,hashlib,sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import assembly_core as a
class Fixtures(unittest.TestCase):
 def test_group_split(self):
  groups=['g'+str(i) for i in range(300)]
  self.assertEqual({a.split(g) for g in groups},{'fit','validation','calibration'})
  self.assertTrue(all(a.split(g)==a.split(g) for g in groups))
 def test_features_have_only_source_and_candidate(self):
  f={'source':'report','candidate':'simplification'}
  self.assertEqual(set(f),{'source','candidate'})
  self.assertFalse(set(f)&{'material_error','source_id','slot','provenance','rationale'})
 def test_conflict_and_exact_pair(self):
  def k(s,c):return a.sha(json.dumps([s,c],ensure_ascii=False,separators=(',',':')).encode())
  self.assertEqual(k('a','x'),k('a','x'));self.assertNotEqual(k('a','x'),k('b','x'))
  self.assertEqual(a.pair_disposition({False,True},{'g'}),'conflict');self.assertEqual(a.pair_disposition({True},{'g','h'}),'cross_component');self.assertEqual(a.pair_disposition({True},{'g'}),'retain')
 def test_repair_exact_once(self):
  obj={'records':[{'slot':'correct','text':'original'},{'slot':'cutoff','text':'old'}]}; data=json.dumps(obj).encode()
  side={'original_checkpoint_sha256':a.sha(data),'updates':[{'slot':'cutoff','text':'new'}]};allowed={'flagged_slots':[{'slot':'cutoff'}]}
  fixed=a.repair(data,side,allowed);self.assertEqual([r['text'] for r in fixed['records']],['original','new'])
  with self.assertRaises(ValueError):a.repair(json.dumps(fixed).encode(),side,allowed)
  with self.assertRaises(ValueError):a.repair(data,side,{'flagged_slots':[{'slot':'other'}]})
 def test_resume_hash(self):
  import tempfile
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d); (p/'x').write_bytes(b'fixture'); (p/'manifest.json').write_text(json.dumps({'output_sha256':{'x':a.sha(b'fixture')}}))
   self.assertTrue(a.verify_resume(p));(p/'x').write_bytes(b'tampered');self.assertFalse(a.verify_resume(p))
if __name__=='__main__':unittest.main()
