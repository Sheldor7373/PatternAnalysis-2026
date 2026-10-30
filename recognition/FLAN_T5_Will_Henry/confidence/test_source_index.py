import hashlib,json,os,pathlib,tempfile,unittest
from source_index import build,lookup

class SourceIndexTests(unittest.TestCase):
    def test_identical_lookup_stale_and_wrong_hash(self):
        with tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).parent) as t:
            p=pathlib.Path(t)/'source.jsonl'; rows=[{'id':'a','input':'invented one','target':'r1'},{'id':'b','input':'invented two','target':'r2'}];p.write_text(''.join(json.dumps(x)+'\n' for x in rows),encoding='utf-8');build(p,t)
            for x in rows:self.assertEqual(lookup(p,x['id'],hashlib.sha256(x['input'].encode()).hexdigest(),t),x)
            with self.assertRaises(ValueError):lookup(p,'a','0'*64,t)
            p.write_text(p.read_text()+'\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'stale'):lookup(p,'a',hashlib.sha256(rows[0]['input'].encode()).hexdigest(),t)
    def test_duplicate_id_rejected(self):
        with tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).parent) as t:
            p=pathlib.Path(t)/'source.jsonl';p.write_text((json.dumps({'id':'a','input':'x'})+'\n')*2,encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'duplicate'):build(p,t)

if __name__=='__main__':unittest.main()
