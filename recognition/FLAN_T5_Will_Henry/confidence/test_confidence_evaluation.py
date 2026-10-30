import unittest
import json,tempfile
from pathlib import Path
import numpy as np
from confidence_evaluation import score_metrics,report_weights,bootstrap,diagnostics,load_test
from confidence_core import sha,pair_id

class EvaluationTests(unittest.TestCase):
    def test_test_hold_mapping_and_feature_contract(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root=Path(directory);files={n:[] for n in ('features.jsonl','labels.jsonl','metadata.jsonl','scoring-mask.jsonl')}
            for i in range(2):
                feature={'source':'invented source','candidate':'invented candidate '+str(i)}
                objects=[feature,{'material_error':None if i else False},
                         {'pair_sha256':pair_id(**feature),'lineage':[{'group_sha256':'g','mapped_official_ids':['validation:1','validation:2'],'source_id':'validation:1','slot':'correct'}]},
                         {'score':i==0}]
                for name,obj in zip(files,objects):files[name].append(obj)
            manifest={'counts':{'components':1,'official_mapped_rows':2,'usable_pairs':1},'sha256':{}}
            def save():
                for name,objects in files.items():
                    p=root/name;p.write_text(''.join(json.dumps(obj)+'\n' for obj in objects));manifest['sha256'][name]=sha(p)
                (root/'manifest.json').write_text(json.dumps(manifest))
            save();rows,receipt=load_test(root)
            self.assertEqual((receipt['all_pairs'],receipt['usable_pairs'],receipt['held_pairs']),(2,1,1))
            self.assertIsNone(rows[1]['label'])
            files['features.jsonl'][0]['category']='leak';save()
            with self.assertRaisesRegex(ValueError,'metadata leakage'):load_test(root)

    def test_dynamic_batches_preserve_all_rows_and_attention_limit(self):
        from evaluate_confidence import batches
        rows=[{'pair_sha256':str(i)} for i in range(5)];encoded={str(i):{'input_ids':[1]*n} for i,n in enumerate([20,20,50,100,200])}
        groups=batches(rows,encoded,maximum=2,area=10000)
        self.assertEqual([r['pair_sha256'] for g in groups for r in g],[str(i) for i in range(5)])
        for group in groups:
            self.assertTrue(len(group)<=2)
            self.assertTrue(len(group)==1 or len(group)*max(len(encoded[r['pair_sha256']]['input_ids']) for r in group)**2<=10000)

    def test_ranking_calibration_and_report_mass(self):
        rows=[{'mapped_ids':['a','b'],'label':0,'group_sha256':'ab','categories':['correct']},
              {'mapped_ids':['a','b'],'label':1,'group_sha256':'ab','categories':['omission']},
              {'mapped_ids':['c'],'label':0,'group_sha256':'c','categories':['correct']},
              {'mapped_ids':['c'],'label':1,'group_sha256':'c','categories':['negation']}]
        weights,count=report_weights(rows);self.assertEqual(count,3);self.assertAlmostEqual(weights.sum(),3)
        np.testing.assert_allclose(weights,[1,1,.5,.5])
        z=np.array([-3,3,-2,2]);y=[r['label'] for r in rows]
        result=score_metrics(y,z,weights);self.assertEqual(result['auroc'],1);self.assertEqual(result['correct_auprc'],1)
        b=bootstrap(y,z,[r['group_sha256'] for r in rows],weights,draws=10)
        self.assertEqual(b['source_groups'],2);self.assertEqual(b['percentile_95']['auroc'],[1.,1.])
        d=diagnostics(rows,z);self.assertEqual(d['categories']['omission']['miss_rate_at_05'],0)
        self.assertEqual(d['risk_coverage'][2]['weak_error_risk'],0)
        self.assertEqual(d['risk_coverage'][2]['accepted_pairs'],1)

if __name__=='__main__':unittest.main()
