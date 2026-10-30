import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from transformers import T5Config,T5EncoderModel
from confidence_core import encode_pair,collate,pooled_states,split_group,pair_id,load_assembly,sha,ConfidenceMLP,JointCritic
from argparse import Namespace

class Tokenizer:
    pad_token_id=0; eos_token_id=1
    def encode(self,text,add_special_tokens=False): return [2+ord(x)%40 for x in text]

class ConfidenceTests(unittest.TestCase):
    def test_adaptation_continues_head_and_checkpoints_exact_cursor(self):
        import adapt_confidence as a
        from inference_identity import artifact_manifest
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root=Path(directory);encoder_path=root/'encoder';stage=root/'stage-a';stage.mkdir()
            torch.manual_seed(3)
            encoder=T5EncoderModel(T5Config(vocab_size=48,d_model=16,d_kv=8,d_ff=24,num_layers=1,num_heads=2,dropout_rate=0));encoder.save_pretrained(encoder_path)
            recipe={'model_files_sha256':artifact_manifest(encoder_path),'source_cap':64,'candidate_cap':64}
            rows={p:[{'source':'invented normal report','candidate':'invented '+str(i),'label':i%2,'pair_sha256':p+str(i)} for i in range(n)] for p,n in [('fit',6),('validation',4),('calibration',4)]}
            assembly={'input_groups':10000,'fixture_only':True}
            (stage/'run.json').write_text(json.dumps({'assembly':assembly,'cache':{'recipe':recipe,'recipe_sha256':'fixture'}}))
            (stage/'results.json').write_text(json.dumps({'mlp':{}}))
            torch.save({'head':ConfidenceMLP(32,8,0).state_dict(),'dimensions':32,'width':8,'dropout':0,'mean':np.zeros(32,dtype='f'),'scale':np.ones(32,dtype='f')},stage/'mlp.pt')
            args=Namespace(cpu_threads=1,seed=4,device='cpu',output=str(root/'adapt'),data='fixture',view='conservative',stage_a=str(stage),encoder=str(encoder_path),
                           encoder_lr=1e-5,head_lr=1e-4,epochs=1,patience=2,batch_size=2,accumulate=2,precision='fp32',resume=False)
            with patch.object(a,'load_assembly',return_value=(rows,assembly)),patch.object(a.AutoTokenizer,'from_pretrained',return_value=Tokenizer()):a.main(args)
            latest=torch.load(root/'adapt/latest.pt',weights_only=False)
            self.assertEqual((latest['epoch'],latest['cursor'],latest['step']),(1,0,2))
            self.assertEqual([g['lr'] for g in latest['optimizer']['param_groups']],[1e-5,1e-4])
            self.assertTrue((root/'adapt/results.json').exists())
            # Completed recovery is idempotent and does not run an extra epoch.
            args.resume=True
            with patch.object(a,'load_assembly',return_value=(rows,assembly)),patch.object(a.AutoTokenizer,'from_pretrained',return_value=Tokenizer()):a.main(args)
            self.assertEqual(torch.load(root/'adapt/latest.pt',weights_only=False)['step'],2)
            original_sha=sha(root/'adapt/latest.pt')
            paused=Namespace(**vars(args));paused.resume=False;paused.output=str(root/'paused');paused.continue_from=None
            (root/'paused').mkdir();(root/'paused/STOP').write_text('fixture pause')
            with patch.object(a,'load_assembly',return_value=(rows,assembly)),patch.object(a.AutoTokenizer,'from_pretrained',return_value=Tokenizer()):a.main(paused)
            state=torch.load(root/'paused/latest.pt',weights_only=False)
            self.assertEqual((state['epoch'],state['cursor'],state['step']),(0,4,1))
            (root/'paused/STOP').unlink();paused.resume=True
            with patch.object(a,'load_assembly',return_value=(rows,assembly)),patch.object(a.AutoTokenizer,'from_pretrained',return_value=Tokenizer()):a.main(paused)
            recovered=torch.load(root/'paused/latest.pt',weights_only=False)
            for key,value in latest['critic'].items():torch.testing.assert_close(value,recovered['critic'][key],atol=0,rtol=0)
            args.resume=False;args.continue_from=str(root/'adapt/latest.pt');args.output=str(root/'late')
            args.epochs=2;args.encoder_lr=3e-6;args.head_lr=3e-5
            with patch.object(a,'load_assembly',return_value=(rows,assembly)),patch.object(a.AutoTokenizer,'from_pretrained',return_value=Tokenizer()):a.main(args)
            continued=torch.load(root/'late/latest.pt',weights_only=False)
            self.assertEqual((continued['epoch'],continued['step']),(2,4))
            self.assertEqual([g['lr'] for g in continued['optimizer']['param_groups']],[3e-6,3e-5])
            self.assertEqual(sha(root/'adapt/latest.pt'),original_sha)

    def test_same_vectors_train_all_heads_with_fit_only_scaling(self):
        from train_confidence import train_mlp,train_trees,temperature
        torch.set_num_threads(1);torch.manual_seed(4)
        rng=np.random.default_rng(4)
        data={}
        for part,n in [('fit',64),('validation',24),('calibration',24)]:
            x=rng.normal(size=(n,8)).astype('f');y=(x[:,0]+x[:,1]>0).astype('int64');data[part]=(x,y)
        args=Namespace(device='cpu',width=8,dropout=0.,lr=.01,batch_size=16,seed=4,epochs=3,patience=2,tree_rounds=5,cpu_threads=1)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            out=Path(directory);predictions,resource=train_mlp(data,out,args)
            checkpoint=torch.load(out/'mlp.pt',weights_only=False)
            np.testing.assert_allclose(checkpoint['mean'],data['fit'][0].mean(0))
            trees=train_trees(data,out,args)
            self.assertEqual(set(trees),{'lightgbm','xgboost'})
            for name,(z,resource) in trees.items():
                self.assertEqual(len(z['validation']),24);self.assertTrue(np.isfinite(z['validation']).all())
            self.assertTrue(np.isfinite(temperature(data['calibration'][1],predictions['calibration'])))

    def test_joint_context_and_padding(self):
        torch.manual_seed(1)
        encoder=T5EncoderModel(T5Config(vocab_size=48,d_model=16,d_kv=8,d_ff=24,num_layers=1,num_heads=2,dropout_rate=0)).eval()
        tok=Tokenizer();a=encode_pair(tok,'left chest','clear');b=encode_pair(tok,'left chest','nodule')
        with torch.no_grad():
            first=pooled_states(encoder,collate([a],0,'cpu'))
            second=pooled_states(encoder,collate([a,b],0,'cpu'))
        torch.testing.assert_close(first[0],second[0],atol=2e-6,rtol=2e-6)
        self.assertFalse(torch.allclose(second[0,:16],second[1,:16])) # candidate changes source states
        self.assertEqual(tuple(second.shape),(2,32))
        head=ConfidenceMLP(32).eval();critic=JointCritic(encoder,head,np.zeros(32,dtype='f'),np.ones(32,dtype='f')).eval()
        with torch.no_grad():torch.testing.assert_close(critic(collate([a,b],0,'cpu')),head(second))
        critic(collate([a,b],0,'cpu')).sum().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in encoder.parameters()))

    def test_explicit_truncation_keeps_both_inputs(self):
        x=encode_pair(Tokenizer(),'abcdef','ghi',source_cap=3,candidate_cap=2)
        self.assertTrue(x['source_truncated'] and x['candidate_truncated'])
        self.assertEqual(sum(x['source_mask']),3);self.assertEqual(sum(x['candidate_mask']),2)
        with self.assertRaises(ValueError):encode_pair(Tokenizer(),'','candidate')

    def test_partition_leakage_and_extra_features_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);hashes={}
            for part in ('fit','validation','calibration'):
                group=next(str(i) for i in range(1000) if split_group(str(i))==part)
                feature={'source':'invented '+part,'candidate':'invented rewrite'}
                for kind,obj in [('features',feature),('labels',{'material_error':False}),('masks',{'conservative':True}),
                                  ('metadata',{'pair_sha256':pair_id(**feature),'lineage':[{'source_id':'train:1','group_sha256':group}]})]:
                    p=root/(part+'-'+kind+'.jsonl');p.write_text(json.dumps(obj)+'\n');hashes[p.name]=sha(p)
            manifest=root/'manifest.json';manifest.write_text(json.dumps({'sha256':hashes}))
            rows,receipt=load_assembly(root);self.assertEqual(receipt['input_groups'],3)
            self.assertEqual(set(rows['fit'][0]),{'source','candidate','pair_sha256','group_sha256','label'})
            p=root/'fit-features.jsonl';f=json.loads(p.read_text());f['category']='error';p.write_text(json.dumps(f)+'\n')
            hashes[p.name]=sha(p);manifest.write_text(json.dumps({'sha256':hashes}))
            with self.assertRaisesRegex(ValueError,'metadata leakage'):load_assembly(root)

if __name__=='__main__':unittest.main()
