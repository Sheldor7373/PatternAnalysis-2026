"""Offline CAD formula/cache checks and real tiny-T5 beam/resume paths."""
from contextlib import redirect_stdout
import io
import tempfile
from pathlib import Path
import unittest
import torch
from transformers import LogitsProcessor
from decode_strategies import ContextAwareProcessor, contrast_logits, decoding_settings, generate_tokens, token_statistics
from test_model import ModelTests
from test_predict_resume import PredictionResumeTests
import predict


class DecodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ModelTests.setUpClass()
        from transformers import T5ForConditionalGeneration
        cls.model = T5ForConditionalGeneration.from_pretrained(ModelTests.path).eval()

    @classmethod
    def tearDownClass(cls):
        ModelTests.tearDownClass()

    def test_contrast_equals_normalized_probability_ratio(self):
        c=torch.tensor([[0.,2.,1.]],dtype=torch.float64)
        p=torch.tensor([[3.,1.,0.]],dtype=torch.float64)
        alpha=.5
        direct=c.softmax(-1).pow(1+alpha)/p.softmax(-1).pow(alpha)
        direct=direct/direct.sum(-1,keepdim=True)
        torch.testing.assert_close(contrast_logits(c,p,alpha).softmax(-1).double(),direct,atol=1e-7,rtol=1e-6)
        for bad in (-1,float('inf'),float('nan')):
            with self.assertRaises(ValueError):decoding_settings('cad',1,bad,'instruction')
        with self.assertRaises(ValueError):decoding_settings('cad',2,.5,'instruction')

    def test_first_eos_stops_content_count_for_beam_filled_tails(self):
        ids=torch.tensor([[0,3,1,1,1],[0,3,4,1,0],[0,3,4,5,6]])
        counts,ended=token_statistics(ids,0,1)
        self.assertEqual(counts,[2,3,4])
        self.assertEqual(ended,[True,True,False])

    def test_alpha_zero_and_standard_beams_equal_native_generate(self):
        tokenizer=ModelTests.tokenizer
        batch=tokenizer(['No effusion','No'],padding=True,return_tensors='pt')
        with torch.inference_mode():
            expected=self.model.generate(**batch,max_new_tokens=7,do_sample=False,num_beams=1)
            actual=generate_tokens(self.model,batch,tokenizer,7,decoder='cad',alpha=0,prior_prompt='No')
            self.assertTrue(torch.equal(expected,actual))
            for beams in (2,4):
                expected=self.model.generate(**batch,max_new_tokens=7,do_sample=False,num_beams=beams)
                actual=generate_tokens(self.model,batch,tokenizer,7,beams=beams)
                self.assertTrue(torch.equal(expected,actual))

    def test_cached_cad_matches_independent_full_prefix_recomputation(self):
        tokenizer=ModelTests.tokenizer; model=self.model
        batch=tokenizer(['No effusion','effusion'],padding=True,return_tensors='pt')
        prior=tokenizer(['No','No'],return_tensors='pt'); alpha=.5
        class Manual(LogitsProcessor):
            def __call__(self,ids,scores):
                logits=model(**prior,decoder_input_ids=ids,use_cache=False).logits[:,-1,:]
                return (1+alpha)*scores.float()-alpha*logits.float()
        with torch.inference_mode():
            expected=model.generate(**batch,max_new_tokens=8,do_sample=False,logits_processor=[Manual()])
            actual=generate_tokens(model,batch,tokenizer,8,decoder='cad',alpha=alpha,prior_prompt='No')
            self.assertTrue(torch.equal(expected,actual))

    def test_cached_scores_match_full_prefix_after_eos_and_padding(self):
        tokenizer=ModelTests.tokenizer;model=self.model
        prior=tokenizer(['No','No'],return_tensors='pt')
        batch=tokenizer(['No effusion','effusion'],padding=True,return_tensors='pt')
        # Finished row0 receives EOS then pads, while row1 keeps decoding.
        scripted=torch.tensor([[0,1,0,0],[0,3,4,1]])
        with torch.inference_mode():
            processor=ContextAwareProcessor(model,prior,.5)
            for length in range(1,5):
                prefix=scripted[:,:length]
                conditional=model(**batch,decoder_input_ids=prefix,use_cache=False).logits[:,-1,:]
                direct_prior=model(**prior,decoder_input_ids=prefix,use_cache=False).logits[:,-1,:]
                actual=processor(prefix,conditional)
                expected=1.5*conditional.float()-.5*direct_prior.float()
                torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)

    def test_generation_handles_controlled_mixed_early_eos(self):
        from unittest.mock import patch
        model=self.model;tokenizer=ModelTests.tokenizer;original=model.forward
        calls=[]
        def scripted_forward(*args,**kwargs):
            result=original(*args,**kwargs)
            result.logits=result.logits.clone()
            if 'cache_position' in kwargs:
                step=len(calls);calls.append(kwargs['decoder_input_ids'].clone())
                result.logits[:,-1,:]=-100
                result.logits[0,-1,1]=100
                result.logits[1,-1,1 if step==2 else 3]=100
            else:
                result.logits[:,-1,:]=0
            return result
        batch=tokenizer(['No effusion','effusion'],padding=True,return_tensors='pt')
        with torch.inference_mode(),patch.object(model,'forward',scripted_forward):
            actual=generate_tokens(model,batch,tokenizer,6,decoder='cad',alpha=.5,prior_prompt='No')
        self.assertEqual(actual.tolist(),[[0,1,0,0],[0,3,3,1]])
        self.assertEqual(len(calls),3)

    def test_actual_beam_and_cad_resume_and_identity_rejection(self):
        helper=PredictionResumeTests()
        for decoder,beams,alpha in [('standard',2,0),('standard',4,0),('cad',1,.5)]:
            with self.subTest(decoder=decoder,beams=beams),tempfile.TemporaryDirectory(dir=ModelTests.path) as folder,redirect_stdout(io.StringIO()):
                root=Path(folder)
                def args(output):
                    a=helper.arguments(root,output)
                    a.decoder,a.beams,a.cad_alpha,a.cad_prior_prompt=decoder,beams,alpha,'No'
                    return a
                predict.main(args(root/'whole'))
                a=args(root/'resume');a.max_minutes=0
                self.assertEqual(predict.main(a)['committed_examples'],2)
                a.resume=True;a.max_minutes=None
                for key,value in [('beams',4 if beams!=4 else 2),('cad_alpha',.25 if decoder=='cad' else .5),('cad_prior_prompt','effusion')]:
                    old=getattr(a,key);setattr(a,key,value)
                    if key=='cad_prior_prompt' and decoder=='standard':
                        setattr(a,key,old);continue
                    with self.assertRaises(ValueError):predict.main(a)
                    setattr(a,key,old)
                predict.main(a)
                self.assertEqual((root/'whole/predictions.jsonl').read_bytes(),(root/'resume/predictions.jsonl').read_bytes())


if __name__=='__main__':unittest.main()
