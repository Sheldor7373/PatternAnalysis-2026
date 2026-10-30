"""Matrix-optimum, scaling/gauge and actual PEFT reload equivalence checks."""
import json
from pathlib import Path
import tempfile
import unittest
import torch
from spectral_adapters import spectral_factors,build_variants
from modules import RadiologySimplifier,load_checkpoint
from checkpoint_storage import mark_complete,file_digests,resolve_checkpoint
from test_model import ModelTests


class SpectralMathTests(unittest.TestCase):
    def test_hard_and_soft_known_spectrum_without_double_scaling(self):
        a=torch.eye(4);b=torch.diag(torch.tensor([8.,4.,2.,1.]))/3.5
        aa,bb,stats=spectral_factors(a,b,3.5,'hard',2)
        torch.testing.assert_close(3.5*bb@aa,torch.diag(torch.tensor([8.,4.,0.,0.])))
        self.assertAlmostEqual(stats['hard_optimal_discarded_energy'],5.,places=5)
        aa,bb,stats=spectral_factors(a,b,3.5,'soft',2)
        self.assertAlmostEqual(stats['threshold'],3.,places=5)
        torch.testing.assert_close(3.5*bb@aa,torch.diag(torch.tensor([5.,1.,0.,0.])))

    def test_product_svd_is_invariant_to_factor_gauge(self):
        torch.manual_seed(3733);a=torch.randn(4,9);b=torch.randn(7,4)
        q,_=torch.linalg.qr(torch.randn(4,4))
        aa,bb,_=spectral_factors(a,b,2.,'hard',2)
        cc,dd,_=spectral_factors(q@a,b@q.T,2.,'hard',2)
        torch.testing.assert_close(2*bb@aa,2*dd@cc,atol=2e-5,rtol=2e-4)

    def test_soft_tie_and_zero_keep_explicit_rank_budget(self):
        a=torch.eye(4);b=torch.diag(torch.tensor([5.,3.,3.,1.]))
        aa,bb,stats=spectral_factors(a,b,1.,'soft',2)
        self.assertTrue(stats['threshold_tie']);self.assertEqual(stats['positive_singular_values_after_shrinkage'],1)
        torch.testing.assert_close(bb@aa,torch.diag(torch.tensor([2.,0.,0.,0.])))
        for method in ['zero','scale_half','full']:
            aa,bb,_=spectral_factors(a,b,1.,method,4)
            expected=torch.zeros_like(b) if method=='zero' else b*.5 if method=='scale_half' else b
            torch.testing.assert_close(bb@aa,expected)


class SpectralCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):ModelTests.setUpClass()
    @classmethod
    def tearDownClass(cls):ModelTests.tearDownClass()

    def test_actual_peft_variants_reload_and_original_is_immutable(self):
        torch.manual_seed(3733)
        with tempfile.TemporaryDirectory(dir=ModelTests.path) as directory:
            root=Path(directory);source=root/'source';model=RadiologySimplifier(str(ModelTests.path),mode='lora',rank=4).eval()
            with torch.no_grad():
                for name,p in model.named_parameters():
                    if 'lora_B' in name:p.normal_(0,.05)
            model.save(source,ModelTests.tokenizer,{'model':str(ModelTests.path),'revision':'main','mode':'lora','rank':4});mark_complete(source)
            before=file_digests(source);result=build_variants(source,root/'derived')
            self.assertEqual(file_digests(source),before);self.assertTrue(result['source_unchanged'])
            tokens=ModelTests.tokenizer(['No effusion','No'],padding=True,return_tensors='pt')
            decoder=torch.tensor([[0,3,4],[0,3,4]])
            with torch.inference_mode():
                expected=model(**tokens,decoder_input_ids=decoder).logits
                full,_,_=load_checkpoint(root/'derived/full_reconstruction',torch.device('cpu'));full.eval()
                torch.testing.assert_close(full(**tokens,decoder_input_ids=decoder).logits,expected,atol=2e-5,rtol=2e-4)
                zero,_,_=load_checkpoint(root/'derived/zero',torch.device('cpu'));zero.eval()
                with model.model.disable_adapter():base=model(**tokens,decoder_input_ids=decoder).logits
                torch.testing.assert_close(zero(**tokens,decoder_input_ids=decoder).logits,base,atol=1e-6,rtol=1e-5)
                for label in ['hard_half','hard_quarter','soft_half','scale_half']:
                    variant,_,_=load_checkpoint(root/'derived'/label,torch.device('cpu'));variant.eval()
                    self.assertTrue(torch.isfinite(variant(**tokens,decoder_input_ids=decoder).logits).all())
            with self.assertRaises(ValueError):build_variants(source,root/'derived')
            self.assertEqual(file_digests(source),before)


if __name__=='__main__':unittest.main()
