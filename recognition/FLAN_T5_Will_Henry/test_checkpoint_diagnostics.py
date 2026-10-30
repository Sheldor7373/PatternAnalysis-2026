import unittest
import torch
from transformers import T5Config, T5ForConditionalGeneration
from checkpoint_diagnostics import row_losses

class ReferenceDiagnosticTests(unittest.TestCase):
    def test_masked_loss_matches_real_model_and_ignores_padding(self):
        torch.manual_seed(11)
        model=T5ForConditionalGeneration(T5Config(vocab_size=31,d_model=16,d_ff=32,num_layers=1,num_decoder_layers=1,num_heads=2,decoder_start_token_id=0,pad_token_id=0,eos_token_id=1,dropout_rate=0)).eval()
        labels=torch.tensor([[2,3,1,-100],[4,5,6,1]])
        out=model(input_ids=torch.tensor([[2,3,1],[4,5,1]]),labels=labels)
        sums,counts=row_losses(out.logits,labels)
        self.assertEqual(counts.tolist(),[3,4])
        self.assertTrue(torch.allclose(sums.sum()/counts.sum(),out.loss,atol=1e-6))
        changed=out.logits.detach().clone();changed[0,3]=1000
        self.assertTrue(torch.equal(row_losses(changed,labels)[0],sums))

    def test_reject_empty_and_nonfinite(self):
        with self.assertRaises(ValueError):row_losses(torch.zeros(1,2,3),torch.full((1,2),-100))
        with self.assertRaises(ValueError):row_losses(torch.full((1,2,3),float('nan')),torch.zeros(1,2,dtype=torch.long))

if __name__=='__main__':unittest.main()
