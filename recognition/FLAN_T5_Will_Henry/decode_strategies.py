"""Bounded deterministic decoding experiments, not a clinical safety filter.

CAD implements Shi et al. (NAACL 2024), Section 2.2:
https://aclanthology.org/2024.naacl-short.69/
softmax((1+alpha)*logits(context, query, prefix)
        -alpha*logits(query, prefix)). The same fine-tuned model and generated
prefix are used for both streams. Here query is the task instruction and context
is the report. No plausibility filter, sampling, teacher model or retraining.
"""
import math
import torch
from transformers import GenerationConfig, LogitsProcessor


def contrast_logits(conditional, prior, alpha):
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError('CAD alpha must be finite and nonnegative')
    if conditional.shape != prior.shape:
        raise ValueError('CAD streams must have identical batch/vocabulary shapes')
    # Float32 subtraction avoids amplifying BF16 rounding in the contrast.
    return (1 + alpha) * conditional.float() - alpha * prior.float()


class ContextAwareProcessor(LogitsProcessor):
    """Independent encoder and KV cache for p(token | instruction, same prefix).

    Restricted to greedy CAD: beam reordering would require reordering this
    private cache as well. Ordinary beams use Transformers' tested beam path.
    A new processor is created for every batch, preventing cross-report state.
    """
    def __init__(self, model, prior_tokens, alpha):
        self.model, self.alpha = model, alpha
        self.mask = prior_tokens['attention_mask']
        self.encoder = model.get_encoder()(**prior_tokens, return_dict=True)
        self.cache = None
        self.prefix_length = 0

    def __call__(self, input_ids, scores):
        if input_ids.shape[0] != self.mask.shape[0]:
            raise ValueError('CAD does not support expanded beam batches')
        if input_ids.shape[1] != self.prefix_length + 1:
            raise ValueError('CAD decoder prefix must advance by one token')
        tokens = input_ids if self.cache is None else input_ids[:, -1:]
        result = self.model(encoder_outputs=self.encoder, attention_mask=self.mask,
                            decoder_input_ids=tokens, past_key_values=self.cache,
                            use_cache=True, return_dict=True)
        self.cache = result.past_key_values
        self.prefix_length = input_ids.shape[1]
        return contrast_logits(scores, result.logits[:, -1, :], self.alpha)


def decoding_settings(decoder='standard', beams=1, alpha=0.0, prior_prompt=''):
    if decoder not in {'standard', 'cad'} or beams < 1:
        raise ValueError('Unknown decoder or nonpositive beam width')
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError('CAD alpha must be finite and nonnegative')
    if decoder == 'standard' and alpha != 0:
        raise ValueError('Nonzero CAD alpha requires --decoder cad')
    if decoder == 'cad' and (beams != 1 or not prior_prompt.strip()):
        raise ValueError('CAD requires beam1 and a nonempty instruction-only prior')
    return {'decoder': decoder, 'cad_alpha': alpha,
            'cad_prior_prompt': prior_prompt if decoder == 'cad' else None,
            'cad_formula': '(1+alpha)*context_logits-alpha*instruction_only_logits' if decoder == 'cad' else None,
            'length_penalty': 1.0, 'early_stopping': False,
            'sampling': False, 'beams': beams}


def generate_tokens(model, tokens, tokenizer, max_new_tokens, *, decoder='standard',
                    beams=1, alpha=0.0, prior_prompt=''):
    decoding_settings(decoder, beams, alpha, prior_prompt)
    # Explicit defaults avoid hidden checkpoint generation options changing CAD
    # scores before our processor. These match the pinned baseline T5 defaults.
    config = GenerationConfig.from_model_config(model.config)
    config.max_new_tokens = max_new_tokens
    config.num_beams = beams
    config.do_sample = False
    config.length_penalty = 1.0
    config.early_stopping = False
    config.use_cache = True
    processors = []
    # Exact alpha0 equivalence includes the same HF generation path and dtype;
    # no needless prior pass or floating-point arithmetic changes the logits.
    if decoder == 'cad' and alpha != 0:
        prior = tokenizer([prior_prompt] * tokens['input_ids'].shape[0],
                          padding=True, truncation=False, return_tensors='pt').to(tokens['input_ids'].device)
        processors.append(ContextAwareProcessor(model, prior, alpha))
    return model.generate(**tokens, generation_config=config, logits_processor=processors)
