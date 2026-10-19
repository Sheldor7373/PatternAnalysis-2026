"""Fixed-basis CPU T5 span-mean intervention and teacher-forced candidate sums.

No model loading, generation, fitting or device movement. Callers supply a plain
T5ForConditionalGeneration already frozen/eval on CPU, tokens, masks and Q.
inference_mode is appropriate ONLY for this fixed-basis inference adapter;
future subspace fitting needs a separately reviewed differentiable path. This
module does not train Q. Experiments/causal interpretation require a protocol.
"""

from contextlib import contextmanager
from dataclasses import dataclass

import torch
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

MAX_TOKENS = 64


@dataclass(frozen=True)
class SourceTokens:
    input_ids: torch.Tensor             # [1,T], int64; no tokenizer implicit
    attention_mask: torch.Tensor        # [1,T], bool; independent of token ID
    eligible_mask: torch.Tensor         # [1,T], bool; explicit span eligibility


@dataclass(frozen=True)
class Candidate:
    input_ids: torch.Tensor             # [1,L], int64; includes EOS iff declared
    target_mask: torch.Tensor           # [1,L], bool; valid prefix, right padding
    eos_included: bool


def frozen_cpu_model(model):
    if not isinstance(model, T5ForConditionalGeneration):
        raise TypeError("Initial adapter supports plain T5ForConditionalGeneration only")
    if any(module.training for module in model.modules()):
        raise ValueError("Model and submodules must already be in eval mode")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Model must already be frozen")
    if any(value.device.type != "cpu" for value in list(model.parameters()) + list(model.buffers())):
        raise ValueError("CPU model required; adapter never moves devices")


def token_ids(ids, vocab_size):
    if (not isinstance(ids, torch.Tensor) or ids.device.type != "cpu" or ids.dtype != torch.long
            or ids.ndim != 2 or ids.shape[0] != 1 or not 1 <= ids.shape[1] <= MAX_TOKENS):
        raise ValueError("Require CPU int64 token IDs of shape [1,T], 1<=T<=64")
    if bool(((ids < 0) | (ids >= vocab_size)).any()):
        raise ValueError("Token ID outside model vocabulary")


def boolean_mask(mask, shape):
    if (not isinstance(mask, torch.Tensor) or mask.device.type != "cpu"
            or mask.dtype != torch.bool or tuple(mask.shape) != tuple(shape)):
        raise ValueError("Require an explicit CPU Boolean mask of the expected shape")


def validate_source(source, vocab_size):
    if not isinstance(source, SourceTokens):
        raise TypeError("SourceTokens required")
    token_ids(source.input_ids, vocab_size)
    for mask in (source.attention_mask, source.eligible_mask):
        boolean_mask(mask, source.input_ids.shape)
    if not bool(source.attention_mask.any()) or bool((source.eligible_mask & ~source.attention_mask).any()):
        raise ValueError("Eligibility must be within a nonempty attended source")


def span_slice(span, mask):
    if (not isinstance(span, (tuple, list)) or len(span) != 2
            or any(type(index) is not int for index in span)
            or not 0 <= span[0] < span[1] <= len(mask)):
        raise ValueError("Span must be nonempty, half-open and in bounds")
    if not bool(mask[span[0]:span[1]].all()):
        raise ValueError("Span includes an ineligible or masked token")
    return slice(*span)


def floating_matrix(value, name):
    if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
            or value.dtype not in (torch.float32, torch.float64) or value.ndim != 2
            or 0 in value.shape or not bool(torch.isfinite(value).all())):
        raise ValueError(f"{name} must be a finite CPU float32/float64 matrix")


def validate_basis(basis, width, dtype):
    floating_matrix(basis, "basis")
    if basis.requires_grad or basis.shape[0] != width or not 1 <= basis.shape[1] <= width or basis.dtype != dtype:
        raise ValueError("Require a fixed basis with matching width/dtype and 1<=rank<=width")
    q = basis.to(torch.float64)
    tolerance = 1e-6 if dtype == torch.float32 else 1e-10
    if not torch.allclose(q.T @ q, torch.eye(q.shape[1], dtype=q.dtype), atol=tolerance, rtol=0):
        raise ValueError("Basis columns must already be orthonormal; no fitting or implicit QR")


@torch.inference_mode()
def span_mean_shift(recipient, donor, basis, *, recipient_span, donor_span,
                    recipient_mask, donor_mask):
    floating_matrix(recipient, "recipient")
    floating_matrix(donor, "donor")
    if donor.shape[1] != recipient.shape[1] or donor.dtype != recipient.dtype:
        raise ValueError("Donor/recipient width and dtype must match")
    validate_basis(basis, recipient.shape[1], recipient.dtype)
    boolean_mask(recipient_mask, (len(recipient),))
    boolean_mask(donor_mask, (len(donor),))
    sr, sd = span_slice(recipient_span, recipient_mask), span_slice(donor_span, donor_mask)
    # Unequal span lengths are intentional. Every recipient token gets ONE shift.
    delta = (donor[sd].mean(0) - recipient[sr].mean(0)) @ basis @ basis.T
    patched = recipient.clone()
    patched[sr] += delta
    if not bool(torch.isfinite(patched).all()):
        raise ValueError("Nonfinite span-mean intervention result")
    return patched


@contextmanager
def scoped_hook(module, callback):
    handle = module.register_forward_hook(callback)
    try:
        yield
    finally:
        handle.remove()  # Remove only our handle; preserve unrelated hooks.


def validate_candidate(candidate, model):
    if not isinstance(candidate, Candidate) or type(candidate.eos_included) is not bool:
        raise TypeError("Candidate with explicit Boolean eos_included required")
    token_ids(candidate.input_ids, model.config.vocab_size)
    boolean_mask(candidate.target_mask, candidate.input_ids.shape)
    length = int(candidate.target_mask.sum())
    expected = torch.arange(candidate.input_ids.shape[1])[None, :] < length
    if length == 0 or not torch.equal(candidate.target_mask, expected):
        raise ValueError("Candidate mask must be a nonempty valid prefix with right padding")
    eos = model.config.eos_token_id
    if type(eos) is not int:
        raise ValueError("Initial adapter requires one configured EOS token ID")
    valid = candidate.input_ids[0, :length]
    eos_count = int(valid.eq(eos).sum())
    if eos_count > 1 or (eos_count == 1 and int(valid[-1]) != eos) or bool(eos_count) != candidate.eos_included:
        raise ValueError("EOS declaration must match exactly one final valid EOS, or none")
    return length


@torch.inference_mode()
def candidate_logits(model, encoded, source_mask, candidate):
    frozen_cpu_model(model)
    validate_candidate(candidate, model)
    hidden = encoded.last_hidden_state
    if hidden.ndim != 3 or hidden.shape[0] != 1:
        raise ValueError("Encoded source must have shape [1,T,d]")
    floating_matrix(hidden[0], "encoded source")
    boolean_mask(source_mask, hidden.shape[:2])
    if hidden.shape[2] != model.config.d_model or not bool(source_mask.any()):
        raise ValueError("Encoded width must match the model and source mask must attend a token")
    labels = candidate.input_ids.masked_fill(~candidate.target_mask, -100)
    decoder_ids = model.prepare_decoder_input_ids_from_labels(labels=labels)
    # T5 decoder start often equals pad_token_id. Its mask MUST still be one.
    decoder_mask = torch.cat((torch.ones((1, 1), dtype=torch.bool), candidate.target_mask[:, :-1]), dim=1)
    result = model(encoder_outputs=encoded, attention_mask=source_mask,
                   decoder_input_ids=decoder_ids, decoder_attention_mask=decoder_mask,
                   use_cache=False, return_dict=True)
    if not bool(torch.isfinite(result.logits).all()):
        raise ValueError("Nonfinite candidate logits")
    return result.logits


@torch.inference_mode()
def score_candidate(model, encoded, source_mask, candidate):
    logits = candidate_logits(model, encoded, source_mask, candidate)
    values = logits.to(torch.float64).log_softmax(-1).gather(-1, candidate.input_ids.unsqueeze(-1)).squeeze(-1)
    return {"summed_logprob": float(values.masked_fill(~candidate.target_mask, 0).sum()),
            "token_count": int(candidate.target_mask.sum()), "eos_included": candidate.eos_included,
            "aggregation": "sum; padding excluded; no length normalisation"}


class T5SpanMeanAdapter:
    def __init__(self, model, site, basis):
        frozen_cpu_model(model)
        sites = {f"encoder.block.{i}": block for i, block in enumerate(model.encoder.block)}
        if not isinstance(site, str) or site not in sites:
            raise ValueError("Site must name an existing encoder.block.<index> output")
        validate_basis(basis, model.config.d_model, model.shared.weight.dtype)
        self.model, self.site, self.block = model, site, sites[site]
        self.basis = basis.detach().clone()

    def _encode(self, source):
        return self.model.encoder(input_ids=source.input_ids, attention_mask=source.attention_mask,
                                  use_cache=False, return_dict=True)

    @torch.inference_mode()
    def encode_pair(self, recipient, donor, *, recipient_span, donor_span):
        frozen_cpu_model(self.model)
        for source in (recipient, donor):
            validate_source(source, self.model.config.vocab_size)
        span_slice(recipient_span, recipient.eligible_mask[0])
        span_slice(donor_span, donor.eligible_mask[0])
        captures = []

        def hidden_output(output):
            if not isinstance(output, tuple) or not output or not isinstance(output[0], torch.Tensor):
                raise ValueError("Unsupported T5 block output; expected hidden states first in a tuple")
            if output[0].ndim != 3 or output[0].shape[0] != 1:
                raise ValueError("Expected a single-example T5 hidden-state batch")
            return output[0]

        def capture(module, inputs, output):
            captures.append(hidden_output(output).detach().clone())

        with scoped_hook(self.block, capture):
            self._encode(donor)
        if len(captures) != 1:
            raise ValueError("Donor site must execute exactly once")
        baseline = self._encode(recipient)
        calls = []

        def patch(module, inputs, output):
            hidden = hidden_output(output)
            calls.append(1)
            shifted = span_mean_shift(hidden[0], captures[0][0], self.basis,
                                      recipient_span=recipient_span, donor_span=donor_span,
                                      recipient_mask=recipient.eligible_mask[0], donor_mask=donor.eligible_mask[0])
            return (shifted.unsqueeze(0),) + output[1:]

        with scoped_hook(self.block, patch):
            patched = self._encode(recipient)
        if len(calls) != 1:
            raise ValueError("Recipient site must execute exactly once")
        # Encoder hooks are gone before any decoder candidate scoring occurs.
        return (BaseModelOutput(last_hidden_state=baseline.last_hidden_state),
                BaseModelOutput(last_hidden_state=patched.last_hidden_state))

    @torch.inference_mode()
    def compare(self, recipient, donor, candidate_a, candidate_b, *, recipient_span, donor_span):
        for candidate in (candidate_a, candidate_b):
            validate_candidate(candidate, self.model)
        baseline, patched = self.encode_pair(recipient, donor, recipient_span=recipient_span, donor_span=donor_span)
        scores = {}
        for name, encoded in (("baseline", baseline), ("patched", patched)):
            a = score_candidate(self.model, encoded, recipient.attention_mask, candidate_a)
            b = score_candidate(self.model, encoded, recipient.attention_mask, candidate_b)
            scores[name] = {"candidate_a": a, "candidate_b": b,
                            "margin": a["summed_logprob"] - b["summed_logprob"]}
        return {"site": self.site, "rank": self.basis.shape[1],
                "recipient_span": list(recipient_span), "donor_span": list(donor_span), **scores,
                "effect": scores["patched"]["margin"] - scores["baseline"]["margin"],
                "scope": "fixed-basis CPU intervention scores; no fitting or causal-mechanism claim"}
