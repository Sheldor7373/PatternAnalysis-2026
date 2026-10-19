"""Bounded inference-only pilot then substantive FIT screen; API only, no loaders.

No CLI, generation, fitting, checkpoint selection, selection/heldout scoring,
downloads, device changes or output writes. Real execution requires separate
authorisation and checkpoint preparation. Tests use tiny random CPU fixtures.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict

import torch

from negation_mapping import POLICY_VERSION
from negation_model_copies import frozen_float32_cpu, state_fingerprint
from negation_preflight import MODELS, digest, token_record, validate_contract
from negation_receipts import SCORE_RECEIPT_VERSION, validate_row_binding, validate_tokenizer_provenance
from negation_summary import AGGREGATION, prefit_gate, summarise, validate_preflight
from t5_intervention_adapter import (Candidate, SourceTokens, T5SpanMeanAdapter,
                                     validate_candidate, validate_source)

# Encoder and decoder MODULE calls, not generated tokens or per-token forwards.
LIMITS = {"pilot": {"frozen": {"encoder": 36, "decoder": 48},
                     "lora_merged_copy": {"encoder": 40, "decoder": 56},
                     "wrapped_reference": {"encoder": 4, "decoder": 8}},
          "substantive": {"frozen": {"encoder": 16, "decoder": 32},
                           "lora_merged_copy": {"encoder": 16, "decoder": 32},
                           "wrapped_reference": {"encoder": 0, "decoder": 0}}}


class StopScreen(RuntimeError):
    pass


class ForwardBudget:
    def __init__(self, copies):
        self.models = {"frozen": copies.frozen, "lora_merged_copy": copies.merged,
                       "wrapped_reference": copies.wrapped_reference}
        self.phase = "pilot"
        self.counts = {phase: {role: {kind: 0 for kind in limits} for role, limits in roles.items()}
                       for phase, roles in LIMITS.items()}

    @contextmanager
    def tracked(self):
        handles = []
        try:
            for role, model in self.models.items():
                for kind, module in (("encoder", model.get_encoder()), ("decoder", model.get_decoder())):
                    def count(module, args, role=role, kind=kind):
                        if self.counts[self.phase][role][kind] >= LIMITS[self.phase][role][kind]:
                            raise StopScreen("Forward budget exhausted before " + self.phase + "/" + role + "/" + kind)
                        self.counts[self.phase][role][kind] += 1
                    handles.append(module.register_forward_pre_hook(count))
            yield self
        finally:
            for handle in handles:
                handle.remove()

    def complete(self, phase):
        if self.counts[phase] != LIMITS[phase]:
            raise StopScreen("Completed stage differs from its exact declared forward budget")


def _bundle(record):
    source = record["source"]
    tokens = SourceTokens(torch.tensor([source["input_ids"]], dtype=torch.long),
                          torch.tensor([source["attention_mask"]], dtype=torch.bool),
                          torch.tensor([source["eligible_mask"]], dtype=torch.bool))
    candidates = {name: Candidate(torch.tensor([candidate["input_ids"]], dtype=torch.long),
                                 torch.tensor([candidate["target_mask"]], dtype=torch.bool), candidate["eos_included"])
                  for name, candidate in record["candidates"].items()}
    return tokens, list(source["token_span"]), candidates


def actual_source_receipt(source_id, source, span, record):
    """Read the ACTUAL SourceTokens and span passed into encoding/intervention.

    Offset/policy metadata comes from the locked mapping; tensor values, masks,
    forward fingerprint and span come from the call arguments, never expected
    preflight input receipts. Validation separately compares these declarations.
    """
    for value, dtype in ((source.input_ids, torch.long), (source.attention_mask, torch.bool),
                         (source.eligible_mask, torch.bool)):
        if value.device.type != "cpu" or value.dtype != dtype or value.ndim != 2 or value.shape[0] != 1:
            raise ValueError("Actual receipt requires single-row CPU long IDs and Boolean masks")
    ids, attention, eligible = [value.detach()[0].tolist() for value in
                                (source.input_ids, source.attention_mask, source.eligible_mask)]
    fingerprint = token_record(ids, attention, eligible)
    result = {"source_id": source_id, "input_ids": ids, "attention_mask": attention,
              "eligible_mask": eligible, "token_span": list(span),
              "forward_input_sha256": fingerprint["forward_input_sha256"], "mapping_policy_version": POLICY_VERSION,
              "raw_offsets_sha256": digest(record["source"]["offset_mapping"]),
              "span_mapping_sha256": digest(record["source"]["span_mapping"])}
    result["bound_input_sha256"] = digest(result)
    return result


def _basis(model):
    basis = torch.zeros((model.config.d_model, 1), dtype=torch.float32)
    basis[0, 0] = 1
    return basis


def _basis_id(basis):
    return digest({"kind": "fixed_coordinate_e0", "dtype": str(basis.dtype), "shape": list(basis.shape),
                   "values": basis.detach().tolist()})


def _binding(data, model, source_id, source, span, records, *, case_id=None, donor_id=None, donor=None, donor_span=None):
    row = {"model": model, "source_id" if case_id is None else "case_id": source_id if case_id is None else case_id,
           "recipient_input": actual_source_receipt(source_id, source, span, records[source_id]),
           "donor_input": actual_source_receipt(donor_id, donor, donor_span, records[donor_id]) if donor_id is not None else None,
           "checkpoint_id": data["checkpoint_ids"][model], "basis_id": data["basis_ids"][model] if data["stage"] == "pilot" else None,
           "basis_applied": case_id is not None}
    validate_row_binding(row, source_id, donor_id, records, row["checkpoint_id"], row["basis_id"])
    return row


@torch.inference_mode()
def _candidate_score(model, encoded, source, candidate, *, source_id, name):
    """Ordinary plain/wrapped teacher forcing; mask-aware sums and actual tokens."""
    frozen_float32_cpu(model, check_finite=False)  # State locks are checked at boundaries; do not scan77M weights per candidate.
    validate_candidate(candidate, model)
    before_ids, before_mask = candidate.input_ids.clone(), candidate.target_mask.clone()
    labels = candidate.input_ids.masked_fill(~candidate.target_mask, -100)
    decoder_ids = model.prepare_decoder_input_ids_from_labels(labels=labels)
    decoder_mask = torch.cat((torch.ones((1,1), dtype=torch.bool), candidate.target_mask[:, :-1]), dim=1)
    logits = model(encoder_outputs=encoded, attention_mask=source.attention_mask, decoder_input_ids=decoder_ids,
                   decoder_attention_mask=decoder_mask, use_cache=False, return_dict=True).logits
    if (logits.dtype != torch.float32 or tuple(logits.shape) != (*candidate.input_ids.shape, model.config.vocab_size)
            or not bool(torch.isfinite(logits).all())):
        raise StopScreen("Invalid/nonfinite teacher-forced logits")
    if not torch.equal(before_ids, candidate.input_ids) or not torch.equal(before_mask, candidate.target_mask):
        raise StopScreen("Candidate call tensors mutated during scoring")
    values = logits.double().log_softmax(-1).gather(-1, candidate.input_ids.unsqueeze(-1)).squeeze(-1)
    ids, mask = before_ids[0].tolist(), before_mask[0].tolist()
    return {"summed_logprob": float(values.masked_fill(~candidate.target_mask, 0).sum()),
            "token_count": sum(mask), "eos_included": candidate.eos_included, "aggregation": AGGREGATION,
            "token_ids_sha256": digest(ids), "target_mask_sha256": digest(mask),
            "actual_input_ids": ids, "actual_target_mask": mask}, logits.detach()


def _score_pair(model, encoded, source, candidates, source_id):
    scores, logits = {}, {}
    for name in ("positive", "negative"):
        scores[name], logits[name] = _candidate_score(model, encoded, source, candidates[name], source_id=source_id, name=name)
    return scores, logits


def _normalised_error(reference, actual, candidates, atol, rtol):
    errors = []
    for name in ("positive", "negative"):
        mask = candidates[name].target_mask
        a, b = reference[name][mask].double(), actual[name][mask].double()
        if a.shape != b.shape or not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
            raise StopScreen("Invalid comparison logits")
        errors.append(float(((a-b).abs() / (atol + rtol*a.abs())).max()))
    return max(errors)


def _margin(scores):
    return scores["positive"]["summed_logprob"] - scores["negative"]["summed_logprob"]


def _envelope(manifest, preflight, copies, stage):
    return {"data_kind": "synthetic_test_scores" if copies.fixture else "saved_model_scores",
            "score_receipt_version": SCORE_RECEIPT_VERSION, "protocol_sha256": digest(manifest),
            "preflight_sha256": digest(preflight), "stage": stage, "basis_condition": "coordinate_e0" if stage == "pilot" else "none",
            "checkpoint_ids": copies.checkpoint_ids, "basis_ids": {role: _basis_id(_basis(model)) for role, model in
                    (("frozen", copies.frozen), ("lora_merged_copy", copies.merged))} if stage == "pilot" else None,
            "checkpoint_lock": asdict(copies.lock), "baselines": [], "patches": []}


def _pilot_model(data, model, role, source_ids, cases, records):
    before = data["checkpoint_ids"][role]  # Already measured by the stage-entry immutable-state check.
    adapter = T5SpanMeanAdapter(model, "encoder.block.3", _basis(model))
    evidence = data["mechanical_evidence"].setdefault(role, {})
    evidence.update(parameters_before_sha256=before, self_logits=[])
    baseline_rows = {}
    jobs = [("self:"+identity, identity, identity) for identity in sorted(source_ids)] + [
            (case["id"], case["recipient_id"], case["donor_id"]) for case in cases]
    for case_id, recipient_id, donor_id in jobs:
        source, span, candidates = _bundle(records[recipient_id])
        donor, donor_span, _ = _bundle(records[donor_id])
        original_row = _binding(data, role, recipient_id, source, span, records)
        patched_row = _binding(data, role, recipient_id, source, span, records, case_id=case_id,
                               donor_id=donor_id, donor=donor, donor_span=donor_span)
        encoded, patched = adapter.encode_pair(source, donor, recipient_span=span, donor_span=donor_span)
        baseline_scores, baseline_logits = _score_pair(model, encoded, source, candidates, recipient_id)
        patched_scores, patched_logits = _score_pair(model, patched, source, candidates, recipient_id)
        # Re-read actual arguments after calls; mutation invalidates this comparison.
        if (original_row != _binding(data, role, recipient_id, source, span, records)
                or patched_row != _binding(data, role, recipient_id, source, span, records, case_id=case_id,
                                          donor_id=donor_id, donor=donor, donor_span=donor_span)
                or _basis_id(adapter.basis) != data["basis_ids"][role]):
            raise StopScreen("Actual source/donor/span/basis arguments changed during comparison")
        original_row.update(baseline_scores); patched_row.update(patched_scores)
        if recipient_id in baseline_rows and original_row != baseline_rows[recipient_id]:
            raise StopScreen("Repeated baseline scores/inputs are inconsistent")
        if recipient_id not in baseline_rows:
            baseline_rows[recipient_id] = original_row; data["baselines"].append(original_row)
        data["patches"].append(patched_row)
        if case_id.startswith("self:"):
            error = _normalised_error(baseline_logits, patched_logits, candidates, 1e-6, 1e-6)
            evidence["self_logits"].append({"source_id": recipient_id, "max_normalised_error": error, "atol": 1e-6, "rtol": 1e-6})
            if error > 1 or abs(_margin(patched_scores)-_margin(baseline_scores)) > 1e-6:
                raise StopScreen("Self-patch logits or absolute margin effect failed tolerance")
    evidence["parameters_after_sha256"] = state_fingerprint(model)
    if evidence["parameters_after_sha256"] != before:
        raise StopScreen("Checkpoint changed during pilot")


def _merge_equivalence(data, copies, source_ids, records):
    evidence = data["mechanical_evidence"].setdefault("lora_merged_copy", {})
    evidence["merge_equivalence"] = []
    for identity in sorted(source_ids):
        source, span, candidates = _bundle(records[identity])
        actual_input = _binding(data, "lora_merged_copy", identity, source, span, records)["recipient_input"]
        scores, logits = [], []
        for model in (copies.wrapped_reference, copies.merged):
            encoded = model.get_encoder()(input_ids=source.input_ids, attention_mask=source.attention_mask, use_cache=False, return_dict=True)
            score, logit = _score_pair(model, encoded, source, candidates, identity)
            scores.append(score); logits.append(logit)
        if actual_input != actual_source_receipt(identity, source, span, records[identity]):
            raise StopScreen("Merge-equivalence source tensors changed")
        error = _normalised_error(logits[0], logits[1], candidates, 1e-5, 1e-5)
        reference_margin, merged_margin = [_margin(score) for score in scores]
        evidence["merge_equivalence"].append({"source_id": identity, "max_normalised_logit_error": error,
            "reference_margin": reference_margin, "merged_margin": merged_margin, "atol": 1e-5, "rtol": 1e-5,
            "recipient_input": actual_input, "reference_checkpoint_id": copies.lock.wrapped_state_sha256,
            "merged_checkpoint_id": copies.lock.merged_state_sha256, "reference_scores": scores[0], "merged_scores": scores[1]})
        if error > 1 or abs(reference_margin-merged_margin) > 1e-5 + 1e-5*abs(reference_margin):
            raise StopScreen("Separate-copy LoRA merge equivalence failed")


def _substantive(data, copies, source_ids, records):
    for role, model in (("frozen", copies.frozen), ("lora_merged_copy", copies.merged)):
        for identity in sorted(source_ids):
            source, span, candidates = _bundle(records[identity])
            row = _binding(data, role, identity, source, span, records)
            encoded = model.get_encoder()(input_ids=source.input_ids, attention_mask=source.attention_mask, use_cache=False, return_dict=True)
            scores, _ = _score_pair(model, encoded, source, candidates, identity)
            if row != _binding(data, role, identity, source, span, records):
                raise StopScreen("Substantive actual source inputs changed")
            row.update(scores); data["baselines"].append(row)


@torch.inference_mode()
def run_screen(manifest, preflight, copies, *, include_substantive=True):
    """Run only the fixed pilot, then fit-only screen IF its pilot gate passes.

    API accepts already prepared locked copies, never a loader/path/live model.
    All unexpected failures stop; partial receipts/counts remain diagnostic only.
    """
    if type(include_substantive) is not bool:
        raise ValueError("Explicit Boolean include_substantive required")
    if torch.is_autocast_enabled("cpu"):
        raise ValueError("CPU autocast must be disabled for the fixed float32 protocol")
    manifest, preflight = deepcopy(manifest), deepcopy(preflight)  # Read immutable local snapshots during all calls.
    copies.lock.validate(copies.fixture)
    sources, cases = validate_contract(manifest)
    records = validate_preflight(manifest, preflight)
    validate_tokenizer_provenance(preflight, "synthetic_test_scores" if copies.fixture else "saved_model_scores")
    copies.assert_unchanged()
    for model in (copies.frozen, copies.wrapped_reference, copies.merged):
        if (model.config.eos_token_id != preflight["eos_token_id"] or model.config.pad_token_id != preflight["pad_token_id"]
                or model.config.decoder_start_token_id != preflight["pad_token_id"] or len(model.get_encoder().block) < 4):
            raise ValueError("Model configuration does not match fixed tokenizer/site contract")
        if not copies.fixture and (model.config.d_model != 512 or len(model.get_encoder().block) != 8 or model.config.num_decoder_layers != 8):
            raise ValueError("Real execution requires the pinned512-wide8/8-layer T5 architecture")
        for record in records.values():
            if record["split"] != "fit":
                continue  # No selection/heldout tensors are built or scored.
            source, _, candidates = _bundle(record)
            validate_source(source, model.config.vocab_size)
            for candidate in candidates.values(): validate_candidate(candidate, model)
    pilot_ids = set(manifest["screened_pilot"]["recipient_ids"])
    substantive_ids = set(manifest["substantive_signal_before_fitting"]["panel_source_ids"])
    expected_pilot = {f"fit/negation-01/{finding}/{polarity}" for finding in ("effusion", "pneumothorax") for polarity in ("absent", "present")}
    expected_substantive = {f"fit/negation-{family:02d}/{finding}/{polarity}" for family in range(1,5)
                            for finding in ("effusion", "pneumothorax") for polarity in ("absent", "present")}
    if pilot_ids != expected_pilot or substantive_ids != expected_substantive or any(sources[i]["split"] != "fit" for i in pilot_ids | substantive_ids):
        raise ValueError("Only fixed4-case pilot and16-case fit-only substantive panel are allowed")
    pilot_cases = [c for c in cases.values() if c["recipient_id"] in pilot_ids]
    if len(pilot_cases) != 8:
        raise ValueError("Pilot must contain exactly eight fixed cross-finding conditions")
    pilot = _envelope(manifest, preflight, copies, "pilot"); pilot["mechanical_evidence"] = {}
    result = {"status": "STOP", "pilot_scores": pilot, "pilot_summary": None, "substantive_scores": None,
              "substantive_summary": None, "prefit_gate": None, "failure": None, "fitting_authorised": False,
              "checkpoint_lock": asdict(copies.lock), "declared_limits": deepcopy(LIMITS)}
    budget = ForwardBudget(copies)
    try:
        with budget.tracked():
            _pilot_model(pilot, copies.frozen, "frozen", pilot_ids, pilot_cases, records)
            _merge_equivalence(pilot, copies, pilot_ids, records)
            _pilot_model(pilot, copies.merged, "lora_merged_copy", pilot_ids, pilot_cases, records)
            budget.complete("pilot")
            copies.assert_unchanged()
            result["pilot_summary"] = summarise(manifest, preflight, pilot)
            if result["pilot_summary"]["gate"]["status"] != "PASS":
                result["failure"] = "Pilot gate failed; substantive screen not executed"
            elif not include_substantive:
                result["status"] = "PILOT_COMPLETE"
            else:
                budget.phase = "substantive"
                substantive = _envelope(manifest, preflight, copies, "substantive")
                result["substantive_scores"] = substantive
                _substantive(substantive, copies, substantive_ids, records)
                budget.complete("substantive")
                copies.assert_unchanged()
                result["substantive_summary"] = summarise(manifest, preflight, substantive)
                result["prefit_gate"] = prefit_gate(result["pilot_summary"], result["substantive_summary"])
                result["status"] = "REVIEW_REQUIRED" if result["prefit_gate"]["status"] == "PASS" else "STOP"
                if result["status"] == "STOP": result["failure"] = "Substantive gate failed; return full panel for review"
    except Exception as error:
        result["status"] = "STOP"; result["failure"] = type(error).__name__ + ": " + str(error)
    finally:
        result["forward_counts"] = budget.counts
        try:
            copies.assert_unchanged()
            result["immutable_state_check"] = "PASS"
        except Exception as error:
            result["status"] = "STOP"; result["immutable_state_check"] = "FAIL: " + str(error)
            result["failure"] = "Immutable state/config check failed; no continuation"
    return result
