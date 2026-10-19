"""Saved-input consistency checks; no model, tokenizer or filesystem access.

Receipts bind declared actual forward inputs to the locked preflight. They do
not independently authenticate a model's execution or supplied log probabilities.
"""
from negation_mapping import POLICY_VERSION
from negation_preflight import PINNED_REVISION, digest

SCORE_RECEIPT_VERSION = "input-bindings-v1"
LOAD_OPTIONS = {"local_files_only": True, "use_fast": True, "trust_remote_code": False}
PINNED_ASSETS = {
    "tokenizer.json": {"bytes": 2424064, "sha256": "fe2ebbbbde2985be723e0ce18217853e4020c5e9d35bd07be2c27ab9d3ead57a"},
    "tokenizer_config.json": {"bytes": 2539, "sha256": "fcde0f79bffda3688119c94330866a8fbf8de20ae65a8c492c9bd47c704655a0"},
    "special_tokens_map.json": {"bytes": 2201, "sha256": "5c87151ef0f72a99d1f766a4c418bd2a1f90aaa30a8e22fe5eca9641daebb64f"},
    "spiece.model": {"bytes": 791656, "sha256": "d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86"},
    "config.json": {"bytes": 1401, "sha256": "439aa0fecf5a5546a1def68b1fc45e538e2c94528ce805378daf091e2bf6e4de"},
}


def validate_tokenizer_provenance(preflight, data_kind):
    identity = preflight["tokenizer"]
    if data_kind == "synthetic_test_scores":
        if identity.get("kind") != "synthetic tokenizer fixture":
            raise ValueError("Synthetic score fixtures require explicitly synthetic tokenizer provenance")
        return
    if data_kind != "saved_model_scores":
        raise ValueError("Unknown score provenance kind")
    if not (identity.get("kind") == "offline cached tokenizer" and identity.get("revision") == PINNED_REVISION
            and digest(identity.get("files")) == digest(PINNED_ASSETS)
            and identity.get("class") == "transformers.models.t5.tokenization_t5_fast.T5TokenizerFast"
            and identity.get("is_fast") is True and digest(identity.get("load_options")) == digest(LOAD_OPTIONS)
            and type(identity.get("eos_token_id")) is int and type(identity.get("pad_token_id")) is int
            and identity.get("eos_token_id") == 1 and identity.get("pad_token_id") == 0
            and preflight["eos_token_id"] == 1 and preflight["pad_token_id"] == 0
            and preflight["special_token_ids"] == [0, 1, 2] + list(range(32000, 32100))):
        raise ValueError("Real scores require the pinned real fast-tokenizer asset/class/options/special-ID receipt; synthetic PASS is insufficient")


def expected_source_input_receipt(source_id, preflight_record):
    """Expected declared ACTUAL source tensors/span; copies avoid fixture aliasing."""
    source = preflight_record["source"]
    result = {"source_id": source_id,
              **{field: list(source[field]) for field in ("input_ids", "attention_mask", "eligible_mask", "token_span")},
              "forward_input_sha256": source["forward_input_sha256"], "mapping_policy_version": POLICY_VERSION,
              "raw_offsets_sha256": digest(source["offset_mapping"]), "span_mapping_sha256": digest(source["span_mapping"])}
    result["bound_input_sha256"] = digest(result)
    return result


def validate_row_binding(row, recipient_id, donor_id, records, checkpoint_id, basis_id):
    expected_recipient = expected_source_input_receipt(recipient_id, records[recipient_id])
    expected_donor = expected_source_input_receipt(donor_id, records[donor_id]) if donor_id is not None else None
    # Canonical JSON comparisons distinguish true from1 and integer IDs from floats.
    if (digest(row.get("recipient_input")) != digest(expected_recipient) or "donor_input" not in row
            or digest(row["donor_input"]) != digest(expected_donor)):
        raise ValueError("Score row recipient/donor actual IDs, masks, eligibility, span or input hash differ from preflight")
    if (row.get("checkpoint_id") != checkpoint_id or "basis_id" not in row or row["basis_id"] != basis_id
            or type(row.get("basis_applied")) is not bool or row["basis_applied"] != (donor_id is not None)):
        raise ValueError("Score row checkpoint/basis binding differs from its envelope/stage")
