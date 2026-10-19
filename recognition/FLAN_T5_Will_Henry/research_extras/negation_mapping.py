"""Shared raw-offset policy; never trims offsets or changes declared spans."""

POLICY_VERSION = "whitespace-overhang-v1"


class MappingFailure(ValueError):
    def __init__(self, code, detail):
        super().__init__(detail)
        self.code = code


def map_source_span(text, prompt_length, source_char_span, ids, offsets, attention, special_ids):
    """Map a fixed span, retaining internal whitespace tokens and raw offsets.

    Only a boundary-crossing token with non-whitespace INSIDE the span may
    overhang it, and every outside character must be whitespace. Eligibility
    independently excludes prompt-only tokens, masked tokens and specials.
    """
    def fail(detail):
        raise MappingFailure("span_boundary", detail)

    if (type(prompt_length) is not int or not 0 <= prompt_length < len(text)
            or len(source_char_span) != 2 or any(type(n) is not int for n in source_char_span)):
        fail("Invalid prompt/source character coordinates")
    left, right = source_char_span
    if not 0 <= left < right <= len(text) - prompt_length:
        fail("Invalid fixed source span")
    start, stop = prompt_length + left, prompt_length + right
    if not any(not char.isspace() for char in text[start:stop]):
        fail("A fixed span must contain non-whitespace report text")
    if (not ids or len(ids) != len(offsets) or len(ids) != len(attention)
            or any(type(n) is not int or n < 0 for n in ids)
            or any(type(valid) is not bool for valid in attention)):
        fail("Invalid token IDs/explicit Boolean attention")
    specials, covered, eligible, selected = set(special_ids), set(), [], []
    for index, (token, pair, attended) in enumerate(zip(ids, offsets, attention)):
        if len(pair) != 2 or any(type(n) is not int for n in pair):
            fail("Invalid raw offset pair")
        a, b = pair
        if pair == [0, 0] and token in specials:
            eligible.append(False)
            continue
        if not 0 <= a < b <= len(text):
            fail("Invalid or unexplained zero-width raw offset")
        covered.update(range(a, b))
        source_content = text[max(a, prompt_length):b]
        wholly_source = a >= prompt_length
        whitespace_prefix_only = (a < prompt_length < b and text[a:prompt_length].isspace()
                                  and any(not char.isspace() for char in source_content))
        eligible.append(bool(attended and token not in specials and (wholly_source or whitespace_prefix_only)))
        if max(a, start) < min(b, stop):
            selected.append(index)
    if any(not char.isspace() and index not in covered for index, char in enumerate(text)):
        fail("Raw offsets omit non-whitespace prompt/source characters")
    if not selected or selected != list(range(selected[0], selected[-1] + 1)):
        fail("Selected token indices must be nonempty and contiguous")
    exceptions, span_covered = [], set()
    for index in selected:
        a, b = offsets[index]
        inside = text[max(a, start):min(b, stop)]
        for boundary, outside, char_span in (("start", text[a:start] if a < start else "", [a, start]),
                                             ("stop", text[stop:b] if b > stop else "", [stop, b])):
            if outside:
                if not outside.isspace() or not any(not char.isspace() for char in inside):
                    fail("Boundary overhang requires only outside whitespace and non-whitespace inside the fixed span")
                exceptions.append({"token_index": index, "token_id": ids[index], "raw_offset": list(offsets[index]),
                                   "boundary": boundary, "outside_char_span": char_span, "outside_text": outside,
                                   "inside_span_text": inside})
        if not eligible[index]:
            raise MappingFailure("eligibility", "Selected span includes prompt-only/masked/special/ineligible token")
        span_covered.update(range(max(a, start), min(b, stop)))
    if any(not text[index].isspace() and index not in span_covered for index in range(start, stop)):
        fail("Selected contiguous tokens must cover every non-whitespace character in the fixed span")
    return {"token_span": [selected[0], selected[-1] + 1], "eligible_mask": eligible,
            "mapping": {"policy_version": POLICY_VERSION, "combined_char_span": [start, stop],
                        "boundary_exceptions": exceptions, "raw_offsets_modified": False}}
