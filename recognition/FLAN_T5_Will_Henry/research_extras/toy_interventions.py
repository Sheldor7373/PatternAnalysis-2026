"""Array-only projection-swap toy; no real model hooks, fitting or inference.

For orthonormal columns Q[d,k], replace only the declared subspace in a span:
  patched = base + ((source - base) @ Q) @ Q.T
The orthogonal complement stays at its base value. A caller supplies Q; this
module does not learn or select a subspace. The toy tests implementation, not
a FLAN-T5 mechanism or a formal causal abstraction.
"""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np


def array2d(value, name):
    value = np.asarray(value)
    if value.ndim != 2 or value.dtype.kind not in "fiu" or 0 in value.shape:
        raise ValueError(f"{name} must be a nonempty real numeric 2D array")
    value = value.astype(np.float64, copy=True)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains nonfinite values")
    return value


def span_indices(span, count, mask):
    if (not isinstance(span, (tuple, list)) or len(span) != 2
            or any(type(index) is not int for index in span)
            or not 0 <= span[0] < span[1] <= count):
        raise ValueError("Span must be a nonempty in-bounds half-open (start, stop) pair")
    mask = np.asarray(mask)
    if mask.dtype.kind != "b" or mask.shape != (count,):
        raise ValueError("Require an explicit 1D Boolean valid-token mask")
    if not mask[span[0]:span[1]].all():
        raise ValueError("Intervention span includes masked tokens")
    return slice(*span)


def projection_swap(base, source, basis, *, base_span, source_span,
                    site, source_site, base_mask, source_mask):
    """Patch copies only; base/source span positions are explicitly caller aligned."""
    if not isinstance(site, str) or not site.strip() or source_site != site:
        raise ValueError("Base/source intervention sites must match exactly and be named")
    base, source, basis = array2d(base, "base"), array2d(source, "source"), array2d(basis, "basis")
    width, rank = basis.shape
    if base.shape[1] != source.shape[1] or base.shape[1] != width or rank > width:
        raise ValueError("Base/source feature widths and rank-k basis dimensions must agree")
    if max(base.shape[0], source.shape[0]) > 256 or width > 2048:
        raise ValueError("Toy arrays exceed the restrained CPU size limits")
    if not np.allclose(basis.T @ basis, np.eye(rank), rtol=0, atol=1e-10):
        raise ValueError("Basis columns must already be orthonormal; no implicit fitting or QR")
    base_index = span_indices(base_span, len(base), base_mask)
    source_index = span_indices(source_span, len(source), source_mask)
    if base_span[1] - base_span[0] != source_span[1] - source_span[0]:
        raise ValueError("Base/source spans must have equal token counts")
    with np.errstate(over="raise", invalid="raise"):
        try:
            delta = ((source[source_index] - base[base_index]) @ basis) @ basis.T
            base[base_index] += delta
        except FloatingPointError as error:
            raise ValueError("Projection swap overflowed; use numerically bounded toy arrays") from error
    if not np.isfinite(base).all():
        raise ValueError("Projection swap produced nonfinite values")
    return base


class ToySite:
    """A tiny callback registry, deliberately unrelated to any model framework."""
    def __init__(self, name):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Toy site needs a nonempty name")
        self.name, self._hooks, self._next = name, {}, 0

    @property
    def active_hooks(self):
        return len(self._hooks)

    def register(self, callback):
        if not callable(callback):
            raise TypeError("Toy hook must be callable")
        token, self._next = self._next, self._next + 1
        self._hooks[token] = callback
        return token

    def remove(self, token):
        del self._hooks[token]

    def forward(self, value):
        value = array2d(value, "toy activation")
        shape = value.shape
        for callback in list(self._hooks.values()):
            value = array2d(callback(value.copy()), "toy callback result")
            if value.shape != shape:
                raise ValueError("Toy callback cannot change the activation shape")
        return value


@contextmanager
def temporary_toy_hook(site, callback):
    if type(site) is not ToySite:
        raise TypeError("Only the local ToySite registry is supported; no real model hooks")
    token = site.register(callback)
    try:
        yield site
    finally:
        site.remove(token)


def run_toy():
    causal = np.array([1., 1., 0.]) / np.sqrt(2.)
    nuisance = np.array([1., -1., 0.]) / np.sqrt(2.)
    base = np.array([2 * causal + 5 * nuisance, np.zeros(3)])
    source = np.array([-causal + 9 * nuisance, np.ones(3)])
    options = {"base_span": (0, 1), "source_span": (0, 1), "site": "toy.hidden",
               "source_site": "toy.hidden", "base_mask": [True, True], "source_mask": [True, True]}
    registry = ToySite("toy.hidden")
    with temporary_toy_hook(registry, lambda value: projection_swap(value, source, causal[:, None], **options)):
        patched = registry.forward(base)
    score = lambda value: float(3 * (value[0] @ causal))
    control = projection_swap(base, source, nuisance[:, None], **options)
    return {"scope": "deterministic synthetic array toy; no model run or causal-mechanism claim",
            "rank": 1, "site": "toy.hidden", "base_span": [0, 1], "source_span": [0, 1],
            "base_score": score(base), "source_score": score(source), "patched_score": score(patched),
            "effect": score(patched) - score(base), "expected_effect": -9.,
            "nuisance_control_effect": score(control) - score(base),
            "remaining_toy_hooks": registry.active_hooks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_toy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
