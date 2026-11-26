"""Compare an exported native fixture trace to its saved HF trace. Standard library only."""

import argparse
import json
import math
from pathlib import Path


def compare(label, ref, actual, atol, rtol, require_state=False):
    def ids(x):
        return isinstance(x, list) and all(type(n) is int for n in x)

    generated = actual["generated"]
    if not ids(generated) or generated != ref["generated"]:
        raise ValueError(f"{label}: generated {generated}, expected {ref['generated']}")
    if not ref["steps"] or len(actual["steps"]) != len(ref["steps"]):
        raise ValueError(f"{label}: wrong step count ({len(actual['steps'])} vs {len(ref['steps'])})")
    peaks = {}

    def vector(t, kind, name, exp, got):
        if not isinstance(got, list) or len(got) != len(exp):
            raise ValueError(f"{label}: wrong shape at step {t} {name}")
        for j, (a, b) in enumerate(zip(got, exp)):
            if type(a) not in (float, int) or not math.isfinite(a) or not math.isfinite(b):
                raise ValueError(f"{label}: invalid value at step {t} {name}[{j}]")
            err = abs(a - b)
            peaks[kind] = max(peaks.get(kind, 0.0), err)
            if err > atol + rtol * abs(b):
                raise ValueError(f"{label}: step {t} {name}[{j}] = {a:.9g}, expected {b:.9g} "
                                 f"(abs error {err:.3g})")

    for t, (expected, got) in enumerate(zip(ref["steps"], actual["steps"])):
        n = len(expected["layers"])
        fields = ["attention", "layers"]
        if require_state or "recurrent" in got:
            fields.append("recurrent")
        if any(len(got[field]) != n for field in [*fields, "routes"]):
            raise ValueError(f"{label}: wrong per-layer field length at step {t}")
        for i in range(n):
            routes = got["routes"][i]
            if not ids(routes) or sorted(routes) != sorted(expected["routes"][i]):
                raise ValueError(f"{label}: step {t} layer {i} routes {routes}, "
                                 f"expected {expected['routes'][i]}")
            for field in fields:
                vector(t, field, f"layer {i} {field}", expected[field][i], got[field][i])
        vector(t, "logits", "logits", expected["logits"], got["logits"])
    print(f"PASS {label}: {len(ref['steps'])} steps, routes exact, generated {generated}; "
          "max abs error " + " ".join(f"{k}={v:.3g}" for k, v in peaks.items()))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path, help="saved HF incremental trace for these exact inputs and weights")
    p.add_argument("native", type=Path, help="runner --trace output")
    args = p.parse_args()
    try:
        compare("native vs saved HF", json.loads(args.reference.read_bytes()), json.loads(args.native.read_bytes()),
                1e-4, 2e-5, require_state=True)
    except (OSError, ValueError, KeyError, TypeError) as e:
        p.exit(1, f"Trace comparison failed: {e}\n")
