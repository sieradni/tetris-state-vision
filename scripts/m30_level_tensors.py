"""Read the declared M30 level parameters from each committed checkpoint.

`docs/M30_OPERATING_POINT.md` predeclares that the learned scale, bias and calibration gate are
reported, and that a mechanism which could reorder cells is flagged rather than silently accepted.
The scale is `1+tanh(delta)`, positive by construction; the calibration gate is `tanh(g)`, and the
map it drives is monotone inside a board only while the gate is non-negative, so that is checked
here and joined into the scoring receipt.

No forward pass, no training, no weight change.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))

from probe_model import require_bounded_job  # noqa: E402

CEILING_BYTES = 640*1024**2


def weights(run):
    path = ROOT/"runs"/run/"checkpoint"/"weights.pt"
    if not path.is_file():
        raise ValueError(f"{run}: the committed checkpoint is absent")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"{run}: the committed checkpoint is not a tensor mapping")
    return state


def main():
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forms", nargs="+", required=True, help="name:study pairs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    forms = {}
    for spec in args.forms:
        name, run = spec.split(":", 1)
        state = weights(run)
        if name == "bias":
            key = "pixel_query_level"
            if key not in state:
                raise ValueError(f"{run}: the declared level parameter is absent")
            delta, bias = (float(value) for value in state[key].reshape(-1)[:2])
            scale = 1.0+torch.tanh(torch.tensor(delta)).item()
            forms[name] = {"run": run, "mechanism": "learned scale and bias",
                           "scale_delta": round(delta, 6), "bias": round(bias, 6),
                           "scale": round(scale, 6), "ordering_preserved": bool(scale > 0)}
        elif name == "calibrate":
            key = "read_calibration"
            if key not in state:
                raise ValueError(f"{run}: the declared calibration gate is absent")
            gate = torch.tanh(state[key].reshape(-1)[0]).item()
            forms[name] = {"run": run, "mechanism": "gated board standardisation",
                           "gate": round(gate, 6),
                           "ordering_preserved": bool(gate >= 0),
                           "note": ("monotone inside a board while the gate is non-negative; "
                                    "the pooled ordering is reported by the scoring receipt")}
        elif name == "balance":
            forms[name] = {"run": run, "mechanism": "data-derived class budget",
                           "ordering_preserved": True,
                           "note": "the budget changes the loss level only; no tensor is added"}
        else:
            raise ValueError(f"{name}: undeclared form")
    job = json.loads((ROOT/"runs"/(args.output.name+"-job.json")).read_text())
    if job.get("memory_bytes") != CEILING_BYTES:
        raise ValueError("the tensor job ceiling differs from the declared 640 MiB")
    result = {"status": "passed", "instrument": "m30-level-tensors-r1",
              "predeclaration": "docs/M30_OPERATING_POINT.md",
              "ceiling_bytes": CEILING_BYTES,
              "job_name": job.get("job_name"), "forms": forms,
              "scope": ("Read-only scalar read of declared parameters from committed checkpoints; "
                        "no forward pass, no training, no weight change.")}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"scope.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(forms, indent=2))


if __name__ == "__main__":
    main()
