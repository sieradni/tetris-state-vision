"""Read the form-1 projection's amplitude against the head that consumes it (M29).

`docs/M29_PIXEL_QUERY_FORM.md` declares this diagnostic. It loads the committed M28 lever
checkpoint and compares the statistics-to-logit gain with the head's own gain, because both
write into the same 192-wide space the occupancy head reads: a projection that never left the
scale of its initialization cannot move the read however informative its inputs are.

No model, no forward pass, no training, no weight change. Ceiling 640 MiB because a checkpoint
load needs an interpreter that can hold the tensors; the form reader stays at 128 MiB and reads
this receipt.
"""
import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
AMPLITUDE_FLOOR = 0.05
CEILING_BYTES = 640*1024**2
FORM_ONE = "m28-pixelread-lever-r1"


def main():
    from sys import path
    path.insert(0, str(ROOT/"scripts"))
    from probe_model import require_bounded_job
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=FORM_ONE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads((ROOT/"runs"/(args.output.name+"-job.json")).read_text())
    if job.get("memory_bytes") != CEILING_BYTES:
        raise ValueError("the diagnostic ceiling differs from the declared 640 MiB")
    weights = ROOT/"runs"/args.run/"checkpoint"/"weights.pt"
    if not weights.exists():
        raise ValueError("the committed checkpoint is absent")
    state = torch.load(weights, map_location="cpu", weights_only=False)
    tensors = state.get("model", state) if isinstance(state, dict) else None
    if not isinstance(tensors, dict):
        raise ValueError("the committed checkpoint is not a tensor mapping")
    projection = [key for key in tensors if "occupancy_pixel_query" in key and key.endswith("weight")]
    bias = [key for key in tensors if "occupancy_pixel_query" in key and key.endswith("bias")]
    head = [key for key in tensors if key.endswith("occupancy.weight")]
    if not projection or not head:
        raise ValueError("the committed checkpoint does not carry the block and the head")
    weight = tensors[sorted(projection)[-1]].float()
    head_weight = tensors[head[0]].float()
    if head_weight.shape[-1] != weight.shape[0]:
        raise ValueError("the projection writes into a different space than the head reads")
    gain = head_weight@weight
    relative = float(gain.norm())/max(float(head_weight.norm()), 1e-12)
    result = {
        "status": "passed", "instrument": "m29-amplitude-r1",
        "run": args.run, "checkpoint": f"runs/{args.run}/checkpoint/weights.pt",
        "tensors": {"projection": sorted(projection)[-1], "head": head[0]},
        "projection_weight_norm": round(float(weight.norm()), 6),
        "projection_bias_norm": round(float(tensors[sorted(bias)[-1]].float().norm()), 6)
        if bias else 0.0,
        "head_weight_norm": round(float(head_weight.norm()), 6),
        "statistics_to_logit_gain_norm": round(float(gain.norm()), 6),
        "relative_gain": round(relative, 6),
        "amplitude_floor": AMPLITUDE_FLOOR,
        "starved": relative < AMPLITUDE_FLOOR,
        "reading": ("the statistics reach the logit at a negligible fraction of the head's own "
                    "gain, so the block never had the amplitude to move the read" if
                    relative < AMPLITUDE_FLOOR else
                    "the statistics reach the logit at a comparable fraction of the head's own "
                    "gain, so amplitude cannot explain a collapse"),
        "job_name": job.get("job_name"),
        "scope": ("Read-only tensor comparison from one committed checkpoint; no forward pass, no "
                  "training and no weight change."),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"scope.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                                          encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("relative_gain", "starved", "reading",
                                                   "projection_weight_norm")}, indent=1))


if __name__ == "__main__":
    main()
