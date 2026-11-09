"""Record the measured refusal of the declared stride-1 resolution lever.

`docs/M3_OCCUPANCY_RESOLUTION.md` declares that a lever arm which cannot be admitted inside 1,152 MiB
at the matched budget is NOT ADMISSIBLE AS DECLARED, recorded with its measured peak, and that no
ceiling is raised, no smaller variant substituted and no run repeated to pass. This recorder applies
that rule to the two admission profiles and refuses to write anything else.
"""
import argparse
import json
from pathlib import Path

from probe_model import require_bounded_job

ROOT = Path(__file__).resolve().parents[1]
CEILING_BYTES = 1152*1024**2
CAPTURE_SIDE = 512
LEVER = "m3-occres-lever-r1-profile-r1"
CONTROL = "m3-occres-control-r1-profile-r1"
STUDIES = ("m3-occres-lever-r1", "m3-occres-control-r1")
STATUS_ACCESS_VIOLATION = 3221225477
FORWARD_STUDY = "m3-occres-lever-r1"


def read(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def main():
    bounded = require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("evidence directory already exists")

    lever = read(ROOT/"runs"/(LEVER+"-job.json"))
    control = read(ROOT/"runs"/(CONTROL+"-job.json"))
    if lever.get("status") != "failed" or lever.get("exit_code") != STATUS_ACCESS_VIOLATION:
        raise ValueError("the lever profile was not terminated by the guard at its ceiling")
    if lever.get("memory_bytes") != CEILING_BYTES or not isinstance(lever.get("peak_job_commit_bytes"), int):
        raise ValueError("the lever profile did not run under the declared 1,152 MiB ceiling")
    if lever["peak_job_commit_bytes"] <= CEILING_BYTES:
        raise ValueError("the refusal is only a refusal if the measured peak exceeds the ceiling")
    if (control.get("status") != "passed" or control.get("exit_code") != 0 or
            control.get("memory_bytes") != CEILING_BYTES or
            control["peak_job_commit_bytes"] > CEILING_BYTES):
        raise ValueError("the control profile did not pass inside the same declared ceiling")
    for job in (lever, control):
        if job.get("work_items") != 2 or job.get("wall_seconds") != 300.0:
            raise ValueError("both profiles must be the declared 2-step / 300 s launch")
    for study in STUDIES:
        if (ROOT/"runs"/study/"summary.json").exists():
            raise ValueError(f"{study} was launched after the refusal; the rule forbids it")

    map_bytes = CAPTURE_SIDE**2*4
    census = {"capture_side": CAPTURE_SIDE, "bytes_per_feature_map": map_bytes,
              "branch_feature_maps_retained_for_backward": 4,
              "lower_bound_bytes": 4*48*map_bytes+3*map_bytes,
              "declaration_estimate_bytes": 48*map_bytes}
    delta = lever["peak_job_commit_bytes"]-control["peak_job_commit_bytes"]
    scope = {
        "status": "refused",
        "instrument": "m3-occupancy-resolution-r1",
        "predeclaration": "docs/M3_OCCUPANCY_RESOLUTION.md",
        "declared_receipt": args.receipt,
        "declared_memory_bytes": bounded,
        "rule_applied": ("the predeclaration's own refusal rule: a lever arm that cannot be admitted "
                         "inside 1,152 MiB at the matched budget is not admissible as declared"),
        "ceiling_bytes": CEILING_BYTES,
        "profiles": {
            "lever": {"receipt": f"runs/{LEVER}-job.json", "status": lever["status"],
                      "exit_code": lever["exit_code"], "occupancy_resolution": 1,
                      "peak_job_commit_bytes": lever["peak_job_commit_bytes"],
                      "elapsed_seconds": lever["elapsed_seconds"]},
            "control": {"receipt": f"runs/{CONTROL}-job.json", "status": control["status"],
                        "exit_code": control["exit_code"], "occupancy_resolution": 0,
                        "peak_job_commit_bytes": control["peak_job_commit_bytes"],
                        "elapsed_seconds": control["elapsed_seconds"]},
        },
        "measured": {"peak_delta_bytes": delta,
                     "ceiling_excess_bytes": lever["peak_job_commit_bytes"]-CEILING_BYTES,
                     "estimate_error_bytes": delta-census["declaration_estimate_bytes"],
                     "estimate_understated_by": round(delta/census["declaration_estimate_bytes"], 4)},
        "activation_census": census,
        "mechanism": ("a stride-1 convolution over the full captured image retains four "
                      "full-resolution 48-channel maps through autograd (input, output, grad output, "
                      "grad input), so the branch costs ~190 MiB, four times the ~48 MiB of one map "
                      "the declaration estimated from a single-activation count"),
        "not_done": ["no ceiling raised", "no smaller branch substituted",
                     "no capture side reduced", "no cumulative-cursor variant tried",
                     "no study, sweep or curation launched", "no re-run under a fresh number"],
        "verdict": {"verdict": "refused", "precision_improved": False, "precision_measured": False,
                    "cleared": False,
                    "reason": "the lever arm could not be admitted inside the declared envelope"},
        "population": {"split": "training", "images": 36, "nonempty_boards": 28, "contexts": 12,
                       "note": "declared and unmeasured: no sweep ran"},
        "unchanged": {"thresholds": {"per_cell_f1": .70, "ranking_auc": .90, "ranking_f1": .70},
                      "attribution_floor": .05, "ceiling_bytes": CEILING_BYTES,
                      "steps": 64, "grid_points": 81, "catalogue": "not re-captured",
                      "prior_evidence": "not touched", "frozen_manifest": "not touched"},
        "next": {"step": "stack occupancy_context with occupancy_detail",
                 "reason": (f"the refused lever needed {delta/2**20:.1f} MiB of headroom it did not "
                            "have, while the undeclared 8x8 context tap retains roughly 0.02 MiB of "
                            "activation, so it is the only remaining lever that changes information "
                            "and fits the declared envelope")},
        "open": ["stacking the occupancy_context and occupancy_detail taps remains untested"],
    }
    args.output.mkdir(parents=True)
    with (args.output/"refusal.json").open("x", encoding="utf-8") as stream:
        json.dump(scope, stream, indent=2, allow_nan=False)
    print(json.dumps({key: scope[key] for key in ("measured", "verdict")}, indent=2))


if __name__ == "__main__":
    main()
