"""Read the M28 arms' own-lattice, oracle-registration and pixel reads on one population.

`docs/M28_PIXEL_QUERY.md` declares this reader. It joins each arm's own committed
receipts - the occupancy sweep (own lattice, full deciding population) and the
registration-attribution probe (per-cell logits, labels, known mask and the model-cell ->
target-cell mapping) - plus the pixel-ceiling receipt, and recomputes the three reads M27
defined for each arm. No model, no forward pass, no weight or label change.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))

from gate_power import f1_from_counts  # noqa: E402
from occupancy_sweep import threshold_grid  # noqa: E402
from oracle_pairing_read import matched_reads, pairs, require_close, score  # noqa: E402
from probe_model import require_bounded_job  # noqa: E402

CELL_GATE = 0.70
AUC_GATE = 0.90
ATTRIBUTION_FLOOR = 0.05
PIXEL_FLOOR = 0.90
CEILING_BYTES = 128*1024**2
M27_OWN_LATTICE = 0.543974
M27_PIXEL = 0.987179


def read(path):
    return json.loads(Path(path).read_text())


def arm(name, sweep_name, registration_suffix):
    registration = read(ROOT/"runs"/(name+registration_suffix)/"result.json")
    sweep = read(ROOT/"runs"/sweep_name/"result.json")
    job = read(ROOT/"runs"/(name+"-job.json"))
    if registration.get("status") != "passed" or sweep.get("status") != "passed":
        raise ValueError(f"{name}: a committed receipt this read joins did not pass")
    if job.get("status") != "passed" or job.get("exit_code") != 0:
        raise ValueError(f"{name}: the trained arm did not complete")
    if registration.get("weights_sha256") != sweep.get("weights_sha256"):
        raise ValueError(f"{name}: the sweep and the probe disagree about the measured weights")
    if registration.get("forwards") != 36:
        raise ValueError(f"{name}: the probe is not the 36-board population")
    cases = registration["cases"]
    if len(cases) != 36 or sum(1 for case in cases if any(case["occupied"])) != 28:
        raise ValueError(f"{name}: not the declared deciding population")
    if sum(sum(case["occupied"]) for case in cases) != 667:
        raise ValueError(f"{name}: not the declared occupied-cell count")
    full = score(pairs(cases, pairing="as_is"))
    deciding = sweep["deciding"]
    require_close(f"{name} full F1", full["f1_at_zero"], deciding["rows"][
        [row["threshold"] for row in deciding["rows"]].index(0.0)]["f1"])
    require_close(f"{name} full AUC", full["auc"], deciding["auc"])
    require_close(f"{name} full best F1", full["best"]["f1"], deciding["best"]["f1"])
    return {"registration": registration, "sweep": sweep, "cases": cases, "full": full}


def main():
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--lever", required=True)
    parser.add_argument("--suffix", default="-sweep-r1")
    parser.add_argument("--registration-suffix", default="-registration-r1")
    parser.add_argument("--pixel-ceiling", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arms = {"base": arm(args.base, args.base+args.suffix, args.registration_suffix),
            "lever": arm(args.lever, args.lever+args.suffix, args.registration_suffix)}
    if arms["base"]["registration"]["weights_sha256"] == arms["lever"]["registration"]["weights_sha256"]:
        raise ValueError("the two arms carry identical weights")
    pixel_cases = read(ROOT/"runs"/args.pixel_ceiling/"result.json")["cases"]
    reads = {}
    for name, payload in arms.items():
        matched = matched_reads(payload["cases"], pixel_cases)
        counts = {key: len(rows) for key, rows in matched.items()}
        if len(set(counts.values())) != 1:
            raise ValueError(f"{name}: the three reads are not on one population")
        reads[name] = {"cells": counts["as_is"],
                       "own_lattice": payload["full"],
                       "matched_as_is": score(matched["as_is"]),
                       "oracle": score(matched["oracle"]),
                       "pixel": score(matched["pixel"])}
    base, lever = reads["base"], reads["lever"]
    attribution = lever["own_lattice"]["best"]["f1"]-base["own_lattice"]["best"]["f1"]
    registration_gain = lever["oracle"]["best"]["f1"]-lever["own_lattice"]["best"]["f1"]
    representation_gap = lever["pixel"]["best"]["f1"]-lever["oracle"]["best"]["f1"]
    closed = (lever["matched_as_is"]["best"]["f1"]-M27_OWN_LATTICE)/(M27_PIXEL-M27_OWN_LATTICE)
    ranking = ("informative" if lever["own_lattice"]["auc"] >= AUC_GATE and
               lever["own_lattice"]["best"]["f1"] >= CELL_GATE else "uninformative")
    conditions = {"per_cell_f1": lever["own_lattice"]["f1_at_zero"] >= CELL_GATE,
                  "ranking": ranking == "informative",
                  "attribution": attribution >= ATTRIBUTION_FLOOR}
    cleared = all(conditions.values())
    oracle_delta = lever["oracle"]["best"]["f1"]-base["oracle"]["best"]["f1"]
    matched_delta = lever["matched_as_is"]["best"]["f1"]-base["matched_as_is"]["best"]["f1"]
    if cleared:
        cause = None
    elif oracle_delta >= ATTRIBUTION_FLOOR:
        cause = "placement-only"
    elif lever["pixel"]["best"]["f1"] >= PIXEL_FLOOR:
        cause = "features-unavailable-at-own-lattice"
    else:
        cause = "head-cannot-exploit"
    base_gain = base["oracle"]["best"]["f1"]-base["matched_as_is"]["best"]["f1"]
    base_gap = base["pixel"]["best"]["f1"]-base["oracle"]["best"]["f1"]
    control_below_own = base["oracle"]["best"]["f1"] < base["matched_as_is"]["best"]["f1"]
    within_arm = ("representation" if representation_gap >= registration_gain else "placement")
    note = None
    if control_below_own:
        note = ("the control's oracle read (%.6f) is below its own-lattice read (%.6f), so granting "
                "registration cannot be an upper bound on that arm and the cross-arm oracle delta that "
                "assigns `placement-only` is confounded; the within-arm lever reads assign the residual "
                "to %s (registration gain %.6f against representation gap %.6f)"
                % (base["oracle"]["best"]["f1"], base["matched_as_is"]["best"]["f1"],
                   within_arm, registration_gain, representation_gap))
    result = {
        "status": "passed", "instrument": "m28-pixel-query-r1",
        "predeclaration": "docs/M28_PIXEL_QUERY.md",
        "base_run": args.base, "lever_run": args.lever,
        "declared_change": {"occupancy_pixel_query": [0, 1]},
        "weights_sha256": {"base": arms["base"]["registration"]["weights_sha256"],
                           "lever": arms["lever"]["registration"]["weights_sha256"]},
        "population": {"deciding_boards": 36, "nonempty_boards": 28,
                       "occupied_cells": 667, "empty_cells": 4933,
                       "matched_cells": lever["cells"]},
        "reads": reads,
        "attribution": {"best_swept_delta": round(attribution, 6),
                        "oracle_best_delta": round(oracle_delta, 6),
                        "matched_best_delta": round(matched_delta, 6),
                        "registration_gain_lever": round(registration_gain, 6),
                        "representation_gap_lever": round(representation_gap, 6),
                        "m27_span_closed": round(closed, 6),
                        "m27_own_lattice": M27_OWN_LATTICE, "m27_pixel": M27_PIXEL,
                        "base_registration_gain": round(base_gain, 6),
                        "base_representation_gap": round(base_gap, 6),
                        "control_oracle_below_own_lattice": control_below_own,
                        "residual_within_arm": within_arm,
                        "cause_note": note},
        "gates": {"conditions": conditions, "cell_bar": CELL_GATE, "auc_bar": AUC_GATE,
                  "attribution_floor": ATTRIBUTION_FLOOR, "ranking": ranking},
        "verdict": "cleared" if cleared else "not-cleared",
        "cause": cause,
        "scope": ("Read-only decomposition of one declared lever against its matched control on the "
                  "committed deciding boards; no model forward, no training, no weight, threshold, "
                  "gate or label change, and no accuracy, real-image or complete-state claim."),
    }
    job = read(ROOT/"runs"/(args.output.name+"-job.json"))
    if job.get("memory_bytes") != CEILING_BYTES:
        raise ValueError("the reader ceiling differs from the declared 128 MiB")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"scope.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                                          encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "cause": cause,
                      "attribution": result["attribution"],
                      "base": {k: {"f1": round(v["f1_at_zero"], 6), "best": round(v["best"]["f1"], 6)}
                               for k, v in base.items() if isinstance(v, dict)},
                      "lever": {k: {"f1": round(v["f1_at_zero"], 6), "best": round(v["best"]["f1"], 6)}
                                for k, v in lever.items() if isinstance(v, dict)}}, indent=2))


if __name__ == "__main__":
    main()
