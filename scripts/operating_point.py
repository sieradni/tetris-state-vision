"""Read-only audit: is a threshold-0.0 attribution a model difference or a moved boundary?

`docs/M3_OPERATING_POINT.md` states the rule; this module is its single implementation. Four
reads are recomputed from the per-cell logits every sweep receipt already commits - `f1_at_zero`,
`auc`, an out-of-fold-threshold `cv_f1` and the logit offset - and each declared pair is classified
by the predeclared state table. No training, no weight change, no gate, no threshold for any gate.

The out-of-fold read keeps the sweep's own grid, criterion and tie-break but selects the threshold
on the other eleven ancestry groups, so the boundary never sees the cell it scores. It is invariant
to a monotone shift of an arm's logits, which is the property a calibration-free attribution needs.
"""
import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path

from occupancy_sweep import pooled_auc, threshold_grid
from probe_model import require_bounded_job

ROOT = Path(__file__).resolve().parents[1]
ATTRIBUTION_FLOOR = 0.05
DECIDING_GROUPS, DECIDING_CASES, OCCUPIED_CELLS, KNOWN_CELLS = 12, 36, 667, 200
BOOTSTRAP_SEED, BOOTSTRAP_DRAWS = 20261026, 1000
Z_95 = 1.959963984540054
AUDIT_SOURCES = tuple("scripts/"+name for name in
                      ("operating_point.py", "occupancy_sweep.py", "probe_model.py"))

# (pair, control run, lever run); resolved to `runs/<run>-sweep-r1/result.json`.
PAIRS = (
    ("representation", "m3-occctx-control-r1", "m3-occctx-lever-r1"),
    ("precision", "m3-occctx-control-r1", "m3-occdetail-lever-r1"),
    ("stencil", "m3-occdetail-lever-r1", "m3-occdstencil-lever-r1"),
    ("stacking", "m3-occdstencil-lever-r1", "m3-occstack-lever-r2"),
    ("scale", "m3-occscale-control-r1", "m3-occscale-lever-r1"),
    ("fit_budget", "m3-occfit-baseline-r1", "m3-occfit-lever-r1"),
    ("geometry_stage", "m3-occgeom-baseline-r1", "m3-occgeom-lever-r1"),
    ("translation", "m3-occorigin-baseline-r1", "m3-occorigin-lever-r1"),
    ("extent", "m3-occextent-baseline-r1", "m3-occextent-lever-r1"),
    ("support_anchors", "m3-occgeocap-baseline-r1", "m3-occgeocap-lever-r1"),
    ("lattice_residual", "m3-occlatres-baseline-r1", "m3-occlatres-lever-r1"),
    ("lattice_free", "m3-occlatfree-baseline-r1", "m3-occlatfree-lever-r1"),
    ("board_refinement", "m3-occbref-baseline-r1", "m3-occbref-lever-r1"),
    ("schedule", "m3-occsched-constant-r1", "m3-occsched-cosine-r1"),
)


def digest(path, maximum=64*1024**2):
    value, consumed = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            consumed += len(block)
            if consumed > maximum:
                raise ValueError("audit source byte budget exceeded")
            value.update(block)
    return value.hexdigest()


def require_cases(cases):
    """One split's saved boards: paired logits and binary labels over known cells."""
    if type(cases) is not list or not cases:
        raise ValueError("at least one saved board is required")
    for case in cases:
        if (type(case) is not dict or not isinstance(case.get("logits"), list) or
                not isinstance(case.get("occupied"), list) or
                len(case["logits"]) != len(case["occupied"]) or not case["logits"] or
                any(type(value) is not int or value not in (0, 1) for value in case["occupied"]) or
                any(type(value) not in (int, float) or value != value for value in case["logits"])):
            raise ValueError("saved board logits and labels required")


def counts(cases, threshold):
    """True positives, false positives and false negatives at a strict `>` threshold."""
    require_cases(cases)
    true_positive = false_positive = false_negative = 0
    for case in cases:
        for value, label in zip(case["logits"], case["occupied"]):
            predicted = value > threshold
            if label and predicted:
                true_positive += 1
            elif label:
                false_negative += 1
            elif predicted:
                false_positive += 1
    return true_positive, false_positive, false_negative


def f1_from_counts(true_positive, false_positive, false_negative):
    """2TP/(2TP+FP+FN), the gates' cell-wise quantity; refuses an empty denominator."""
    for value in (true_positive, false_positive, false_negative):
        if type(value) is not int or value < 0:
            raise ValueError("nonnegative integer counts required")
    denominator = 2*true_positive+false_positive+false_negative
    if denominator == 0:
        raise ValueError("occupied-cell F1 needs occupied or predicted cells")
    return 2*true_positive/denominator


def precision_from_counts(true_positive, false_positive):
    predicted = true_positive+false_positive
    return true_positive/predicted if predicted else 0.0


def select_threshold(cases, thresholds=None):
    """The sweep's rule - highest F1, then precision, then the lower threshold - on these boards."""
    thresholds = threshold_grid() if thresholds is None else tuple(thresholds)
    require_cases(cases)
    return max(thresholds, key=lambda threshold: (
        f1_from_counts(*counts(cases, threshold)),
        precision_from_counts(*counts(cases, threshold)[:2]), -threshold))


def exact_boards(cases, threshold):
    """Boards whose every known cell is decided correctly at this threshold."""
    require_cases(cases)
    exact = 0
    for case in cases:
        if all((value > threshold) == bool(label)
               for value, label in zip(case["logits"], case["occupied"])):
            exact += 1
    return exact


def nonempty(cases):
    """The boards the sweep's F1 curve is computed over; an empty board is scored separately."""
    require_cases(cases)
    return [case for case in cases if any(case["occupied"])]


def group_folds(cases):
    """Sorted (group, rows) pairs, so a fold's training complement is exactly determined."""
    require_cases(cases)
    grouped = {}
    for case in cases:
        if "group_id" not in case:
            raise ValueError("saved boards must carry their ancestry group")
        grouped.setdefault(case["group_id"], []).append(case)
    if len(grouped) < 3:
        raise ValueError("at least three ancestry groups required for a held-out boundary")
    return tuple(sorted(grouped.items()))


def cross_validated(cases, thresholds=None):
    """Out-of-fold-threshold F1: the boundary is chosen on the other groups, never the scored one."""
    thresholds = threshold_grid() if thresholds is None else tuple(thresholds)
    cases = nonempty(cases)
    folds = []
    for group, held in group_folds(cases):
        training = [case for case in cases if case["group_id"] != group]
        threshold = select_threshold(training, thresholds)
        true_positive, false_positive, false_negative = counts(held, threshold)
        folds.append({"group_id": group, "threshold": threshold, "boards": len(held),
                      "true_positive": true_positive, "false_positive": false_positive,
                      "false_negative": false_negative,
                      "exact_boards": exact_boards(held, threshold)})
    pooled = tuple(sum(fold[key] for fold in folds)
                   for key in ("true_positive", "false_positive", "false_negative"))
    return {"f1": f1_from_counts(*pooled), "counts": dict(zip(
                ("true_positive", "false_positive", "false_negative"), pooled)),
            "exact_boards": sum(fold["exact_boards"] for fold in folds),
            "boards": sum(fold["boards"] for fold in folds),
            "folds": folds, "groups": len(folds),
            "selection": "leave-one-group-out; grid, criterion and tie-break from occupancy_sweep"}


def percentile(ordered, q):
    if not ordered or not 0 <= q <= 1:
        raise ValueError("nonempty ordered values and q in [0, 1] required")
    position = (len(ordered)-1)*q
    low, high = int(position//1), -int(-position//1)
    if low == high:
        return ordered[low]
    return ordered[low]*(high-position)+ordered[high]*(position-low)


def bootstrap_cv_interval(folds, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """Whole-group percentile interval over the held-out count triples (selection fixed)."""
    if type(folds) is not list or len(folds) < 3:
        raise ValueError("at least three held-out folds required")
    if type(draws) is not int or not 100 <= draws <= 5000:
        raise ValueError("draws 100..5000 required")
    triples = [(fold["true_positive"], fold["false_positive"], fold["false_negative"])
               for fold in folds]
    if any(any(type(value) is not int or value < 0 for value in triple) for triple in triples):
        raise ValueError("nonnegative integer fold counts required")
    rng, values = random.Random(seed), []
    for _ in range(draws):
        true_positive = false_positive = false_negative = 0
        for _ in range(len(triples)):
            first, second, third = triples[rng.randrange(len(triples))]
            true_positive, false_positive, false_negative = (true_positive+first,
                                                             false_positive+second,
                                                             false_negative+third)
        if 2*true_positive+false_positive+false_negative:
            values.append(2*true_positive/(2*true_positive+false_positive+false_negative))
    if len(values) < draws*0.5:
        raise ValueError("too many degenerate bootstrap draws")
    values.sort()
    return {"draws": draws, "seed": seed, "groups": len(folds),
            "low": round(percentile(values, 0.025), 6),
            "median": round(percentile(values, 0.5), 6),
            "high": round(percentile(values, 0.975), 6),
            "standard_error": round(statistics.pstdev(values), 6)}


def paired_delta_interval(control_folds, lever_folds, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """Whole-group paired interval for `cv_f1(lever) - cv_f1(control)`; boundaries fixed.

    One index draw is shared by both arms, so between-group variation that both arms share is
    removed from the difference. The fold boundaries are the committed ones - this does not
    re-select a threshold inside a resample, and that limit is recorded with the interval.
    """
    if (type(control_folds) is not list or type(lever_folds) is not list or
            len(control_folds) != len(lever_folds) or len(control_folds) < 3):
        raise ValueError("paired folds of at least three groups required")
    if [fold["group_id"] for fold in control_folds] != [fold["group_id"] for fold in lever_folds]:
        raise ValueError("paired folds must share their group order")
    control = [(fold["true_positive"], fold["false_positive"], fold["false_negative"])
               for fold in control_folds]
    lever = [(fold["true_positive"], fold["false_positive"], fold["false_negative"])
             for fold in lever_folds]
    if any(any(type(value) is not int or value < 0 for value in triple)
           for triple in control+lever):
        raise ValueError("nonnegative integer fold counts required")
    rng, values = random.Random(seed), []
    for _ in range(draws):
        control_total = [0, 0, 0]
        lever_total = [0, 0, 0]
        for _ in range(len(control)):
            index = rng.randrange(len(control))
            for slot in range(3):
                control_total[slot] += control[index][slot]
                lever_total[slot] += lever[index][slot]
        if (2*control_total[0]+control_total[1]+control_total[2] and
                2*lever_total[0]+lever_total[1]+lever_total[2]):
            values.append(f1_from_counts(*lever_total)-f1_from_counts(*control_total))
    if len(values) < draws*0.5:
        raise ValueError("too many degenerate paired draws")
    values.sort()
    low, high = percentile(values, 0.025), percentile(values, 0.975)
    return {"draws": draws, "seed": seed, "groups": len(control),
            "low": round(low, 6), "median": round(percentile(values, 0.5), 6),
            "high": round(high, 6), "standard_error": round(statistics.pstdev(values), 6),
            "excludes_zero": bool(low > 0 or high < 0),
            "selection": "paired whole-group resample; committed fold boundaries held fixed"}


def arm_read(cases, thresholds=None):
    """The four declared reads plus the exactness and scale side numbers for one arm."""
    thresholds = threshold_grid() if thresholds is None else tuple(thresholds)
    all_cases = require_cases(cases) or cases
    cases = nonempty(all_cases)
    positive = [value for case in cases
                for value, label in zip(case["logits"], case["occupied"]) if label]
    negative = [value for case in cases
                for value, label in zip(case["logits"], case["occupied"]) if not label]
    scores = [value for case in cases for value in case["logits"]]
    cv = cross_validated(cases, thresholds)
    return {"auc": pooled_auc(positive, negative),
            "f1_at_zero": f1_from_counts(*counts(cases, 0.0)),
            "exact_at_zero": exact_boards(cases, 0.0),
            "cv": cv, "cv_interval": bootstrap_cv_interval(cv["folds"]),
            "logit_offset": round(statistics.median(positive)-statistics.median(negative), 6),
            "median_occupied": round(statistics.median(positive), 6),
            "median_empty": round(statistics.median(negative), 6),
            "logit_scale": round(statistics.pstdev(scores), 6),
            "nonempty_boards": len(cases), "empty_boards": len(all_cases)-len(cases),
            "scored_cells": len(scores), "occupied_cells": len(positive)}


def pair_state(d_zero, d_cv, floor=ATTRIBUTION_FLOOR):
    """The predeclared classification of one pair; no branch is chosen after the numbers."""
    for value in (d_zero, d_cv):
        if type(value) not in (int, float) or value != value:
            raise ValueError("finite attribution required")
    if type(floor) not in (int, float) or not floor > 0:
        raise ValueError("positive attribution floor required")
    if d_zero < floor and d_cv >= floor:
        return "refusal_reversed"
    if d_zero >= floor and d_cv < 0:
        return "gain_reversed"
    if abs(d_zero-d_cv) >= floor:
        return "shifted"
    return "aligned"


def audit_verdict(pairs):
    """Overall label from the predeclared state table, in its declared order."""
    if type(pairs) is not list or not pairs:
        raise ValueError("declared pairs required")
    states = [pair["state"] for pair in pairs]
    unknown = set(states)-{"refusal_reversed", "gain_reversed", "shifted", "aligned"}
    if unknown:
        raise ValueError("unknown pair state: %s" % sorted(unknown))
    tally = {state: states.count(state) for state in
             ("refusal_reversed", "gain_reversed", "shifted", "aligned")}
    if tally["refusal_reversed"] or tally["gain_reversed"]:
        label = "standing_read_confounded"
    elif tally["shifted"]:
        label = "operating_point_matters"
    else:
        label = "standing_read_sufficient"
    return {"label": label, "states": tally, "pairs": len(pairs),
            "floor": ATTRIBUTION_FLOOR}


def require_population(cases):
    """The declared deciding population, so two receipts of different populations cannot join."""
    cases = [case for case in cases if case.get("split") == "training"]
    groups = {case.get("group_id") for case in cases}
    if (len(cases) != DECIDING_CASES or len(groups) != DECIDING_GROUPS or
            sum(sum(case.get("occupied", [])) for case in cases) != OCCUPIED_CELLS or
            any(len(case.get("logits", [])) != KNOWN_CELLS or
                len(case.get("occupied", [])) != KNOWN_CELLS for case in cases)):
        raise ValueError("not the declared deciding population: %d cases, %d groups, %d occupied"
                         % (len(cases), len(groups),
                            sum(sum(case.get("occupied", [])) for case in cases)))
    return cases


def load_receipt(run):
    """One committed sweep receipt, guarded on population, status and a self-reproducing read."""
    path = ROOT/"runs"/(run+"-sweep-r1")/"result.json"
    if not path.is_file():
        raise ValueError("missing committed sweep receipt: "+str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != "passed" or data.get("run") != run:
        raise ValueError("sweep receipt is not a passed receipt of its own run: "+str(path))
    try:
        cases = require_population(data.get("cases", []))
    except ValueError as error:
        raise ValueError(str(error)+": "+str(path)) from None
    published = data.get("deciding", {})
    rows = {row.get("threshold"): row for row in published.get("rows", [])}
    if 0.0 not in rows:
        raise ValueError("sweep receipt has no threshold 0.0 row: "+str(path))
    scored = nonempty(cases)
    positive = [value for case in scored for value, label in zip(case["logits"], case["occupied"]) if label]
    negative = [value for case in scored for value, label in zip(case["logits"], case["occupied"]) if not label]
    if (abs(published.get("auc", -1)-pooled_auc(positive, negative)) > 1e-9 or
            abs(rows[0.0].get("f1", -1)-f1_from_counts(*counts(scored, 0.0))) > 1e-9):
        raise ValueError("sweep receipt does not reproduce its own published read: "+str(path))
    return {"run": run, "path": str(path.relative_to(ROOT)), "sha256": digest(path),
            "weights_sha256": data.get("weights_sha256"), "cases": cases,
            "published": {"auc": published["auc"], "f1_at_zero": rows[0.0]["f1"]}}


def main(argv=None):
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.output.resolve().is_relative_to((ROOT/"docs/evidence").resolve()):
        parser.error("audit output belongs under docs/evidence")
    if args.output.exists():
        parser.error("audit output must be a new evidence directory")
    names = {run for _, control, lever in PAIRS for run in (control, lever)}
    receipts = {run: load_receipt(run) for run in sorted(names)}
    thresholds = list(threshold_grid())
    reads = {run: arm_read(receipts[run]["cases"], thresholds) for run in receipts}
    pairs = []
    for name, control, lever in PAIRS:
        if receipts[control]["weights_sha256"] == receipts[lever]["weights_sha256"]:
            raise ValueError("a declared pair resolved to one weight file: "+name)
        d_zero = reads[lever]["f1_at_zero"]-reads[control]["f1_at_zero"]
        d_cv = reads[lever]["cv"]["f1"]-reads[control]["cv"]["f1"]
        pairs.append({
            "pair": name, "control": control, "lever": lever,
            "control_weights_sha256": receipts[control]["weights_sha256"],
            "lever_weights_sha256": receipts[lever]["weights_sha256"],
            "d_f1_at_zero": round(d_zero, 6), "d_cv_f1": round(d_cv, 6),
            "d_auc": round(reads[lever]["auc"]-reads[control]["auc"], 6),
            "d_logit_offset": round(reads[lever]["logit_offset"]-reads[control]["logit_offset"], 6),
            "d_cv_minus_zero": round(d_cv-d_zero, 6),
            "d_cv_interval": paired_delta_interval(reads[control]["cv"]["folds"],
                                                  reads[lever]["cv"]["folds"]),
            "state": pair_state(d_zero, d_cv)})
    verdict = audit_verdict(pairs)
    # Report-only, exactly as gate_power's intervals are: the states read point estimates.
    verdict["reversals_with_interval_excluding_zero"] = sum(
        1 for pair in pairs if pair["state"] in ("refusal_reversed", "gain_reversed")
        and pair["d_cv_interval"]["excludes_zero"])
    verdict["interval_role"] = "reported beside the states; no state or verdict reads it"
    result = {"status": "passed", "verdict": verdict,
              "population": {"training_cases": DECIDING_CASES, "training_groups": DECIDING_GROUPS,
                             "occupied_cells": OCCUPIED_CELLS, "known_cells_per_board": KNOWN_CELLS},
              "grid": {"low": thresholds[0], "high": thresholds[-1],
                       "step": round(thresholds[1]-thresholds[0], 2), "count": len(thresholds)},
              "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "z_95": Z_95},
              "reads": {run: reads[run] for run in sorted(reads)},
              "receipts": {run: {key: receipts[run][key]
                                 for key in ("path", "sha256", "weights_sha256", "published")}
                           for run in sorted(receipts)},
              "pairs": pairs,
              "sources": {name: digest(ROOT/name) for name in AUDIT_SOURCES},
              "scope": ("Read-only audit of the standing threshold-0.0 occupancy read against an "
                        "out-of-fold threshold on committed sweep receipts. No training, no weight "
                        "change, no gate, no threshold for any gate, and no re-verdict: the state "
                        "rule was predeclared in docs/M3_OPERATING_POINT.md before this receipt.")}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"result.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                                          encoding="utf-8")
    (args.output/"scope.json").write_text(json.dumps(
        {key: result[key] for key in ("status", "verdict", "population", "grid", "bootstrap",
                                      "pairs", "receipts", "sources", "scope")},
        indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict,
                      "pairs": [{key: pair[key] for key in ("pair", "state", "d_f1_at_zero",
                                                            "d_cv_f1", "d_auc")}
                                for pair in pairs]}), flush=True)


if __name__ == "__main__":
    main()
