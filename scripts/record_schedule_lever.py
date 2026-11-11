"""Curate the declared learning-rate schedule lever from its own run records.

`docs/M3_OPTIMISER_SCHEDULE.md` fixes the two arms, the criteria and the falsifiers before any
receipt existed; this recorder applies that rule and asserts the arithmetic before writing.

The schedule is proved from the runs, not from the code: every curve record carries the rate that
step used, so the control must show exactly one rate and the lever a strictly decreasing 256-value
sequence from `1e-3` to `1e-4`. Matched start is proved separately from the same ledgers - identical
initial weight bytes, identical seed, and an identical first record apart from the rate.
"""
import argparse
import json
from pathlib import Path

from fit_gate import FIT_GATE_THRESHOLD
from occupancy_sweep import AUC_THRESHOLD, F1_THRESHOLD, sweep_verdict

ROOT = Path(__file__).resolve().parents[1]
CONTROL, LEVER = "constant", "cosine"
STEPS = 256
SOURCE_RUN = "m3-percell-balance-r1"
CEILING_BYTES = 1152*1024**2
SWEEP_FORWARDS = 54
ATTRIBUTION_FLOOR = 0.05
PLACEMENT_TOLERANCE = 0.05
START_RATE, END_RATE = .001, .0001
SEED = 20261026
INSTRUMENT = "m3-optimiser-schedule-r1"
EXPECTED_LR_FINAL = {CONTROL: None, LEVER: END_RATE}


def read(path, maximum=16*1024**2):
    path = Path(path)
    if path.stat().st_size > maximum:
        raise ValueError(f"{path} exceeds the {maximum}-byte record budget")
    return json.loads(path.read_text())


def curve_rows(run):
    rows = []
    for line in (run/"curve.jsonl").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def inspect_arm(name):
    """Everything the predeclaration requires of one arm, or the reasons it is unreadable."""
    study = f"m3-occsched-{name}-r1"
    run = ROOT/"runs"/study
    problems = []
    summary, job = read(run/"summary.json"), read(ROOT/"runs"/(study+"-job.json"))
    rows = curve_rows(run)
    if job.get("status") != "passed" or job.get("exit_code") != 0:
        problems.append(f"{name}: run did not pass inside its envelope ({job.get('status')})")
    if summary.get("status") != "complete" or summary.get("steps") != STEPS:
        problems.append(f"{name}: {summary.get('status')} at {summary.get('steps')} of {STEPS} steps")
    if job.get("work_items") != STEPS or job.get("memory_bytes") != CEILING_BYTES:
        problems.append(f"{name}: envelope differs from the declared 256 updates / 1,152 MiB")
    arguments = summary.get("arguments", {})
    if (arguments.get("learning_rate") != START_RATE or arguments.get("lr_final") != EXPECTED_LR_FINAL[name] or
            arguments.get("seed") != SEED or arguments.get("steps") != STEPS):
        problems.append(f"{name}: declared schedule/seed/budget arguments differ")
    origin = summary.get("initial_weights_from") or {}
    if not str(origin.get("run", "")).endswith(SOURCE_RUN) or not origin.get("weights_sha256"):
        problems.append(f"{name}: not initialized from the published checkpoint")
    if [row.get("step") for row in rows] != list(range(1, STEPS+1)):
        problems.append(f"{name}: curve does not record exactly steps 1..{STEPS}")
    rates = [row.get("learning_rate") for row in rows]
    if any(not isinstance(rate, float) for rate in rates):
        problems.append(f"{name}: curve is missing per-step learning rates")
        rates = [rate for rate in rates if isinstance(rate, float)]
    if name == CONTROL:
        if len(set(rates)) != 1 or (rates and rates[0] != START_RATE):
            problems.append(f"{name}: the control arm's rate is not the single constant {START_RATE}")
    elif rates:
        if len(set(rates)) != len(rates) or rates[0] != START_RATE or rates[-1] != END_RATE:
            problems.append(f"{name}: the lever arm does not decay exactly from {START_RATE} to {END_RATE}")
        if any(later > earlier for earlier, later in zip(rates, rates[1:])):
            problems.append(f"{name}: the lever arm's rate is not non-increasing")
    return {"study": study, "summary": summary, "job": job, "rows": rows, "rates": rates,
            "origin": origin, "problems": problems}


def require_sweep(study, weights):
    sweep, job = read(ROOT/"runs"/(study+"-sweep-r1")/"result.json"), read(ROOT/"runs"/(study+"-sweep-r1-job.json"))
    problems = []
    if job.get("status") != "passed" or job.get("exit_code") != 0:
        problems.append(f"{study}: sweep did not pass inside its envelope")
    if sweep.get("status") != "passed" or sweep.get("forwards") != SWEEP_FORWARDS:
        problems.append(f"{study}: sweep is not a complete {SWEEP_FORWARDS}-forward measurement")
    if sweep.get("weights_sha256") != weights:
        problems.append(f"{study}: sweep measured different weights from its arm")
    deciding = sweep.get("deciding") or {}
    if sweep_verdict(deciding) != sweep.get("verdict"):
        problems.append(f"{study}: swept verdict does not re-derive")
    return sweep, deciding, problems


def zero_row(deciding):
    rows = [row for row in deciding.get("rows", []) if row.get("threshold") == 0.0]
    if len(rows) != 1:
        raise ValueError("the standing grid must contain exactly one zero threshold")
    return rows[0]


def placement_of(arm):
    """The trainer's own geometry error, in cells, over the development probe population."""
    cases = arm["summary"]["development"]["cases"]
    if not cases:
        return None
    return sum(case["mean_geometry_error_cells"] for case in cases)/len(cases)


def main():
    from probe_model import require_bounded_job
    bounded = require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("evidence directory already exists")

    arms = {name: inspect_arm(name) for name in (CONTROL, LEVER)}
    problems = [problem for arm in arms.values() for problem in arm["problems"]]
    measured = {}
    for name, arm in arms.items():
        sweep, deciding, sweep_problems = require_sweep(arm["study"], arm["summary"]["checkpoint_sha256"])
        problems += sweep_problems
        row = zero_row(deciding)
        measured[name] = {"auc": deciding["auc"], "per_cell_f1": row["f1"],
                          "operating_counts": {key: row[key] for key in
                                               ("true_positive", "false_positive", "false_negative")},
                          "exact_nonempty_boards_at_zero": row["exact_nonempty_boards"],
                          "best": {key: deciding["best"][key] for key in
                                   ("threshold", "f1", "precision", "recall", "exact_nonempty_boards")},
                          "exact_boards_at_best": deciding["exact_boards_at_best"],
                          "verdict": sweep.get("verdict"), "sweep": sweep, "deciding": deciding,
                          "placement_cells": placement_of(arm)}
    if arms[CONTROL]["summary"]["initial_weights_sha256"] != arms[LEVER]["summary"]["initial_weights_sha256"]:
        problems.append("the arms did not start from the same initial weight bytes")
    if arms[CONTROL]["summary"]["checkpoint_sha256"] == arms[LEVER]["summary"]["checkpoint_sha256"]:
        problems.append("the arms ended in identical weights; the schedule did not run")
    first = {name: arm["rows"][0] for name, arm in arms.items()}
    matched = {key: first[CONTROL].get(key) == first[LEVER].get(key)
               for key in ("step", "sample_id", "loss", "gradient_norm")}
    if not all(matched.values()):
        problems.append("step 1 differs between the arms beyond the schedule")
    if first[CONTROL].get("learning_rate") == first[LEVER].get("learning_rate") is None:
        problems.append("step 1 records no schedule identity")

    attribution = measured[LEVER]["per_cell_f1"]-measured[CONTROL]["per_cell_f1"]
    placement = {name: measured[name]["placement_cells"] for name in measured}
    placement_change = ((placement[LEVER]-placement[CONTROL])/placement[CONTROL]
                        if placement[CONTROL] else None)
    fit_gate = {name: {"threshold": 0.0, "exact_nonempty_boards": measured[name]["exact_nonempty_boards_at_zero"],
                       "opens": measured[name]["exact_nonempty_boards_at_zero"] >= FIT_GATE_THRESHOLD}
                for name in measured}
    conditions = {"attribution": attribution >= ATTRIBUTION_FLOOR,
                  "placement": placement_change is not None and placement_change <= PLACEMENT_TOLERANCE,
                  "fit_gate": fit_gate[LEVER]["opens"]}
    if problems:
        status, verdict = "unreadable", "unreadable"
    elif not conditions["attribution"]:
        status, verdict = "passed", "null"
    elif not conditions["placement"]:
        status, verdict = "passed", "not-cleared-placement"
    else:
        status, verdict = "passed", "schedule_engaged"

    scope = {
        "status": status,
        "instrument": INSTRUMENT,
        "predeclaration": "docs/M3_OPTIMISER_SCHEDULE.md",
        "declared_receipt": args.receipt,
        "declared_memory_bytes": bounded,
        "scope": ("Matched 256-step continuation of the published checkpoint, two arms differing only in the "
                  "learning-rate schedule: constant 1e-3 against cosine 1e-3 -> 1e-4. Zero added parameters, "
                  "one seed, one population and one objective. Read only by the standing gates; no threshold, "
                  "population, ceiling or prior verdict changes, and no real-image, complete-state or quality claim."),
        "envelope": {"steps": STEPS, "ceiling_bytes": CEILING_BYTES, "work_items": STEPS,
                     "attribution_floor": ATTRIBUTION_FLOOR, "placement_tolerance": PLACEMENT_TOLERANCE,
                     "fit_gate_threshold": FIT_GATE_THRESHOLD},
        "source": {"run": SOURCE_RUN, "checkpoint_sha256": arms[CONTROL]["origin"].get("weights_sha256")},
        "arms": {name: {"study": arms[name]["study"],
                        "lr_final": EXPECTED_LR_FINAL[name],
                        "rates": {"steps": len(arms[name]["rates"]),
                                  "distinct": len(set(arms[name]["rates"])),
                                  "first": arms[name]["rates"][0] if arms[name]["rates"] else None,
                                  "last": arms[name]["rates"][-1] if arms[name]["rates"] else None},
                        "weights_sha256": arms[name]["summary"]["checkpoint_sha256"],
                        "catalogue_sha256": arms[name]["summary"]["catalogue_sha256"],
                        "elapsed_seconds": arms[name]["summary"]["elapsed_seconds"],
                        "peak_commit_bytes": arms[name]["job"].get("peak_job_commit_bytes"),
                        "parameters": None,
                        "auc": measured[name]["auc"],
                        "per_cell_f1_at_zero": measured[name]["per_cell_f1"],
                        "operating_counts_at_zero": measured[name]["operating_counts"],
                        "best": measured[name]["best"],
                        "exact_boards_at_best": measured[name]["exact_boards_at_best"],
                        "sweep_verdict": measured[name]["verdict"],
                        "placement_cells": measured[name]["placement_cells"]}
                    for name in measured},
        "matched_start": {"initial_weights_sha256": arms[CONTROL]["summary"]["initial_weights_sha256"],
                          "seed": SEED, "step_one_identical": matched,
                          "step_one": {name: {key: first[name].get(key) for key in
                                              ("sample_id", "loss", "gradient_norm", "learning_rate")}
                                       for name in first}},
        "attribution": {"per_cell_f1_delta": round(attribution, 6),
                        "placement_change": None if placement_change is None else round(placement_change, 6),
                        "fit_gate_opened": fit_gate[LEVER]["opens"]},
        "gates": {"attribution": conditions["attribution"],
                  "ranking_auc_bar": AUC_THRESHOLD, "ranking_f1_bar": F1_THRESHOLD,
                  "fit_gate": fit_gate, "conditions": conditions},
        "verdict": verdict,
        "problems": problems,
        "limits": ["One seed, synthetic training fit, committed lattice; no real-image or complete-state claim.",
                   "Pairwise attribution stands: both arms share the clip regime V-086 measured.",
                   "Placement is the trainer's own mean geometry error over the development probe population."],
    }
    args.output.mkdir(parents=True)
    with (args.output/"scope.json").open("x", encoding="utf-8") as stream:
        json.dump(scope, stream, indent=2, allow_nan=False)
    print(json.dumps({key: scope[key] for key in ("status", "verdict", "attribution", "matched_start")},
                     indent=2))


if __name__ == "__main__":
    main()
