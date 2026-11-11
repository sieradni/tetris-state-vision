"""Power and minimum-detectable-effect analysis of the committed replicate block.

`docs/M3_MINIMUM_DETECTABLE_EFFECT.md` states the rules; this module is their single implementation.
It re-reads the nine committed sweep receipts, rebuilds the same 36 `(seed, group)` clusters the
published read used, and first proves it reproduces that read exactly - pooled F1 per arm, both
paired point deltas and both bootstrap standard errors. Only then does it measure what the design can
resolve.

Three quantities are derived from the receipts rather than assumed:

- the paired delta's standard error, both by the declared whole-cluster bootstrap and by a
  delta-method linearisation of pooled F1 (the cross-check that says the bootstrap width is not an
  artifact of resampling);
- the design effect of those clusters, by comparing the cluster-level standard error with the
  standard error the same read would show if boards, then cells, were independent;
- the between-population variance, from the three per-seed deltas corrected for their own sampling
  variance, which is what a claim about "the method" has to generalise over.

`minimum_detectable_effect` is the two-sided alpha=0.05, power=0.80 effect, `seeds_needed` inverts it
for a target effect, and `decompose` splits the out-of-fold F1 into the arm's own ranking quality and
the two boundary terms. Read-only: no training, no weight change, no gate, and no verdict is
recomputed.
"""
import argparse
import json
import math
import statistics
from pathlib import Path

from compute_replication import (ARMS, CONTROL, LEVERS, SEEDS, block_clusters, digest, load_arm,
                                 paired_delta_interval, pooled_f1, require_one_population)
from operating_point import counts, cross_validated, f1_from_counts, nonempty, select_threshold
from probe_model import require_bounded_job

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT/"docs/evidence/m3-compute-replication-r1/result.json"
AUDIT = ROOT/"docs/evidence/m3-operating-point-r2/result.json"
ALPHA, POWER = 0.05, 0.80
FLOOR, AUDIT_EFFECT = 0.05, 0.149
AUDIT_SEEDS, GROUPS_PER_SEED = 3, 12
PROPOSED_STEPS = {"budget": 256, "lever": 384}
ANALYSIS_SOURCES = tuple("scripts/"+name for name in
                         ("power_analysis.py", "compute_replication.py", "operating_point.py",
                          "occupancy_sweep.py", "probe_model.py"))


def z(value):
    return statistics.NormalDist().inv_cdf(value)


def minimum_detectable_effect(standard_error, alpha=ALPHA, power=POWER):
    """Smallest true effect that reaches significance at `alpha` with probability `power`."""
    if type(standard_error) not in (int, float) or not standard_error > 0:
        raise ValueError("a positive standard error is required")
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("alpha and power must lie in (0, 1)")
    return (z(1-alpha/2)+z(power))*standard_error


def power_at(effect, standard_error, alpha=ALPHA):
    """Two-sided power of a design with this standard error against a true effect."""
    if type(standard_error) not in (int, float) or not standard_error > 0:
        raise ValueError("a positive standard error is required")
    if type(effect) not in (int, float) or effect != effect:
        raise ValueError("a finite effect is required")
    ratio, bound = abs(effect)/standard_error, z(1-alpha/2)
    return statistics.NormalDist().cdf(ratio-bound)+statistics.NormalDist().cdf(-ratio-bound)


def audit_design(path=AUDIT, name="fit_budget", effect=AUDIT_EFFECT):
    """Was the audit's own claim resolvable at the audit's own design?"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pairs = [pair for pair in data.get("pairs", []) if pair.get("pair") == name]
    if len(pairs) != 1:
        raise ValueError("exactly one audit pair of the declared name is required")
    interval = pairs[0]["d_cv_interval"]
    standard_error = interval["standard_error"]
    floor = minimum_detectable_effect(standard_error)
    return {"pair": name, "evidence": str(Path(path).resolve().relative_to(ROOT)),
            "groups": interval["groups"], "point": pairs[0]["d_cv_f1"],
            "standard_error": standard_error,
            "minimum_detectable_effect": round(floor, 6),
            "claim_resolvable_at_its_own_design": bool(effect >= floor),
            "power_at_claim": round(power_at(effect, standard_error), 4)}


def pooled_totals(clusters, arm):
    totals = [sum(cluster[arm][slot] for cluster in clusters) for slot in range(3)]
    if 2*totals[0]+totals[1]+totals[2] == 0:
        raise ValueError("empty pooled denominator")
    return totals


def influence(clusters, arm):
    """Per-cluster first-order influence on pooled F1; their sum is zero by construction."""
    true_positive, false_positive, false_negative = pooled_totals(clusters, arm)
    denominator = 2*true_positive+false_positive+false_negative
    positive = 2*(false_positive+false_negative)/denominator**2
    negative = 2*true_positive/denominator**2
    values = [positive*cluster[arm][0]-negative*(cluster[arm][1]+cluster[arm][2])
              for cluster in clusters]
    if abs(sum(values)) > 1e-9*max(1.0, abs(positive*true_positive)):
        raise ValueError("cluster influence did not telescope to zero")
    return values


def cluster_standard_error(clusters, arm):
    values = influence(clusters, arm)
    return math.sqrt(len(values)*statistics.pvariance(values))


def paired_standard_error(clusters, lever):
    """Delta-method standard error of `f1(lever) - f1(control)` over these clusters."""
    differences = [left-right for left, right in
                   zip(influence(clusters, lever), influence(clusters, CONTROL))]
    return math.sqrt(len(differences)*statistics.pvariance(differences))


def board_counts(cases, folds):
    thresholds = {fold["group_id"]: fold["threshold"] for fold in folds}
    return {case["sample_id"]: counts([case], thresholds[case["group_id"]])
            for case in nonempty(cases)}


def cell_weights(cases, folds, positive, negative):
    """One weight per known cell at its own group's out-of-fold boundary."""
    thresholds = {fold["group_id"]: fold["threshold"] for fold in folds}
    weights = []
    for case in nonempty(cases):
        threshold = thresholds[case["group_id"]]
        for value, label in zip(case["logits"], case["occupied"]):
            predicted = value > threshold
            if label:
                weights.append(positive if predicted else -negative)
            else:
                weights.append(-negative if predicted else 0.0)
    return weights


def coefficients(clusters, arm):
    true_positive, false_positive, false_negative = pooled_totals(clusters, arm)
    denominator = 2*true_positive+false_positive+false_negative
    return 2*(false_positive+false_negative)/denominator**2, 2*true_positive/denominator**2


def independent_standard_error(unit_counts, positive, negative):
    """The same read if these units were independent draws of one population."""
    values = {name: positive*triple[0]-negative*(triple[1]+triple[2])
              for name, triple in unit_counts.items()}
    ordered = list(values.values())
    return math.sqrt(len(ordered)*statistics.pvariance(ordered)), values


def decompose(f1_best, f1_zero, cv_f1):
    """Ranking quality plus the two boundary terms; the parts add to the out-of-fold F1."""
    return {"ranking_best_boundary": f1_best,
            "zero_boundary_cost": f1_zero-f1_best,
            "transfer_cost": cv_f1-f1_zero}


def population_variance(per_seed_deltas, within_variances):
    """Between-population variance, corrected for each seed's own sampling variance."""
    raw = statistics.pvariance(list(per_seed_deltas))
    corrected = raw-statistics.mean(list(within_variances))
    return {"raw": raw, "corrected": max(corrected, 0.0), "clamped": corrected < 0}


def seeds_needed(effect, variance_between, variance_within, alpha=ALPHA, power=POWER):
    """Seeds at `GROUPS_PER_SEED` groups each so that `effect` clears the detectable floor."""
    if not effect > 0:
        raise ValueError("a positive target effect is required")
    multiplier = (z(1-alpha/2)+z(power))**2
    return math.ceil(multiplier*(variance_between+variance_within)/effect**2)


def main(argv=None):
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.output.resolve().is_relative_to((ROOT/"docs/evidence").resolve()):
        parser.error("analysis output belongs under docs/evidence")
    if args.output.exists():
        parser.error("analysis output must be a new evidence directory")
    published = json.loads(PUBLISHED.read_text(encoding="utf-8"))

    reads, receipts, arms, per_arm = {}, {}, {}, {}
    for seed in SEEDS:
        for arm in ARMS:
            data = load_arm(arm, seed)
            validation = cross_validated(data["cases"])
            boards = nonempty(data["cases"])
            best = select_threshold(boards)
            reads[(seed, arm)] = validation["folds"]
            receipts[f"{arm}-s{seed}"] = {"path": data["path"], "sha256": data["sha256"]}
            arms[(seed, arm)] = data
            per_arm[(seed, arm)] = {
                "auc": data["auc"], "f1_best": f1_from_counts(*counts(boards, best)),
                "f1_zero": f1_from_counts(*counts(boards, 0.0)), "cv_f1": validation["f1"],
                "best_threshold": best, "boards": boards, "folds": validation["folds"]}
        require_one_population({arm: arms[(seed, arm)] for arm in ARMS}, seed)

    clusters = block_clusters(reads)
    reproduced = {}
    for arm in ARMS:
        recomputed = pooled_f1(clusters, arm)
        expected = published["levers"][LEVERS[0]]["pooled_f1"][arm]
        if abs(recomputed-expected) > 1e-6:
            raise ValueError(f"pooled F1 of {arm} did not reproduce the published read")
        reproduced[arm] = round(recomputed, 6)
    intervals = {}
    for lever in LEVERS:
        interval = paired_delta_interval(clusters, lever)
        expected = published["levers"][lever]["interval"]
        if interval != expected:
            raise ValueError(f"{lever}: bootstrap interval did not reproduce the published one")
        point = pooled_f1(clusters, lever)-pooled_f1(clusters, CONTROL)
        if abs(point-published["levers"][lever]["point"]) > 1e-6:
            raise ValueError(f"{lever}: point delta did not reproduce the published read")
        intervals[lever] = interval

    board_units = {lever: {name: triple for seed in SEEDS for name, triple in
                           board_counts(per_arm[(seed, lever)]["boards"],
                                        per_arm[(seed, lever)]["folds"]).items()}
                   for lever in LEVERS}
    board_units[CONTROL] = {name: triple for seed in SEEDS for name, triple in
                            board_counts(per_arm[(seed, CONTROL)]["boards"],
                                         per_arm[(seed, CONTROL)]["folds"]).items()}
    cells = {arm: [] for arm in ARMS}
    for seed in SEEDS:
        for arm in ARMS:
            positive, negative = coefficients(clusters, arm)
            cells[arm] += cell_weights(per_arm[(seed, arm)]["boards"],
                                       per_arm[(seed, arm)]["folds"], positive, negative)

    effects, decomposition = {}, {}
    for lever in LEVERS:
        bootstrap = intervals[lever]["standard_error"]
        analytic = paired_standard_error(clusters, lever)
        positive, negative = coefficients(clusters, lever)
        control_positive, control_negative = coefficients(clusters, CONTROL)
        board_differences = {}
        for name, triple in board_units[lever].items():
            control_triple = board_units[CONTROL][name]
            board_differences[name] = (positive*triple[0]-negative*(triple[1]+triple[2])
                                       - (control_positive*control_triple[0]
                                          - control_negative*(control_triple[1]+control_triple[2])))
        board_values = list(board_differences.values())
        board_se = math.sqrt(len(board_values)*statistics.pvariance(board_values))
        cell_values = [left-right for left, right in zip(cells[lever], cells[CONTROL])]
        cell_se = math.sqrt(len(cell_values)*statistics.pvariance(cell_values))
        observed = abs(published["levers"][lever]["point"])
        effects[lever] = {
            "observed_point": published["levers"][lever]["point"],
            "observed_magnitude": round(observed, 6),
            "standard_error_bootstrap": bootstrap,
            "standard_error_analytic": round(analytic, 6),
            "analytic_over_bootstrap": round(analytic/bootstrap, 4),
            "minimum_detectable_effect": round(minimum_detectable_effect(analytic), 6),
            "inside_detectable_floor": bool(observed < minimum_detectable_effect(analytic)),
            "power_at_declared_floor": round(power_at(FLOOR, analytic), 4),
            "power_at_audit_effect": round(power_at(AUDIT_EFFECT, analytic), 4),
            "design_effect_board": round((analytic/board_se)**2, 3),
            "design_effect_cell": round((analytic/cell_se)**2, 3),
            "units": {"clusters": len(clusters), "boards": len(board_values),
                      "cells": len(cell_values)},
            "per_seed": published["levers"][lever]["per_seed"]}
        within = [paired_standard_error([c for c in clusters if c["seed"] == seed], lever)**2
                  for seed in SEEDS]
        variances = population_variance(list(published["levers"][lever]["per_seed"].values()),
                                        within)
        needed = seeds_needed(AUDIT_EFFECT, variances["corrected"], statistics.mean(within))
        population_se = math.sqrt((variances["corrected"]+statistics.mean(within))/AUDIT_SEEDS)
        fixed_se = math.sqrt(statistics.mean(within)/AUDIT_SEEDS)
        effects[lever]["variance"] = {
            "within_seed_standard_errors": [round(math.sqrt(value), 6) for value in within],
            "between_population_variance_corrected": round(variances["corrected"], 6),
            "between_population_variance_clamped": variances["clamped"],
            "population_standard_error_at_three_seeds": round(population_se, 6),
            "population_mde_at_three_seeds": round(minimum_detectable_effect(population_se), 6),
            "fixed_population_standard_error_at_three_seeds": round(fixed_se, 6),
            "fixed_population_mde_at_three_seeds": round(minimum_detectable_effect(fixed_se), 6),
            "power_at_audit_effect_population": round(power_at(AUDIT_EFFECT, population_se), 4),
            "seeds_for_audit_effect": needed,
            "seeds_for_floor": seeds_needed(FLOOR, variances["corrected"],
                                            statistics.mean(within))}
        cells_table = {}
        for arm in ARMS:
            seed_rows = {}
            for seed in SEEDS:
                row = per_arm[(seed, arm)]
                seed_rows[str(seed)] = {key: round(row[key], 6) for key in
                                        ("auc", "f1_best", "f1_zero", "cv_f1")}
                seed_rows[str(seed)]["parts"] = {key: round(value, 6) for key, value in
                                                 decompose(row["f1_best"], row["f1_zero"],
                                                           row["cv_f1"]).items()}
            cells_table[arm] = seed_rows
        decomposition[lever] = cells_table

    steps = []
    for path in sorted((ROOT/"runs").glob("compute-replication-s*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for job in manifest.get("jobs", []):
            if job.get("kind") == "arm" and job.get("status") == "passed" and job.get("steps"):
                seconds = job.get("elapsed_seconds", job.get("elapsed", 0.0))
                if seconds:
                    steps.append(seconds/job["steps"])
    seconds_per_step = statistics.median(steps) if steps else None
    def block_cost(seeds):
        training = seeds*(PROPOSED_STEPS["budget"]+PROPOSED_STEPS["lever"])*seconds_per_step
        return {"seeds": seeds, "clusters": seeds*GROUPS_PER_SEED,
                "arm_hours": round(training/3600, 3)}

    cost = {}
    for lever in LEVERS:
        rows = {}
        for name, key in (("audit_effect", "seeds_for_audit_effect"), ("floor", "seeds_for_floor")):
            seeds = effects[lever]["variance"][key]
            if seeds and seconds_per_step:
                rows[name] = block_cost(seeds)
        if rows:
            rows["seconds_per_step_median"] = round(seconds_per_step, 6)
            rows["steps_per_seed"] = PROPOSED_STEPS["budget"]+PROPOSED_STEPS["lever"]
            cost[lever] = rows

    audit = audit_design()
    result = {"status": "passed",
              "reproduction": {"published_read": str(PUBLISHED.relative_to(ROOT)),
                               "pooled_f1": reproduced,
                               "point": {lever: published["levers"][lever]["point"]
                                         for lever in LEVERS},
                               "standard_error": {lever: intervals[lever]["standard_error"]
                                                  for lever in LEVERS}},
              "design": {"alpha": ALPHA, "power": POWER, "floor": FLOOR,
                         "audit_effect": AUDIT_EFFECT, "seeds": list(SEEDS),
                         "groups_per_seed": GROUPS_PER_SEED, "clusters": len(clusters)},
              "levers": effects,
              "decomposition": decomposition,
              "cost": cost,
              "audit": audit,
              "receipts": receipts,
              "sources": {name: digest(ROOT/name) for name in ANALYSIS_SOURCES},
              "scope": ("Read-only power analysis of the committed replicate block: reproducible "
                        "paired standard errors, the design effect those 36 clusters show, the "
                        "minimum detectable effect per lever, the seed count a 256/384 block would "
                        "need, and a ranking/boundary decomposition of the out-of-fold F1. No "
                        "training, no gate, no verdict change.")}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"result.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                                          encoding="utf-8")
    (args.output/"scope.json").write_text(json.dumps(
        {key: result[key] for key in ("status", "reproduction", "design", "levers", "cost",
                                      "audit", "sources", "scope")}, indent=2)+"\n",
        encoding="utf-8")
    print(json.dumps({"reproduction": result["reproduction"]["standard_error"],
                      "levers": {lever: {key: effects[lever][key] for key in
                                         ("observed_magnitude", "minimum_detectable_effect",
                                          "inside_detectable_floor", "power_at_declared_floor",
                                          "design_effect_cell", "analytic_over_bootstrap")}
                                 for lever in LEVERS},
                      "population": {lever: effects[lever]["variance"] for lever in LEVERS},
                      "cost": cost, "audit": audit}), flush=True)


if __name__ == "__main__":
    main()
