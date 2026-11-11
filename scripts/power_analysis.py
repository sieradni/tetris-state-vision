"""Power and minimum-detectable-effect analysis of the committed replicate block.

`docs/M3_MINIMUM_DETECTABLE_EFFECT.md` states the rules; this module is the single source of every
figure that document quotes. `analyse` recomputes them from the nine committed sweep receipts and
the same 36 `(seed, group)` clusters the published read used, after proving it reproduces that read
exactly - pooled F1 per arm, both paired point deltas and both bootstrap standard errors - and
`render` writes the tables the document embeds between its figure markers, so a hand edit to a table
is a test failure rather than a silent stale number.

Three quantities are derived from the receipts rather than assumed: the paired delta's standard
error, both by the declared whole-cluster bootstrap and by a delta-method linearisation of pooled F1;
the design effect of those clusters, by comparing the cluster-level standard error with the standard
error the same read would show if boards, then cells, were independent; and the between-population
variance, from the three per-seed deltas corrected for their own sampling variance.

That variance rests on two degrees of freedom, so it is also reported as a band: a scaled chi-square
interval on a variance of three draws, propagated through the same `seeds_needed` arithmetic, so the
recommendation is a range whose weakest end is visible instead of one number. Read-only: no training,
no weight change, no gate, and no verdict is recomputed.
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
BAND_DF, BAND_COVERAGE = 2, 0.90
MARKERS = ("<!-- power-figures:begin -->", "<!-- power-figures:end -->")
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


def chi_square_band(variance, df=BAND_DF, coverage=BAND_COVERAGE):
    """Scaled chi-square interval for a variance estimated on `df` draws.

    For two degrees of freedom the quantiles are closed-form: chi2(p, 2) = -2 ln(1-p). This is an
    approximation for a variance component (a corrected difference of mean squares), not an exact
    pivot, and it is reported as such.
    """
    if type(variance) not in (int, float) or variance < 0:
        raise ValueError("a nonnegative variance is required")
    if df != BAND_DF:
        raise ValueError(f"the closed-form band is declared for df={BAND_DF}")
    if not 0 < coverage < 1:
        raise ValueError("coverage must lie in (0, 1)")
    tail = (1-coverage)/2
    lower_factor = df/(-2*math.log(tail))
    upper_factor = df/(-2*math.log(1-tail))
    return {"df": df, "coverage": coverage,
            "variance_low": variance*lower_factor, "variance": variance,
            "variance_high": variance*upper_factor,
            "factor_low": lower_factor, "factor_high": upper_factor}


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
            "standard_error": standard_error, "minimum_detectable_effect": round(floor, 6),
            "claim_resolvable_at_its_own_design": bool(effect >= floor),
            "power_at_claim": round(power_at(effect, standard_error), 4)}


def steps_per_second():
    """Median measured seconds per training step across the nine committed arms."""
    values = []
    for path in sorted((ROOT/"runs").glob("compute-replication-s*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for job in manifest.get("jobs", []):
            if job.get("kind") == "arm" and job.get("status") == "passed" and job.get("steps"):
                seconds = job.get("elapsed_seconds", job.get("elapsed", 0.0))
                if seconds:
                    values.append(seconds/job["steps"])
    if not values:
        raise ValueError("no committed arm timings to measure the cost from")
    return statistics.median(values)


def analyse(published_path=PUBLISHED):
    """Every figure the method document quotes, recomputed from the committed receipts."""
    published = json.loads(Path(published_path).read_text(encoding="utf-8"))
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
        if interval != published["levers"][lever]["interval"]:
            raise ValueError(f"{lever}: bootstrap interval did not reproduce the published one")
        point = pooled_f1(clusters, lever)-pooled_f1(clusters, CONTROL)
        if abs(point-published["levers"][lever]["point"]) > 1e-6:
            raise ValueError(f"{lever}: point delta did not reproduce the published read")
        intervals[lever] = interval

    seconds = steps_per_second()
    board_units = {lever: {name: triple for seed in SEEDS for name, triple in
                           board_counts(per_arm[(seed, lever)]["boards"],
                                        per_arm[(seed, lever)]["folds"]).items()}
                   for lever in list(LEVERS)+[CONTROL]}
    cells = {arm: [] for arm in ARMS}
    for seed in SEEDS:
        for arm in ARMS:
            positive, negative = coefficients(clusters, arm)
            cells[arm] += cell_weights(per_arm[(seed, arm)]["boards"],
                                       per_arm[(seed, arm)]["folds"], positive, negative)

    effects, decomposition, deltas = {}, {}, {}
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
        within = [paired_standard_error([c for c in clusters if c["seed"] == seed], lever)**2
                  for seed in SEEDS]
        within_mean = statistics.mean(within)
        variances = population_variance(list(published["levers"][lever]["per_seed"].values()),
                                        within)
        band = chi_square_band(variances["corrected"])
        band["seeds"] = {
            name: seeds_needed(AUDIT_EFFECT, band[key], within_mean)
            for name, key in (("low", "variance_low"), ("point", "variance"),
                              ("high", "variance_high"))}
        band["arm_hours"] = {name: round(count*sum(PROPOSED_STEPS.values())*seconds/3600, 3)
                             for name, count in band["seeds"].items()}
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
            "per_seed": published["levers"][lever]["per_seed"],
            "variance": {
                "within_seed_standard_errors": [round(math.sqrt(value), 6) for value in within],
                "within_seed_variance_mean": round(within_mean, 9),
                "between_population_variance_corrected": round(variances["corrected"], 6),
                "between_population_variance_raw": round(variances["raw"], 6),
                "between_population_variance_clamped": variances["clamped"],
                "population_standard_error_at_three_seeds":
                    round(math.sqrt((variances["corrected"]+within_mean)/AUDIT_SEEDS), 6),
                "population_mde_at_three_seeds": round(minimum_detectable_effect(
                    math.sqrt((variances["corrected"]+within_mean)/AUDIT_SEEDS)), 6),
                "fixed_population_standard_error_at_three_seeds":
                    round(math.sqrt(within_mean/AUDIT_SEEDS), 6),
                "fixed_population_mde_at_three_seeds": round(minimum_detectable_effect(
                    math.sqrt(within_mean/AUDIT_SEEDS)), 6),
                "power_at_audit_effect_population": round(power_at(
                    AUDIT_EFFECT, math.sqrt((variances["corrected"]+within_mean)/AUDIT_SEEDS)), 4),
                "band": band,
            }}
        table = {}
        for arm in ARMS:
            table[arm] = {}
            for seed in SEEDS:
                row = per_arm[(seed, arm)]
                table[arm][str(seed)] = {key: round(row[key], 6) for key in
                                         ("auc", "f1_best", "f1_zero", "cv_f1")}
                table[arm][str(seed)]["parts"] = {key: round(value, 6) for key, value in
                                                 decompose(row["f1_best"], row["f1_zero"],
                                                           row["cv_f1"]).items()}
        decomposition[lever] = table
        deltas[lever] = {}
        for seed in map(str, SEEDS):
            baseline, arm_row = table[CONTROL][seed], table[lever][seed]
            deltas[lever][seed] = {key: round(arm_row[key]-baseline[key], 6)
                                   for key in ("auc", "f1_best", "f1_zero", "cv_f1")}
            deltas[lever][seed]["parts"] = {
                key: round(arm_row["parts"][key]-baseline["parts"][key], 6)
                for key in ("ranking_best_boundary", "zero_boundary_cost", "transfer_cost")}

    def design(seeds, between, within):
        error = math.sqrt((between+within)/seeds)
        return {"seeds": seeds, "clusters": seeds*GROUPS_PER_SEED,
                "arm_hours": round(seeds*sum(PROPOSED_STEPS.values())*seconds/3600, 3),
                "standard_error": round(error, 6),
                "minimum_detectable_effect": round(minimum_detectable_effect(error), 6)}

    cost = {}
    for lever in LEVERS:
        between = effects[lever]["variance"]["between_population_variance_corrected"]
        within = effects[lever]["variance"]["within_seed_variance_mean"]
        band = effects[lever]["variance"]["band"]
        cost[lever] = {
            "audit_effect": design(band["seeds"]["point"], between, within),
            "audit_low": design(band["seeds"]["low"], band["variance_low"], within),
            "audit_high": design(band["seeds"]["high"], band["variance_high"], within),
            "floor": design(seeds_needed(FLOOR, between, within), between, within),
            "seconds_per_step_median": round(seconds, 6),
            "steps_per_seed": sum(PROPOSED_STEPS.values())}

    audit = audit_design()
    return {"status": "passed",
            "reproduction": {"published_read": str(Path(published_path).resolve().relative_to(ROOT)),
                             "pooled_f1": reproduced,
                             "point": {lever: published["levers"][lever]["point"]
                                       for lever in LEVERS},
                             "standard_error": {lever: intervals[lever]["standard_error"]
                                                for lever in LEVERS}},
            "design": {"alpha": ALPHA, "power": POWER, "floor": FLOOR,
                       "audit_effect": AUDIT_EFFECT, "seeds": list(SEEDS),
                       "groups_per_seed": GROUPS_PER_SEED, "clusters": len(clusters),
                       "band_df": BAND_DF, "band_coverage": BAND_COVERAGE},
            "levers": effects,
            "decomposition": decomposition,
            "decomposition_deltas": deltas,
            "cost": cost,
            "audit": audit,
            "receipts": receipts,
            "sources": {name: digest(ROOT/name) for name in ANALYSIS_SOURCES},
            "scope": ("Read-only power analysis of the committed replicate block: reproducible "
                      "paired standard errors, the design effect those 36 clusters show, the "
                      "minimum detectable effect per lever, the seed count and cost a 256/384 block "
                      "would need across the 2 df band on the between-population variance, and a "
                      "ranking/boundary decomposition of the out-of-fold F1. No training, no gate, "
                      "no verdict change.")}


def headline(result):
    """Exact strings the roadmap and the continuation state must carry."""
    budget, schedule = result["levers"]["budget192"], result["levers"]["cosine128"]
    band = budget["variance"]["band"]
    floor = result["cost"]["budget192"]["floor"]
    return {"budget_mde": f"{budget['minimum_detectable_effect']:.6f}",
            "schedule_mde": f"{schedule['minimum_detectable_effect']:.6f}",
            "budget_power_at_floor": f"{budget['power_at_declared_floor']:.4f}",
            "schedule_power_at_floor": f"{schedule['power_at_declared_floor']:.4f}",
            "audit_mde": f"{result['audit']['minimum_detectable_effect']:.6f}",
            "audit_power": f"{result['audit']['power_at_claim']:.4f}",
            "budget_design_effect_cell": f"{budget['design_effect_cell']:.1f}",
            "schedule_design_effect_cell": f"{schedule['design_effect_cell']:.1f}",
            "band_seeds": f"{band['seeds']['low']} to {band['seeds']['high']}",
            "band_arm_hours": f"{band['arm_hours']['low']:.2f} to {band['arm_hours']['high']:.2f}",
            "floor_design": (f"{floor['seeds']} seeds, {floor['clusters']} clusters, "
                             f"{floor['arm_hours']:.2f} arm-hours")}


def render(result):
    """The figure block the method document embeds, generated from `result` alone."""
    begin, end = MARKERS
    lines = [begin, "",
             "| lever | bootstrap SE | delta-method SE | ratio | MDE | observed | inside the floor "
             "| power at a true +0.05 |",
             "|---|---|---|---|---|---|---|---|"]
    for lever in LEVERS:
        entry = result["levers"][lever]
        lines.append(f"| `{lever}` | {entry['standard_error_bootstrap']:.6f} | "
                     f"{entry['standard_error_analytic']:.6f} | "
                     f"{entry['analytic_over_bootstrap']:.4f} | "
                     f"**{entry['minimum_detectable_effect']:.6f}** | "
                     f"{entry['observed_point']:+.6f} | "
                     f"{'yes' if entry['inside_detectable_floor'] else 'no'} | "
                     f"{entry['power_at_declared_floor']:.4f} |")
    budget, schedule = result["levers"]["budget192"], result["levers"]["cosine128"]
    lines += ["",
              f"Design effect of the 36 clusters: `budget192` {budget['design_effect_board']:.1f} "
              f"against boards and {budget['design_effect_cell']:.1f} against cells; `cosine128` "
              f"{schedule['design_effect_board']:.1f} and {schedule['design_effect_cell']:.1f}.",
              "",
              f"Audit pair `{result['audit']['pair']}`, {result['audit']['groups']} groups: SE "
              f"`{result['audit']['standard_error']:.6f}`, MDE "
              f"`{result['audit']['minimum_detectable_effect']:.6f}`, power against its own claim "
              f"`{result['audit']['power_at_claim']:.4f}`, resolvable at its own design: "
              f"{'yes' if result['audit']['claim_resolvable_at_its_own_design'] else 'no'}.",
              "",
              "| target | seeds | clusters | arm-hours | SE | MDE |",
              "|---|---|---|---|---|---|"]
    labels = (("audit_effect", "+0.149, point estimate"), ("audit_low", "+0.149, band low"),
              ("audit_high", "+0.149, band high"), ("floor", "the +0.05 floor"))
    for lever in LEVERS:
        for key, label in labels:
            row = result["cost"][lever][key]
            lines.append(f"| `{lever}` {label} | {row['seeds']} | {row['clusters']} | "
                         f"{row['arm_hours']:.2f} | {row['standard_error']:.6f} | "
                         f"{row['minimum_detectable_effect']:.6f} |")
    band = budget["variance"]["band"]
    lines += ["",
              f"2 df band on the between-population variance: `{band['variance_low']:.6f}` to "
              f"`{band['variance_high']:.6f}` ({band['coverage']:.0%} scaled chi-square, df "
              f"{band['df']}), point `{band['variance']:.6f}`.",
              "",
              "| seed | lever | dAUC | d F1 at best boundary | d cost of `0.0` | d transfer cost "
              "| d out-of-fold F1 |",
              "|---|---|---|---|---|---|---|"]
    for lever in LEVERS:
        for seed in map(str, SEEDS):
            row = result["decomposition_deltas"][lever][seed]
            parts = row["parts"]
            lines.append(f"| `{seed}` | `{lever}` | {row['auc']:+.3f} | "
                         f"{parts['ranking_best_boundary']:+.3f} | "
                         f"{parts['zero_boundary_cost']:+.3f} | "
                         f"{parts['transfer_cost']:+.3f} | {row['cv_f1']:+.3f} |")
    lines += ["", end]
    return "\n".join(lines)+"\n"


def main(argv=None):
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render", action="store_true",
                        help="print the figure block for the method document and write nothing")
    args = parser.parse_args(argv)
    result = analyse()
    if args.render:
        print(render(result), end="")
        return 0
    if not args.output.resolve().is_relative_to((ROOT/"docs/evidence").resolve()):
        parser.error("analysis output belongs under docs/evidence")
    if args.output.exists():
        parser.error("analysis output must be a new evidence directory")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"result.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                                          encoding="utf-8")
    (args.output/"scope.json").write_text(json.dumps(
        {key: result[key] for key in ("status", "reproduction", "design", "levers", "cost",
                                      "audit", "sources", "scope")}, indent=2)+"\n",
        encoding="utf-8")
    print(json.dumps({"reproduction": result["reproduction"]["standard_error"],
                      "levers": {lever: {key: result["levers"][lever][key] for key in
                                         ("observed_magnitude", "minimum_detectable_effect",
                                          "inside_detectable_floor", "power_at_declared_floor",
                                          "design_effect_cell", "analytic_over_bootstrap")}
                                 for lever in LEVERS},
                      "band": {lever: result["levers"][lever]["variance"]["band"]["seeds"]
                               for lever in LEVERS},
                      "cost": result["cost"], "audit": result["audit"]}), flush=True)


if __name__ == "__main__":
    main()
