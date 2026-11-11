"""Read the declared replicate block: do the two compute-side levers survive replication?

`docs/M3_COMPUTE_REPLICATION.md` fixes the block, the primary read, the floor and the rule; this
module is its single implementation. It reads the nine committed sweep receipts (three arms across
three seeds), requires each to reproduce its own published read under the nonempty-board convention,
pairs the arms within a seed by ancestry group, and bootstraps whole `(seed, group)` clusters with one
draw shared by all arms. Read-only: no training, no weight change, no gate.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

from occupancy_sweep import pooled_auc
from operating_point import cross_validated, f1_from_counts, nonempty, percentile
from probe_model import require_bounded_job

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20261101, 20261102, 20261103)
ARMS = {"base128": 128, "budget192": 192, "cosine128": 128}
LEVERS = ("budget192", "cosine128")
CONTROL = "base128"
DECIDING_GROUPS, DECIDING_CASES, KNOWN_CELLS = 12, 36, 200
FLOOR, AUC_FLOOR = 0.05, 0.70
BOOTSTRAP_SEED, BOOTSTRAP_DRAWS = 20261026, 1000
READ_SOURCES = tuple("scripts/"+name for name in
                     ("compute_replication.py", "operating_point.py", "occupancy_sweep.py",
                      "probe_model.py"))


def digest(path, maximum=64*1024**2):
    value, consumed = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            consumed += len(block)
            if consumed > maximum:
                raise ValueError("read source byte budget exceeded")
            value.update(block)
    return value.hexdigest()


def arm_run(arm, seed):
    return f"m3-stream-repl-{arm}-s{seed}"


def load_arm(arm, seed):
    """One sweep receipt, guarded on its population, its status and its own published read."""
    run = arm_run(arm, seed)
    path = ROOT/"runs"/(run+"-sweep-r1")/"result.json"
    if not path.is_file():
        raise ValueError("missing sweep receipt: "+str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != "passed" or data.get("run") != run:
        raise ValueError("not a passed sweep receipt of its own run: "+str(path))
    cases = [case for case in data.get("cases", []) if case.get("split") == "training"]
    groups = {case.get("group_id") for case in cases}
    if (len(cases) != DECIDING_CASES or len(groups) != DECIDING_GROUPS or
            any(len(case.get("logits", [])) != KNOWN_CELLS or
                len(case.get("occupied", [])) != KNOWN_CELLS for case in cases)):
        raise ValueError("not the declared 12-group, 36-board population: "+str(path))
    published = data.get("deciding", {})
    rows = {row.get("threshold"): row for row in published.get("rows", [])}
    if 0.0 not in rows:
        raise ValueError("no threshold 0.0 row: "+str(path))
    scored = nonempty(cases)
    positive = [v for c in scored for v, label in zip(c["logits"], c["occupied"]) if label]
    negative = [v for c in scored for v, label in zip(c["logits"], c["occupied"]) if not label]
    if (abs(published.get("auc", -1)-pooled_auc(positive, negative)) > 1e-9 or
            abs(rows[0.0].get("f1", -1)-f1_from_counts(*counts_of(scored, 0.0))) > 1e-9):
        raise ValueError("receipt does not reproduce its own published read: "+str(path))
    return {"run": run, "path": str(path.relative_to(ROOT)), "sha256": digest(path),
            "weights_sha256": data.get("weights_sha256"), "cases": cases,
            "auc": pooled_auc(positive, negative),
            "published": {"auc": published["auc"], "f1_at_zero": rows[0.0]["f1"]}}


def counts_of(cases, threshold):
    from operating_point import counts
    return counts(cases, threshold)


def require_one_population(arms, seed):
    """The three arms of a seed must score the same boards with the same labels."""
    first = arms[CONTROL]["cases"]
    key = [(case["sample_id"], tuple(case["occupied"])) for case in first]
    for arm in ARMS:
        other = [(case["sample_id"], tuple(case["occupied"])) for case in arms[arm]["cases"]]
        if other != key:
            raise ValueError(f"{seed}: {arm} scored a different board population than the control")
    return key


def block_clusters(reads):
    """One row per `(seed, group)` cluster: each arm's held-out counts on that group's boards."""
    clusters = []
    for seed in SEEDS:
        folds = {arm: {fold["group_id"]: fold for fold in reads[(seed, arm)]} for arm in ARMS}
        if len({tuple(sorted(folds[arm])) for arm in ARMS}) != 1:
            raise ValueError(f"{seed}: arms did not fold the same ancestry groups")
        for group in sorted(folds[CONTROL]):
            row = {"seed": seed, "group_id": group}
            for arm in ARMS:
                fold = folds[arm][group]
                row[arm] = (fold["true_positive"], fold["false_positive"], fold["false_negative"])
            clusters.append(row)
    return clusters


def pooled_f1(clusters, arm):
    totals = [sum(cluster[arm][slot] for cluster in clusters) for slot in range(3)]
    return f1_from_counts(*totals)


def paired_delta_interval(clusters, lever, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """Paired whole-cluster percentile interval for `f1(lever) - f1(control)`."""
    if type(clusters) is not list or len(clusters) < 6:
        raise ValueError("at least six clusters required")
    if type(draws) is not int or not 100 <= draws <= 5000:
        raise ValueError("draws 100..5000 required")
    rng, values = random.Random(seed), []
    for _ in range(draws):
        lever_total, control_total = [0, 0, 0], [0, 0, 0]
        for _ in range(len(clusters)):
            cluster = clusters[rng.randrange(len(clusters))]
            for slot in range(3):
                lever_total[slot] += cluster[lever][slot]
                control_total[slot] += cluster[CONTROL][slot]
        if (2*lever_total[0]+lever_total[1]+lever_total[2] and
                2*control_total[0]+control_total[1]+control_total[2]):
            values.append(f1_from_counts(*lever_total)-f1_from_counts(*control_total))
    if len(values) < draws*0.5:
        raise ValueError("too many degenerate draws")
    values.sort()
    low, high = percentile(values, 0.025), percentile(values, 0.975)
    return {"draws": draws, "seed": seed, "clusters": len(clusters),
            "low": round(low, 6), "median": round(percentile(values, 0.5), 6),
            "high": round(high, 6), "standard_error": round(pstdev(values), 6),
            "excludes_zero": bool(low > 0 or high < 0)}


def pstdev(values):
    import statistics
    return statistics.pstdev(values)


def replication_verdict(control_aucs, per_seed_deltas, point, interval, floor=FLOOR,
                        auc_floor=AUC_FLOOR):
    """The predeclared three-condition rule; no branch is chosen after the numbers."""
    if (type(control_aucs) is not dict or type(per_seed_deltas) is not dict or
            type(interval) is not dict or not per_seed_deltas or
            set(control_aucs) != set(per_seed_deltas)):
        raise ValueError("paired per-seed controls and deltas required")
    if type(point) not in (int, float) or point != point:
        raise ValueError("finite point delta required")
    low = interval.get("low")
    if type(low) not in (int, float):
        raise ValueError("interval low bound required")
    behind = sorted(seed for seed, auc in control_aucs.items() if auc < auc_floor)
    if behind:
        return {"label": "floor_limited", "reason": "control_auc_below_floor",
                "seeds": behind, "auc_floor": auc_floor, "floor": floor}
    reasons = []
    if point < floor:
        reasons.append("point_below_floor")
    if low <= 0:
        reasons.append("interval_touches_zero")
    if any(delta < 0 for delta in per_seed_deltas.values()):
        reasons.append("seed_reverses")
    if not reasons:
        return {"label": "confirmed", "reason": "point_clears_floor_and_interval_excludes_zero"
                "_and_every_seed_agrees", "auc_floor": auc_floor, "floor": floor}
    return {"label": "unconfirmed", "reason": "+".join(reasons), "auc_floor": auc_floor,
            "floor": floor}


def main(argv=None):
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.output.resolve().is_relative_to((ROOT/"docs/evidence").resolve()):
        parser.error("read output belongs under docs/evidence")
    if args.output.exists():
        parser.error("read output must be a new evidence directory")
    reads, receipts, arms = {}, {}, {}
    for seed in SEEDS:
        for arm in ARMS:
            data = load_arm(arm, seed)
            reads[(seed, arm)] = cross_validated(data["cases"])["folds"]
            receipts[f"{arm}-s{seed}"] = {key: data[key]
                                          for key in ("path", "sha256", "weights_sha256",
                                                      "published", "auc")}
            arms[(seed, arm)] = data
        require_one_population({arm: arms[(seed, arm)] for arm in ARMS}, seed)
    clusters = block_clusters(reads)
    if len(clusters) != len(SEEDS)*DECIDING_GROUPS:
        raise ValueError("block did not yield one cluster per seed and group")
    control_aucs = {seed: arms[(seed, CONTROL)]["auc"] for seed in SEEDS}
    levers = {}
    for lever in LEVERS:
        per_seed = {}
        for seed in SEEDS:
            seed_clusters = [cluster for cluster in clusters if cluster["seed"] == seed]
            per_seed[seed] = round(pooled_f1(seed_clusters, lever)-
                                   pooled_f1(seed_clusters, CONTROL), 6)
        point = pooled_f1(clusters, lever)-pooled_f1(clusters, CONTROL)
        interval = paired_delta_interval(clusters, lever)
        levers[lever] = {"point": round(point, 6), "per_seed": per_seed,
                         "interval": interval,
                         "pooled_f1": {arm: round(pooled_f1(clusters, arm), 6) for arm in ARMS},
                         "verdict": replication_verdict(control_aucs, per_seed, point, interval)}
    per_seed_table = {}
    for seed in SEEDS:
        seed_clusters = [cluster for cluster in clusters if cluster["seed"] == seed]
        per_seed_table[seed] = {arm: round(pooled_f1(seed_clusters, arm), 6) for arm in ARMS}
    result = {"status": "passed", "block": {"seeds": list(SEEDS), "arms": ARMS,
                                            "steps": {arm: ARMS[arm] for arm in ARMS},
                                            "clusters": len(clusters),
                                            "boards_per_seed": DECIDING_CASES},
              "control_auc": {str(seed): round(auc, 6) for seed, auc in control_aucs.items()},
              "per_seed_f1": per_seed_table,
              "levers": levers,
              "receipts": receipts,
              "cluster_counts": [{"seed": cluster["seed"], "group_id": cluster["group_id"],
                                  **{arm: list(cluster[arm]) for arm in ARMS}}
                                 for cluster in clusters],
              "sources": {name: digest(ROOT/name) for name in READ_SOURCES},
              "scope": ("Read-only replicate read of the declared three-seed block: out-of-fold F1 "
                        "per arm, paired whole-cluster interval, rule fixed before the receipts. No "
                        "training, no gate, no quality claim.")}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"result.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                                          encoding="utf-8")
    (args.output/"scope.json").write_text(json.dumps(
        {key: result[key] for key in ("status", "block", "control_auc", "per_seed_f1", "levers",
                                      "receipts", "sources", "scope")}, indent=2)+"\n",
        encoding="utf-8")
    print(json.dumps({"control_auc": result["control_auc"], "per_seed_f1": per_seed_table,
                      "levers": {name: {key: value[key] for key in
                                        ("point", "per_seed", "interval", "verdict")}
                                 for name, value in levers.items()}}), flush=True)


if __name__ == "__main__":
    main()
