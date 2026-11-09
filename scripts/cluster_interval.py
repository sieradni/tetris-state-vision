"""Correlation-respecting intervals for clustered probe cells (methodological addendum).

`docs/M3_WIDENED_DECIDING_POPULATION.md` reports cell-level intervals (Wilson, Hanley-McNeil)
beside each gate number. Those formulas treat every known cell as an independent observation,
while the widened deciding population is 667 occupied and 4,933 empty cells drawn from 28
nonempty boards nested in 12 ancestry groups, and the repo's own scope lines already say
"matched views share ancestry; no independent-sample interval".

This module measures that clustering and supplies the interval that respects it: a one-way
ICC(1) of the per-cell error indicator, its design effect and effective sample size, a
whole-cluster percentile bootstrap, and a design-effect-reduced Wilson interval for rates. No
gate imports this module - the rules read point estimates - so nothing here can move a verdict.
"""
import math
import random
import statistics

BOOTSTRAP_SEED = 20261026
AUC_DRAWS = 400
COUNT_DRAWS = 1000
Z_95 = 1.959963984540054


def clusters_of(rows, key):
    """Sorted (cluster_id, [rows]) pairs, so a seeded bootstrap is exactly reproducible."""
    grouped = {}
    for row in rows:
        if type(row) is not dict or key not in row:
            raise ValueError("rows with an explicit cluster key required")
        grouped.setdefault(row[key], []).append(row)
    return tuple(sorted(grouped.items()))


def intra_cluster_correlation(values, cluster_ids):
    """One-way ANOVA ICC(1) for unbalanced clusters; a binary outcome is a valid variance case.

    Returns 0.0 when every cluster has the same mean (including the all-zero case), because no
    between-cluster variance is then estimable rather than because clustering is absent.
    """
    if type(values) is not list or type(cluster_ids) is not list or not values or len(values) != len(cluster_ids):
        raise ValueError("paired values and cluster ids required")
    grouped = {}
    for value, cluster in zip(values, cluster_ids):
        if type(value) not in (int, float) or isinstance(value, bool) or value != value:
            raise ValueError("finite numeric outcome required")
        grouped.setdefault(cluster, []).append(float(value))
    count, total = len(values), float(sum(values))
    clusters = len(grouped)
    if clusters < 2 or count <= clusters:
        raise ValueError("at least two clusters and one within-cluster pair required")
    grand = total/count
    means = {name: sum(group)/len(group) for name, group in grouped.items()}
    ss_between = sum(len(group)*(means[name]-grand)**2 for name, group in grouped.items())
    ss_within = sum((value-means[name])**2 for name, group in grouped.items() for value in group)
    ms_between, ms_within = ss_between/(clusters-1), ss_within/(count-clusters)
    n0 = (count-sum(len(group)**2 for group in grouped.values())/count)/(clusters-1)
    denominator = ms_between+(n0-1)*ms_within
    return 0.0 if denominator <= 0 else (ms_between-ms_within)/denominator


def design_effect(icc, mean_cluster_size):
    """1 + (mean cluster size - 1) * ICC; the classic cluster design effect."""
    if type(icc) not in (int, float) or isinstance(icc, bool):
        raise ValueError("numeric ICC required")
    if type(mean_cluster_size) not in (int, float) or mean_cluster_size < 1:
        raise ValueError("mean cluster size >= 1 required")
    return 1+(mean_cluster_size-1)*max(0.0, icc)


def effective_size(size, deff):
    if type(size) not in (int, float) or type(deff) not in (int, float) or not deff > 0:
        raise ValueError("numeric size and positive design effect required")
    return size/deff


def rate_design_effect(clusters):
    """Variance inflation of a proportion from between-cluster rate dispersion.

    ``clusters`` is a sequence of ``(trials, successes)``. A proportion at 0 or 1 has no
    estimable between-cluster variance and returns 1.0, so a degenerate rate is never given a
    fabricated design effect.
    """
    pairs = []
    for entry in clusters:
        if type(entry) not in (tuple, list) or len(entry) != 2:
            raise ValueError("(trials, successes) pairs required")
        trials, successes = entry
        if (type(trials) is not int or type(successes) is not int or trials < 0 or
                not 0 <= successes <= trials):
            raise ValueError("bounded integer cluster counts required")
        if trials:
            # A cluster with no supervised trial contributes no dispersion and no weight.
            pairs.append((trials, successes))
    if len(pairs) < 2:
        raise ValueError("at least two clusters with trials required")
    total, hits = sum(trials for trials, _ in pairs), sum(successes for _, successes in pairs)
    if total <= len(pairs):
        raise ValueError("more trials than clusters required")
    estimate = hits/total
    if estimate <= 0 or estimate >= 1:
        return 1.0
    mean_square = sum(trials*(successes/trials-estimate)**2 for trials, successes in pairs)/(len(pairs)-1)
    return max(1.0, mean_square/(estimate*(1-estimate)))


def percentile(ordered, q):
    if not ordered or not 0 <= q <= 1:
        raise ValueError("nonempty ordered values and q in [0, 1] required")
    position = (len(ordered)-1)*q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low]*(high-position)+ordered[high]*(position-low)


def cluster_bootstrap(statistic, clusters, keys, *, draws=COUNT_DRAWS, seed=BOOTSTRAP_SEED):
    """Percentile interval from resampling whole clusters; degenerate draws are skipped."""
    if not callable(statistic) or type(clusters) is not tuple or len(clusters) < 2:
        raise ValueError("statistic and at least two sorted clusters required")
    if type(draws) is not int or not 100 <= draws <= 5000:
        raise ValueError("draws100..5000 required")
    if type(keys) is not tuple or not keys:
        raise ValueError("at least one statistic key required")
    rng, collected = random.Random(seed), {key: [] for key in keys}
    for _ in range(draws):
        sample = [clusters[rng.randrange(len(clusters))][1] for _ in range(len(clusters))]
        value = statistic(sample)
        if value is None:
            continue
        for key in keys:
            if value.get(key) is not None and value[key] == value[key]:
                collected[key].append(float(value[key]))
    result = {"draws": draws, "seed": seed, "clusters": len(clusters)}
    for key, values in collected.items():
        if len(values) < draws*0.5:
            raise ValueError("too many degenerate bootstrap draws")
        values.sort()
        result[key] = {"valid_draws": len(values), "low": round(percentile(values, 0.025), 6),
                       "high": round(percentile(values, 0.975), 6),
                       "median": round(percentile(values, 0.5), 6),
                       "standard_error": round(statistics.pstdev(values), 6)}
    return result


def wilson_interval_with_design_effect(successes, trials, deff, z=Z_95):
    """Wilson interval evaluated at the design-effect-reduced sample size."""
    if (type(successes) is not int or type(trials) is not int or type(successes) is bool or
            trials < 1 or not 0 <= successes <= trials):
        raise ValueError("bounded integer successes/trials required")
    if type(deff) not in (int, float) or not deff >= 1:
        raise ValueError("design effect >= 1 required")
    estimate, size = successes/trials, trials/deff
    denominator = 1+z*z/size
    centre = (estimate+z*z/(2*size))/denominator
    half = z*math.sqrt(estimate*(1-estimate)/size+z*z/(4*size*size))/denominator
    return {"successes": successes, "trials": trials, "estimate": round(estimate, 6),
            "low": round(max(0., centre-half), 6), "high": round(min(1., centre+half), 6),
            "design_effect": round(deff, 6), "effective_trials": round(size, 3)}
