"""Confidence intervals reported beside the gate numbers; no verdict reads them.

The gates keep their predeclared thresholds and decide on counts, so nothing in
`fit_gate`, `percell_gate`, `occupancy_sweep` or `identity_gate` imports this module. An
interval is printed next to each deciding quantity so a reader can see how much of a gap the
population can resolve: Hanley-McNeil for a pooled AUC, Wilson score for a proportion. Both
are closed form, so a re-derivation is exact and needs no resampling.

An interval never upgrades or downgrades a verdict: a gate opens on its predeclared
threshold against the point estimate, exactly as before.
"""
import math

Z_95 = 1.959963984540054


def _critical(z):
    if type(z) not in (int, float) or isinstance(z, bool) or not z > 0:
        raise ValueError("positive critical value required")


def wilson_interval(successes, trials, z=Z_95):
    """Wilson score interval for a binomial proportion."""
    if (type(successes) is not int or isinstance(successes, bool) or type(trials) is not int or
            isinstance(trials, bool) or trials < 1 or not 0 <= successes <= trials):
        raise ValueError("bounded integer successes/trials required")
    _critical(z)
    estimate = successes/trials
    denominator = 1+z*z/trials
    centre = (estimate+z*z/(2*trials))/denominator
    half = z*math.sqrt(estimate*(1-estimate)/trials+z*z/(4*trials*trials))/denominator
    return {"successes": successes, "trials": trials, "estimate": round(estimate, 6),
            "low": round(max(0., centre-half), 6), "high": round(min(1., centre+half), 6)}


def auc_interval(auc, positives, negatives, z=Z_95):
    """Hanley-McNeil standard error and interval for a pooled AUC (chance .5)."""
    if type(auc) not in (int, float) or isinstance(auc, bool) or not 0 <= auc <= 1:
        raise ValueError("AUC in [0, 1] required")
    for value in (positives, negatives):
        if type(value) is not int or isinstance(value, bool) or value < 1:
            raise ValueError("positive integer class counts required")
    _critical(z)
    q1, q2 = auc/(2-auc), 2*auc*auc/(1+auc)
    variance = (auc*(1-auc)+(positives-1)*(q1-auc*auc)+(negatives-1)*(q2-auc*auc))/(positives*negatives)
    standard_error = math.sqrt(max(variance, 0.))
    return {"auc": round(float(auc), 6), "standard_error": round(standard_error, 6),
            "low": round(max(0., auc-z*standard_error), 6),
            "high": round(min(1., auc+z*standard_error), 6),
            "positives": positives, "negatives": negatives}


def f1_from_counts(true_positive, false_positive, false_negative):
    """2TP/(2TP+FP+FN), the gates' cell-wise quantity."""
    for value in (true_positive, false_positive, false_negative):
        if type(value) is not int or isinstance(value, bool) or value < 0:
            raise ValueError("nonnegative integer cell counts required")
    denominator = 2*true_positive+false_positive+false_negative
    if denominator == 0:
        raise ValueError("occupied-cell F1 needs occupied or predicted cells")
    return 2*true_positive/denominator


def precision_recall_interval(true_positive, false_positive, false_negative, z=Z_95):
    """Wilson intervals for the precision and recall that compose the F1 point.

    No interval is invented for F1 itself: the interval-bearing quantities are the two
    proportions behind it, and F1 is reported as the point the gates score.
    """
    if true_positive < 1:
        raise ValueError("precision/recall intervals need a positive prediction")
    return {"precision": wilson_interval(true_positive, true_positive+false_positive, z),
            "recall": wilson_interval(true_positive, true_positive+false_negative, z)}


def f1_gap(f1, threshold=0.70):
    """Signed distance to a predeclared bar: positive means short of it."""
    if type(f1) not in (int, float) or isinstance(f1, bool) or not 0 <= f1 <= 1:
        raise ValueError("F1 in [0, 1] required")
    if type(threshold) not in (int, float) or isinstance(threshold, bool) or not 0 <= threshold <= 1:
        raise ValueError("threshold in [0, 1] required")
    return round(threshold-f1, 6)
