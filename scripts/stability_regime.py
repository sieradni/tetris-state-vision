"""Clip-saturation census over committed training curves.

`docs/M3_OPTIMISER_REGIME.md` declares this instrument once. It is read-only: no model, no
training, no threshold and no label. It answers one question the campaign never asked of its
own record - whether the gradient clip ever *releases* - because `scripts/train_m2_stream.py`
and `scripts/train_m2_pilot.py` call `clip_grad_norm_(parameters, 1.)` on every step with no
learning-rate schedule anywhere in the tree.

The distinction matters because `clip_grad_norm_` rescales the gradient to the clip norm.
When the pre-clip norm exceeds the clip on every step, the update is `lr * unit direction` for
a constant `lr`: its magnitude cannot shrink near an optimum, whatever the curvature. V-085
measured that regime on the pilot harness (peak, then drift, then recovery under cosine decay).
This instrument measures whether the *published* record sits in the same regime, from the
`gradient_norm` field the trainer already writes, so the claim is a census of committed bytes
rather than an inference from loss curves.

Only stdlib is imported, so the census never needs a GPU, Torch or a thread policy.
"""
import json
import statistics
from pathlib import Path

CLIP_NORM = 1.0


def curve_norms(path):
    """Pre-clip gradient norms recorded in one curve file, in file order.

    Malformed lines, lines that are not objects and records without a finite numeric
    `gradient_norm` are skipped rather than coerced, so a corrupt file narrows the census
    instead of inventing a step.
    """
    values = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict):
            continue
        value = record.get("gradient_norm")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        value = float(value)
        if value != value or value in (float("inf"), float("-inf")):
            continue
        values.append(value)
    return values


def census(runs):
    """Summarise `{run name: [norms]}`; runs with no recorded step are omitted."""
    steps = [value for values in runs.values() for value in values]
    if not steps:
        return {"runs": 0, "files": 0, "steps": 0, "clipped_steps": 0, "saturation": None,
                "median_norm": None, "min_norm": None, "max_norm": None,
                "unclipped_runs": 0, "fully_clipped_runs": 0}
    clipped = [value for value in steps if value > CLIP_NORM]
    fully = [name for name, values in runs.items() if values and all(v > CLIP_NORM for v in values)]
    return {"runs": len(runs), "files": len(runs), "steps": len(steps),
            "clipped_steps": len(clipped), "saturation": len(clipped)/len(steps),
            "median_norm": statistics.median(steps), "min_norm": min(steps),
            "max_norm": max(steps), "unclipped_runs": len(runs)-len(fully),
            "fully_clipped_runs": len(fully)}


def collect(root):
    """Census every `*.jsonl` under `root`, grouped by parent directory."""
    runs = {}
    for path in sorted(Path(root).rglob("*.jsonl")):
        values = curve_norms(path)
        if values:
            runs[str(path.parent.relative_to(root)).replace("\\", "/")] = (
                runs.get(str(path.parent.relative_to(root)).replace("\\", "/"), [])) + values
    return runs


def report(root="docs/evidence"):
    """The measured census of `root`, as JSON-serialisable data."""
    runs = collect(root)
    summary = census(runs)
    summary["root"] = str(root).replace("\\", "/")
    summary["clip_norm"] = CLIP_NORM
    summary["per_run"] = {name: census({name: values}) for name, values in sorted(runs.items())}
    return summary


def main(argv=None):
    import sys
    arguments = list(sys.argv[1:] if argv is None else argv)
    root = arguments[0] if arguments else "docs/evidence"
    summary = report(root)
    print(json.dumps({key: value for key, value in summary.items() if key != "per_run"},
                     indent=2, allow_nan=False))
    print(f"\nper-run census ({len(summary['per_run'])} runs):")
    for name, row in summary["per_run"].items():
        print(f"  {name:52s} steps {row['steps']:6d} "
              f"saturation {row['saturation']*100:6.2f}% median {row['median_norm']:9.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
