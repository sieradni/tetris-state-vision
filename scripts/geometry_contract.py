"""Pure contracts for explicitly conditional M2 lattice diagnostics."""
import math

from record_stream_run import checkpoint_architecture, validate_completion

REPORT_SOURCES = tuple("scripts/"+name for name in
    ("probe_stream_geometry.py", "geometry_contract.py", "record_stream_run.py", "source_snapshot.py", "probe_model.py"))


def input_contract(summary, job, manifest, *, allow_partial=False):
    validate_completion(summary, job, allow_partial=allow_partial)
    architecture = checkpoint_architecture(summary, manifest)
    if (summary["config"].get("component_capacity", 0) != 0 or architecture != "TetrisStateNet-M2" or
            manifest["weights_sha256"] != summary["checkpoint_sha256"]):
        raise ValueError("M2-only artifact/weight identity required for this diagnostic")
    return architecture


def summarize(records):
    if type(records) is not list or len(records) != 27:
        raise ValueError("exact27-case conditional diagnostic required")
    if (any(row["split"] not in {"training", "development"} or type(row["sample_id"]) is not str or
            not row["sample_id"] or type(row["group_id"]) is not str or not row["group_id"] for row in records) or
            len({row["sample_id"] for row in records}) != 27):
        raise ValueError("unique case identities and explicit split/ancestry required")
    totals = {}
    for split, count in (("training", 9), ("development", 18)):
        cases = [row for row in records if row["split"] == split]
        if len(cases) != count:
            raise ValueError("declared9+18 population differs")
        totals[split] = {"images": count, "ancestry_groups": len({row["group_id"] for row in cases})}
        for mode in ("automatic", "oracle_geometry"):
            for row in cases:
                metric = row[mode]
                if (set(metric) != {"known_cells", "cell_accuracy", "exact_locked_board", "note", "exact_known_layers", "known_layers"} or
                        type(metric["known_cells"]) is not int or metric["known_cells"] != 200 or
                        type(metric["cell_accuracy"]) not in (int, float) or not math.isfinite(metric["cell_accuracy"]) or
                        not 0 <= metric["cell_accuracy"] <= 1 or type(metric["exact_locked_board"]) is not bool or
                        metric["exact_locked_board"] != (metric["cell_accuracy"] == 1) or
                        type(metric["known_layers"]) is not int or not 0 <= metric["known_layers"] <= 2 or
                        type(metric["exact_known_layers"]) is not int or not 0 <= metric["exact_known_layers"] <= metric["known_layers"]):
                    raise ValueError("finite supported diagnostic metrics required; oracle geometry error is excluded")
            totals[split][mode] = {"exact_locked_boards": sum(row[mode]["exact_locked_board"] for row in cases),
                                   "cell_accuracy": sum(row[mode]["cell_accuracy"] for row in cases)/count,
                                   "exact_known_layers": sum(row[mode]["exact_known_layers"] for row in cases),
                                   "known_layers": sum(row[mode]["known_layers"] for row in cases)}
    return totals
