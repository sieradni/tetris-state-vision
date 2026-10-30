"""Finite, label-safe metadata and independent metrics for head-only diagnostics."""
from collections import Counter
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from record_stream_errors import digest, read
from record_stream_run import checkpoint_architecture, validate_completion
from source_snapshot import verify_source

RECIPE = {"format": "frozen-feature-type-head-v1", "steps": 1024,
          "optimizer": "AdamW", "learning_rate": .01, "weight_decay": 0.,
          "betas": [.9, .999], "eps": 1e-8, "dtype": "float64",
          "normalization": "training-only population mean/std", "scale_floor": 1e-6,
          "batch": 64, "classes": 8, "profile_steps": 2, "profile_work": 4}
STUDIES = {"ordered": "allclass1024", "interleaved": "interleave1024"}


def validate_rows(rows, dimension):
    if type(dimension) is not int or not 1 <= dimension <= 256 or len(rows) != 82:
        raise ValueError("finite82-vector population required")
    seen = set()
    for row in rows:
        identity = (row["split"], row["sample_id"], row["role"])
        if (identity in seen or row["split"] not in ("training", "development") or
                row["role"] not in ("active", "ghost") or not row["group_id"] or
                type(row["known"]) is not bool or type(row["target_type"]) is not int or
                not 0 <= row["target_type"] < 8 or len(row["features"]) != dimension or
                len(row["logits"]) != 8 or any(type(x) not in (int, float) or not math.isfinite(x)
                    for x in row["features"] + row["logits"])):
            raise ValueError("unique finite supported role-vector contract required")
        seen.add(identity)
    for role in ("active", "ghost"):
        training = [row for row in rows if row["split"] == "training" and row["role"] == role and row["known"]]
        development = [row for row in rows if row["split"] == "development" and row["role"] == role]
        if (len(training) != 32 or len(development) != 9 or
                Counter(row["target_type"] for row in training) != Counter({i: 4 for i in range(8)}) or
                len({row["group_id"] for row in training}) != 16 or any(
                    len({row["group_id"] for row in training if row["target_type"] == label}) != 2 for label in range(8))):
            raise ValueError("balanced64 known training labels/two ancestries per class required")
    for index in range(0, 82, 2):
        active, ghost = rows[index:index+2]
        if (active["role"] != "active" or ghost["role"] != "ghost" or
                any(active[key] != ghost[key] for key in ("split", "sample_id", "group_id"))):
            raise ValueError("paired image/role order differs")


def metrics(rows, logits):
    if len(rows) != len(logits) or any(len(values) != 8 or not all(math.isfinite(x) for x in values) for values in logits):
        raise ValueError("complete finite8-class logits required")
    result = {}
    for split in ("training", "development"):
        population = [(row, values) for row, values in zip(rows, logits) if row["split"] == split and row["known"]]
        cases = []
        for row, values in population:
            largest = max(values)
            loss = largest + math.log(math.fsum(math.exp(value-largest) for value in values))-values[row["target_type"]]
            cases.append({"sample_id": row["sample_id"], "group_id": row["group_id"], "role": row["role"],
                          "target": row["target_type"], "predicted": max(range(8), key=values.__getitem__), "loss": loss})
        result[split] = {"known": len(cases), "unknown_excluded": sum(row["split"] == split and not row["known"] for row in rows),
                         "ancestries": len({row["group_id"] for row, _ in population}),
                         "correct": sum(case["target"] == case["predicted"] for case in cases),
                         "mean_cross_entropy": math.fsum(case["loss"] for case in cases)/len(cases) if cases else None,
                         "by_role": {role: {"known": sum(case["role"] == role for case in cases),
                                            "correct": sum(case["role"] == role and case["target"] == case["predicted"] for case in cases)}
                                     for role in ("active", "ghost")}, "cases": cases}
    return result


def training_indices(rows):
    return [index for index, row in enumerate(rows) if row["split"] == "training" and row["known"]]


def load_bank(study):
    name = "m2-stream-"+STUDIES[study]+"-features-r1"
    evidence = ROOT / "docs/evidence" / name
    scope, bank = read(evidence / "scope.json"), read(evidence / "features.json")
    receipt_scope = read(evidence / "curation-receipt-scope.json")
    curator = read(evidence / "curation-job.json")
    if (receipt_scope["status"] != "passed" or receipt_scope["forwards"] != 41 or
            curator["status"] != "passed" or curator["exit_code"] != 0 or
            curator["work_items"] != 41 or curator["memory_bytes"] != 128*1024**2):
        raise ValueError("native independent feature collector must have passed")
    for relative, expected in {**scope["artifact_sha256"], **receipt_scope["artifact_sha256"]}.items():
        path = (evidence / relative).resolve()
        if not path.is_relative_to(evidence.resolve()) or digest(path) != expected:
            raise ValueError("curated feature evidence bytes differ")
    model = ROOT / "runs" / scope["run"]
    summary, manifest = read(model / "summary.json"), read(model / "checkpoint/manifest.json")
    validate_completion(summary, read(ROOT / "runs" / (model.name+"-job.json")))
    architecture = checkpoint_architecture(summary, manifest)
    verify_source(model / "code", bank["source_catalogue_sha256"])
    if (scope["status"] != "passed" or bank["status"] != "passed" or scope["run"] != "m2-stream-"+STUDIES[study]+"-s3" or
            scope["global_steps"] != 1024 or scope["target_steps"] != 1024 or scope["forwards"] != 41 or
            bank["weights_unchanged"] is not True or bank["exact_linear_reconstruction"] is not True or
            scope["observer_ordinary_cases_and_totals_identical"] is not True or scope["grouped_analysis_recomputed"] is not True or
            bank["weights_sha256"] != scope["weights_sha256"] or bank["weights_sha256"] != manifest["weights_sha256"] or
            bank["source_catalogue_sha256"] != summary["catalogue_sha256"] or
            bank["core_architecture"] != architecture or bank["config"] != manifest["config"] or
            manifest["config"]["dimension"] != 64 or manifest["config"].get("component_capacity", 0) or
            manifest["config"].get("layer_evidence", False) or digest(model / "checkpoint/weights.pt") != bank["weights_sha256"]):
        raise ValueError("actual complete controlled M2 feature/head identity differs")
    validate_rows(bank["features"], manifest["config"]["dimension"])
    files = [evidence / "scope.json", evidence / "features.json", evidence / "curation-receipt-scope.json",
             evidence / "curation-job.json", model / "summary.json",
             model / "checkpoint/manifest.json", model / "checkpoint/weights.pt", model / "code/source.json"]
    identities = {path.relative_to(ROOT).as_posix(): digest(path) for path in files}
    return bank, model, identities
