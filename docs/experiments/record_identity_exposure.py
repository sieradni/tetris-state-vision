"""Curate the identity-exposure audit: receipts, result and re-derived counts."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from probe_model import require_bounded_job  # noqa: E402

PIECES = tuple("IOTSZJL")


def read(path, limit=2 * 1024**2):
    if path.stat().st_size > limit:
        raise ValueError("curation JSON byte budget exceeded")
    return json.loads(path.read_text())


def rederive(result):
    """Recompute the exposure/correctness counts from the per-case rows."""
    splits = {}
    for split in ("training", "development"):
        rows = [row for row in result["cases"] if row["split"] == split]
        if not rows:
            raise ValueError("split present without audited cases")
        classes, positions, hold_classes = {}, {}, {}
        errors_by_class, errors_by_position = {}, {}
        correct = hold_correct = 0
        for row in rows:
            errors = set(row["error_indices"])
            for index, token in row["labels"]:
                name = PIECES[token]
                classes[name] = classes.get(name, 0) + 1
                positions[str(index)] = positions.get(str(index), 0) + 1
                if index in errors:
                    errors_by_class[name] = errors_by_class.get(name, 0) + 1
                    errors_by_position[str(index)] = errors_by_position.get(str(index), 0) + 1
                else:
                    correct += 1
            if row["hold_known"]:
                name = PIECES[row["hold_piece"]]
                hold_classes[name] = hold_classes.get(name, 0) + 1
                hold_correct += 1 - int(row["hold_wrong"])
        splits[split] = {
            "queue": {"supervised_piece_tokens": sum(classes.values()),
                      "classes": dict(sorted(classes.items())),
                      "positions": dict(sorted(positions.items(), key=lambda item: int(item[0]))),
                      "errors_by_class": dict(sorted(errors_by_class.items())),
                      "errors_by_position": dict(sorted(errors_by_position.items(), key=lambda item: int(item[0]))),
                      "correct_tokens": correct},
            "hold": {"supervised_pieces": sum(hold_classes.values()),
                     "classes": dict(sorted(hold_classes.items())), "correct_pieces": hold_correct}}
    return splits


def main():
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--retain", action="append", default=[], help="earlier receipt stem to preserve as a retained attempt")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9-]{1,64}", args.name) or not re.fullmatch(r"[a-z0-9-]{1,64}", args.stem):
        raise ValueError("finite owned evidence name and receipt stem required")
    runner = hashlib.sha256((ROOT / "scripts/run_bounded.py").read_bytes()).hexdigest()
    receipt = read(ROOT / "runs" / (args.stem + ".json"))
    if receipt["status"] != "passed" or receipt["work_items"] != 27:
        raise ValueError("complete 27-case audited receipt required")
    if receipt["supervisor_script_sha256"] != runner:
        raise ValueError("audit receipt was produced by another supervisor")
    output = ROOT / "runs" / args.stem
    result = read(output / "result.json")
    if result["status"] != "passed" or result["audit_format_version"] != 1 or result["images"] != 27:
        raise ValueError("complete audited result required")
    derived = rederive(result)
    for split, values in derived.items():
        for section, fields in values.items():
            reported = result["splits"][split][section]
            if any(reported[key] != value for key, value in fields.items()):
                raise ValueError("audit summary does not follow from its per-case rows")
    evidence = ROOT / "docs/evidence" / args.name
    evidence.mkdir(exist_ok=False)
    for suffix in (".json", ".log", ".stderr.log"):
        shutil.copyfile(ROOT / "runs" / (args.stem + suffix), evidence / (args.stem + suffix))
    shutil.copyfile(output / "result.json", evidence / "result.json")
    retained = {}
    for stem in args.retain:
        if not re.fullmatch(r"[a-z0-9-]{1,64}", stem) or stem == args.stem:
            raise ValueError("finite earlier attempt stem required")
        source = ROOT / "runs" / (stem + ".json")
        retained[stem] = hashlib.sha256(source.read_bytes()).hexdigest()
        shutil.copyfile(source, evidence / ("attempt-" + stem + ".json"))
    if hashlib.sha256((output / "result.json").read_bytes()).hexdigest() != \
            hashlib.sha256((evidence / "result.json").read_bytes()).hexdigest():
        raise ValueError("curated result bytes differ from the audited output")
    sources = {}
    for relative in ("docs/experiments/audit_identity_exposure.py", "docs/experiments/record_identity_exposure.py",
                     "scripts/run_bounded.py", "scripts/probe_model.py", "scripts/record_stream_run.py",
                     "scripts/source_snapshot.py"):
        sources[relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    scope = {
        "status": "passed", "audit_format_version": result["audit_format_version"],
        "run": result["run"], "run_status": result["run_status"], "curriculum": result["curriculum"],
        "component_capacity": result["component_capacity"], "images": result["images"],
        "source_catalogue_sha256": result["source_catalogue_sha256"], "weights_sha256": result["weights_sha256"],
        "published_diagnostics": result["published_diagnostics"],
        "published_result_sha256": result["published_result_sha256"],
        "published_counts_reproduced": result["published_counts_reproduced"],
        "splits": derived,
        "result_sha256": hashlib.sha256((evidence / "result.json").read_bytes()).hexdigest(),
        "audit_receipt": {"stem": args.stem, "status": receipt["status"], "work_items": receipt["work_items"],
                          "peak_job_commit_bytes": receipt["peak_job_commit_bytes"],
                          "declared_wall_seconds": receipt["wall_seconds"],
                          "declared_memory_bytes": receipt["memory_bytes"],
                          "elapsed_seconds": receipt["elapsed_seconds"]},
        "sources": sources, "supervisor_script_sha256": runner,
        "retained_attempts": retained,
        "scope": result["scope"],
        "curator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (evidence / "scope.json").write_text(json.dumps(scope, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "evidence": str(evidence), "images": result["images"],
                      "training_tokens": derived["training"]["queue"]["supervised_piece_tokens"],
                      "development_tokens": derived["development"]["queue"]["supervised_piece_tokens"]}))


if __name__ == "__main__":
    main()
