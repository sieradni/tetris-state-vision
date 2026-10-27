"""Read-only matched-source synthetic diagnostic; no model selection claim."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
from probe_model import require_bounded_job
from record_stream_run import checkpoint_architecture, validate_completion
from source_snapshot import verify_source


def read(path):
    if path.stat().st_size > 4 * 1024**2:
        raise ValueError("JSON budget")
    return json.loads(path.read_text())


def sha(path):
    if path.stat().st_size > 64 * 1024**2:
        raise ValueError("file budget")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_run(name, enabled):
    run = ROOT / "runs" / name
    if run.parent != ROOT / "runs":
        raise ValueError("owned run required")
    summary = read(run / "summary.json")
    job = read(ROOT / "runs" / (name + "-job.json"))
    validate_completion(summary, job, allow_partial=True)
    manifest = read(run / "checkpoint/manifest.json")
    checkpoint_architecture(summary, manifest)
    verify_source(run / "code", summary["catalogue_sha256"])
    if summary["config"]["layer_evidence"] is not enabled:
        raise ValueError("actual option differs")
    if sha(run / "checkpoint/weights.pt") != summary["checkpoint_sha256"]:
        raise ValueError("actual weights differ")
    scope = read(ROOT / "docs/evidence" / name / "scope.json")
    if (scope["weights_sha256"] != summary["checkpoint_sha256"] or
            scope["source_catalogue_sha256"] != summary["catalogue_sha256"] or
            scope["steps"] != summary["steps"]):
        raise ValueError("curated identity differs")
    populations = {}
    for split, field in (("training", "final_training_probe"), ("development", "development")):
        score = summary[field]
        recipes = [json.loads(line) for line in (run / (split + "-probe.jsonl")).read_text().splitlines()]
        cases = {case["sample_id"]: case for case in score["cases"]}
        if len(cases) != len(recipes) or len(cases) != score["images"]:
            raise ValueError("probe population differs")
        counts = Counter()
        role_counts = {role: Counter() for role in ("active", "ghost")}
        for recipe in recipes:
            path = run / (split + "-probe") / (recipe["sample_id"] + ".json")
            if sha(path) != recipe["state_sha256"]:
                raise ValueError("supported state changed")
            state = read(path)
            cells = [cell for row in state["board"]["rows"] for cell in row]
            if len(cells) != 200 or any(type(cell["occupied"]) is not bool for cell in cells):
                raise ValueError("fully known visible board required")
            kind = "nonempty" if any(cell["occupied"] for cell in cells) else "empty"
            counts[kind + "_boards"] += 1
            counts["exact_" + kind + "_boards"] += int(cases[recipe["sample_id"]]["exact_locked_board"])
            for role in role_counts:
                layer = state[role]
                key = layer["piece"] if layer["status"] == "observed" else layer["status"]
                role_counts[role][key] += 1
        if (sum(c["exact_locked_board"] for c in cases.values()) != score["exact_locked_boards"] or
                sum(c["known_layers"] for c in cases.values()) != score["known_layers"] or
                sum(c["exact_known_layers"] for c in cases.values()) != score["exact_known_layers"]):
            raise ValueError("saved aggregate differs from cases")
        values = {key: score[key] for key in ("images", "ancestry_groups", "exact_locked_boards",
                  "cell_accuracy", "exact_known_layers", "known_layers", "mean_geometry_error_cells")}
        values.update(counts)
        values["supported_layer_classes_or_statuses"] = role_counts
        values.update({key: value for key, value in score.items() if key.startswith("dense_")})
        populations[split] = values
    curve = [json.loads(line) for line in (run / "curve.jsonl").read_text().splitlines()]
    if [row["step"] for row in curve] != list(range(1, summary["steps"] + 1)):
        raise ValueError("curve cursor differs")
    terms = {key: sum(row["terms"][key] for row in curve[-32:]) / len(curve[-32:])
             for key in curve[-1]["terms"]}
    if any(not math.isfinite(value) for value in terms.values()):
        raise ValueError("nonfinite terms")
    report = {"name": name, "steps": summary["steps"], "requested_steps": summary["requested_steps"],
              "status": summary["status"], "parameters": manifest["parameters"], "config": summary["config"],
              "weights_sha256": summary["checkpoint_sha256"], "source_catalogue_sha256": summary["catalogue_sha256"],
              "last_32_update_mean_terms": terms, "populations": populations,
              "native_training_peak_bytes": job["peak_job_commit_bytes"]}
    return run, summary, report


def main():
    require_bounded_job()
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    left, control, a = inspect_run(args.control, False)
    right, candidate, b = inspect_run(args.candidate, True)
    contracts = []
    for summary in (control, candidate):
        contract = json.loads(json.dumps(summary["recovery_contract"]))
        contract["config"].pop("layer_evidence")
        contracts.append(contract)
    if contracts[0] != contracts[1] or control["steps"] != candidate["steps"]:
        raise ValueError("matched recipe/source/step contract differs")
    ledgers = ("training.jsonl", "training-probe.jsonl", "development-probe.jsonl")
    if any((left / file).read_bytes() != (right / file).read_bytes() for file in ledgers):
        raise ValueError("matched data/pixel/label identities differ")
    if b["parameters"] - a["parameters"] != 2 * (candidate["config"]["dimension"] + 1):
        raise ValueError("unexpected parameter delta")
    common = set(a["last_32_update_mean_terms"]) & set(b["last_32_update_mean_terms"])
    output = {"status": "verified_matched_diagnostic", "control": a, "candidate": b,
              "identical_recovery_recipe_except_option": True, "identical_data_ledgers": True,
              "common_loss_terms": sorted(common), "ledger_sha256": {file: sha(left / file) for file in ledgers},
              "analysis_sha256": sha(Path(__file__)),
              "scope": "Same-source/data synthetic pilot. Total objectives differ; 18 development views cover six ancestries and omit classes/statuses. No convergence, real accuracy or model selection claim."}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "comparison.json").write_text(json.dumps(output, indent=2) + "\n")
    shutil.copyfile(__file__, args.output / "analysis-source.py")
    print(json.dumps({"status": output["status"], "steps": control["steps"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
