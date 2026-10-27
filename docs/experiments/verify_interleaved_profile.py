"""Exact same-input warm-profile comparison; permits only declared policy metadata change."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from probe_model import require_bounded_job
from record_stream_run import checkpoint_architecture
from source_snapshot import verify_source


def read(path):
    if path.stat().st_size > 1024**2:
        raise ValueError("finite metadata byte budget")
    return json.loads(path.read_text())


def sha(path):
    if path.stat().st_size > 64 * 1024**2:
        raise ValueError("finite artifact byte budget")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    if path.stat().st_size > 1024**2:
        raise ValueError("finite ledger byte budget")
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    require_bounded_job()
    names = ("m2-stream-recovery-gate-r5/profile", "m2-stream-interleave-profile-r1")
    policies = ("all-class-fit-v1", "all-class-interleaved-v1")
    runs, summaries, archives, inputs = [], [], [], {}
    for name, policy in zip(names, policies):
        run = ROOT / "runs" / name
        summary = read(run / "summary.json")
        job_path = run.parent / (run.name + "-job.json")
        job = read(job_path)
        if (job["status"] != "passed" or job["exit_code"] != 0 or job["work_items"] != 2 or
                job["memory_bytes"] != 640 * 1024**2 or summary["status"] != "complete" or summary["steps"] != 2 or
                not summary["profile_square"] or summary["curriculum"] != policy or
                summary["arguments"]["curriculum"] != policy or summary["recovery_contract"]["curriculum"] != policy or
                summary["config"].get("component_capacity", 0) != 0 or summary["config"].get("layer_evidence", False) or
                not summary["training_recovery_profile_exact"] or not summary["checkpoint_round_trip_exact"]):
            raise ValueError("actual same-input complete640MiB two-update policy profiles required")
        archive = verify_source(run / "code", summary["catalogue_sha256"])
        manifest = read(run / "checkpoint/manifest.json")
        checkpoint_architecture(summary, manifest)
        if sha(run / "checkpoint/weights.pt") != summary["checkpoint_sha256"] or manifest["weights_sha256"] != summary["checkpoint_sha256"]:
            raise ValueError("actual weight identity differs")
        runs.append(run)
        summaries.append(summary)
        archives.append(archive)
        inputs[name] = {file: sha(run / file) for file in
                        ("summary.json", "curve.jsonl", "training.jsonl", "training-probe.jsonl", "development-probe.jsonl",
                         "checkpoint/manifest.json", "checkpoint/weights.pt", "code/source.json")}
        inputs[name]["native-job.json"] = sha(job_path)
    for field in ("config", "checkpoint_sha256", "initial_training_probe", "initial_development", "final_training_probe", "development"):
        if summaries[0][field] != summaries[1][field]:
            raise ValueError("same-input numerical field differs: " + field)
    arguments = [{key: value for key, value in summary["arguments"].items() if key not in ("output", "curriculum")}
                 for summary in summaries]
    if arguments[0] != arguments[1]:
        raise ValueError("profile recipe differs beyond declared policy/output")
    for file, expected in (("training.jsonl", 2), ("training-probe.jsonl", 32), ("development-probe.jsonl", 9)):
        compared = []
        for run, policy in zip(runs, policies):
            records = rows(run / file)
            if len(records) != expected:
                raise ValueError("complete same-input ledger count differs")
            if file != "development-probe.jsonl":
                if any(record.get("curriculum") != policy for record in records):
                    raise ValueError("declared policy metadata missing")
                records = [{key: value for key, value in record.items() if key != "curriculum"} for record in records]
            compared.append(records)
        if compared[0] != compared[1]:
            raise ValueError("pixel/public-state/recipe identity differs beyond declared policy")
    curves = []
    for run in runs:
        curve = rows(run / "curve.jsonl")
        if [row["step"] for row in curve] != [1, 2]:
            raise ValueError("two-update global cursor differs")
        curves.append([{key: value for key, value in row.items() if key != "elapsed_seconds"} for row in curve])
    if curves[0] != curves[1]:
        raise ValueError("loss/terms/gradient/sample trajectory differs")
    packages = [{path: value for path, value in archive["files"].items() if path.startswith("tetris_vision/")} for archive in archives]
    if packages[0] != packages[1]:
        raise ValueError("neural package source differs")
    output = ROOT / "docs/evidence/interleave-profile-numerics-r1"
    output.mkdir(exist_ok=False)
    shutil.copyfile(__file__, output / "verify.py")
    proof = {"status": "passed", "runs": names, "policies": policies,
             "source_catalogues": [summary["catalogue_sha256"] for summary in summaries],
             "identical_actual_weights": True, "identical_neural_package_sources": True,
             "identical_all_four_probe_metric_objects": True, "identical_loss_terms_gradients_samples_excluding_elapsed": True,
             "identical_pixel_state_recipe_ledgers_except_declared_policy": True,
             "compared_optimizer_updates": 2, "compared_probe_recipes_per_run": 41,
             "weights_sha256": summaries[0]["checkpoint_sha256"], "input_sha256": inputs,
             "script_sha256": sha(Path(__file__)),
             "scope": "Same-input warm-profile numerical compatibility only. Interleaved training order, adequate learning and real accuracy are separate gates."}
    (output / "scope.json").write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "updates": 2}))


if __name__ == "__main__":
    main()
