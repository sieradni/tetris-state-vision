"""Bind the actual128-update comparison to matching starts/data and different order."""
from collections import Counter
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
    if path.stat().st_size > 2 * 1024**2:
        raise ValueError("metadata byte budget")
    return json.loads(path.read_text())


def rows(path, count):
    if path.stat().st_size > 2 * 1024**2:
        raise ValueError("ledger byte budget")
    values = [json.loads(line) for line in path.read_text().splitlines()]
    if len(values) != count:
        raise ValueError("exact finite ledger count differs")
    return values


def sha(path):
    if path.stat().st_size > 64 * 1024**2:
        raise ValueError("artifact byte budget")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    require_bounded_job()
    names = ("m2-stream-allclass1024-s1", "m2-stream-interleave1024-s1")
    policies = ("all-class-fit-v1", "all-class-interleaved-v1")
    runs, summaries, archives, training, inputs = [], [], [], [], {}
    for name, policy in zip(names, policies):
        run = ROOT / "runs" / name
        summary = read(run / "summary.json")
        job = read(ROOT / "runs" / (name + "-job.json"))
        parent = read(ROOT / "runs" / (name + "-pipeline-job.json"))
        scope = read(ROOT / "docs/evidence" / name / "scope.json")
        manifest = read(run / "checkpoint/manifest.json")
        if (summary["status"] != "step_partial" or summary["steps"] != 128 or summary["requested_steps"] != 1024 or
                summary["updates_this_run"] != 128 or summary["resume"] is not None or summary["curriculum"] != policy or
                summary["model_kind"] != "pilot" or job["status"] != "failed" or job["exit_code"] != 2 or
                job["work_items"] != 128 or job["memory_bytes"] != 640 * 1024**2 or parent["status"] != "passed" or
                scope["status"] != "verified_partial" or scope["steps"] != 128 or scope["verified_saved_probe_images"] != 41 or
                scope["source_catalogue_sha256"] != summary["catalogue_sha256"] or
                scope["weights_sha256"] != summary["checkpoint_sha256"] or
                checkpoint_architecture(summary, manifest) != "TetrisStateNet-M2"):
            raise ValueError("actual curated fresh128/1024 M2 comparison required")
        if sha(run / "checkpoint/weights.pt") != summary["checkpoint_sha256"] or manifest["weights_sha256"] != summary["checkpoint_sha256"]:
            raise ValueError("actual checkpoint differs")
        archive = verify_source(run / "code", summary["catalogue_sha256"])
        records = rows(run / "training.jsonl", 128)
        if [record["step"] for record in records] != list(range(1, 129)) or any(record.get("curriculum") != policy for record in records):
            raise ValueError("global order/policy identity differs")
        curve = rows(run / "curve.jsonl", 128)
        if [row["sample_id"] for row in curve] != [row["sample_id"] for row in records]:
            raise ValueError("actual optimizer sequence and data ledger differ")
        runs.append(run)
        summaries.append(summary)
        archives.append(archive)
        training.append(records)
        inputs[name] = {file: sha(run / file) for file in
                        ("summary.json", "training.jsonl", "training-probe.jsonl", "development-probe.jsonl", "curve.jsonl",
                         "checkpoint/manifest.json", "checkpoint/weights.pt", "code/source.json")}
        inputs[name]["native-job.json"] = sha(ROOT / "runs" / (name + "-job.json"))
        inputs[name]["curated-scope.json"] = sha(ROOT / "docs/evidence" / name / "scope.json")
    for field in ("config", "initial_training_probe", "initial_development"):
        if summaries[0][field] != summaries[1][field]:
            raise ValueError("matched initial prediction/loss/config field differs: " + field)
    recipes = [{key: value for key, value in summary["recovery_contract"].items() if key not in ("catalogue_sha256", "curriculum")}
               for summary in summaries]
    if recipes[0] != recipes[1] or recipes[0]["seed"] != 20261026:
        raise ValueError("training recipe differs beyond declared source/policy")
    for file, count in (("training-probe.jsonl", 32), ("development-probe.jsonl", 9)):
        compared = []
        for run, policy in zip(runs, policies):
            records = rows(run / file, count)
            if file == "training-probe.jsonl" and any(record.get("curriculum") != policy for record in records):
                raise ValueError("fixed probe policy missing")
            compared.append([{key: value for key, value in record.items() if key != "curriculum"} for record in records])
        if compared[0] != compared[1]:
            raise ValueError("fixed probe recipe/pixel/state population differs")
    for start in (0, 32, 64, 96):
        populations = [Counter(json.dumps({key: value for key, value in record.items() if key not in ("step", "curriculum")}, sort_keys=True)
                               for record in records[start:start + 32]) for records in training]
        if populations[0] != populations[1] or len(populations[0]) != 32 or any(count != 1 for count in populations[0].values()):
            raise ValueError("same32 image/public-state pairs once per cycle required")
    if [row["sample_id"] for row in training[0]] == [row["sample_id"] for row in training[1]]:
        raise ValueError("candidate did not actually change training order")
    packages = [{path: value for path, value in archive["files"].items() if path.startswith("tetris_vision/")} for archive in archives]
    if packages[0] != packages[1]:
        raise ValueError("neural package/loss/target source differs")
    warm = read(ROOT / "docs/evidence/interleave-profile-numerics-r1/scope.json")
    if (warm["status"] != "passed" or not warm["identical_actual_weights"] or
            warm["source_catalogues"] != [summary["catalogue_sha256"] for summary in summaries]):
        raise ValueError("same-source warm tensor/numerical compatibility missing")
    output = ROOT / "docs/evidence/interleave128-matched-start-r1"
    output.mkdir(exist_ok=False)
    shutil.copyfile(__file__, output / "verify.py")
    proof = {"status": "passed", "runs": names, "policies": policies, "steps": 128, "fixed_target": 1024,
             "initial_prediction_loss_metrics_exact": True, "common_training_recipe": recipes[0],
             "fixed_probe_recipes_exact_except_policy": True, "each_cycle_has_identical32_pixel_state_pairs": True,
             "actual_training_order_differs": True, "neural_package_sources_identical": True,
             "same_input_warm_actual_tensors_exact": True, "input_sha256": inputs,
             "training_fit_at128": {name: summary["final_training_probe"]["fit"] for name, summary in zip(names, summaries)},
             "script_sha256": sha(Path(__file__)),
             "scope": "Matched starts,recipes,neural package and data populations with actual changed order.128/1024 is partial;not final causal,independent-generalization or real-accuracy acceptance."}
    (output / "scope.json").write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "steps": 128, "fixed_target": 1024}))


if __name__ == "__main__":
    main()
