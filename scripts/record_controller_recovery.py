"""Verify the named finite controller experiment: continuous3 == partial1+resume2.

Metadata/byte comparison only, under128MiB/30s/work7. Each run already replayed
its own training/probe recipes through record_stream_run. No Torch import.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from probe_model import require_bounded_job
from probe_stream_recovery import verify_dead
from record_stream_run import checkpoint_architecture, validate_completion
from source_snapshot import verify_source

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "m3-stream-controller-recovery-"
CONTROLLER = "11a0a81aeccddbb62451c46d24d2d4a844d762653b44cdb7b5edbb168bec0ac4"
CATALOGUE = "b6bb05f3a82b68ae41f5fb6ba8a256a89ddb69ed94c917807511bb636cf34255"


def digest(path, limit=32*1024**2):
    if path.stat().st_size > limit:
        raise ValueError("finite evidence byte budget exceeded")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    if path.stat().st_size > 1024**2:
        raise ValueError("finite JSON byte budget exceeded")
    return json.loads(path.read_text())


def rows(path):
    if path.stat().st_size > 1024**2:
        raise ValueError("finite ledger byte budget exceeded")
    return [json.loads(line) for line in path.read_text().splitlines()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    require_bounded_job()
    require(digest(ROOT/"scripts/run_stream_segment.py") == CONTROLLER, "controller source differs")
    records, selected = {}, []
    for case, steps, updates in (("p1", 1, 1), ("p2", 3, 2), ("full", 3, 3)):
        name = PREFIX+case
        run, pipe = ROOT/"runs"/name, ROOT/"runs"/(name+"-pipeline")
        saved = ROOT/"docs/evidence"/name
        summary, result = read(run/"summary.json"), read(pipe/"result.json")
        child = read(ROOT/"runs"/(name+"-job.json"))
        parent = read(ROOT/"runs"/(name+"-pipeline-job.json"))
        curator, scope = read(pipe/"curation-job.json"), read(saved/"scope.json")
        archive = verify_source(run/"code", CATALOGUE)
        trainer_parent = ROOT/"runs"/(PREFIX+"p1" if case == "p2" else "m3-stream-pilot-profile-r1")
        require(Path(child["command"][1]).resolve() == trainer_parent/"code/scripts/train_m2_stream.py",
                "worker did not execute the declared archived trainer")
        supervisor_sha = archive["files"]["scripts/run_bounded.py"]
        require(digest(ROOT/"scripts/run_bounded.py") == supervisor_sha, "live supervisor differs")
        for receipt, memory, work in ((parent, 1280, updates), (child, 640, updates), (curator, 128, steps+27)):
            require(receipt["memory_bytes"] == memory*1024**2 and receipt["work_items"] == work and
                    receipt["peak_job_commit_bytes"] <= receipt["memory_bytes"] and
                    receipt["supervisor_script_sha256"] == supervisor_sha and
                    receipt["system_pressure_guard"] == "GetPerformanceInfo/0.2s", "native envelope differs")
            for key in ("supervisor_pid", "child_pid"):
                verify_dead(receipt[key])
        require(parent["status"] == curator["status"] == "passed" and
                parent["exit_code"] == curator["exit_code"] == 0 and parent["wall_seconds"] == 120,
                "controller/curator did not complete within declared deadline")
        validate_completion(summary, child, allow_partial=True)
        manifest = read(run/"checkpoint/manifest.json")
        architecture = checkpoint_architecture(summary, manifest)
        require(architecture == "TetrisStateNet-M3" and summary["model_kind"] == "pilot" and
                summary["component_capacity"] == 64 and summary["steps"] == steps and
                summary["requested_steps"] == 3 and summary["updates_this_run"] == updates,
                "actual model or cursor differs")
        expected_status = "step_partial" if case == "p1" else "complete"
        require(summary["status"] == result["training_status"] == expected_status and
                result["status"] == "segment_verified" and result["global_steps"] == steps and
                result["updates_this_segment"] == updates and result["target_steps"] == 3 and
                result["target_complete"] is (steps == 3) and result["verified_probe_images"] == 27 and
                result["controller_sha256"] == CONTROLLER and
                result["source_catalogue_sha256"] == scope["source_catalogue_sha256"] == CATALOGUE and
                result["core_architecture"] == scope["core_architecture"] == architecture and
                result["component_capacity"] == scope["component_capacity"] == 64,
                "controller/evidence identity differs")
        require(scope["status"] == result["evidence_status"] == ("verified_partial" if steps == 1 else "passed") and
                scope["replayed_training_samples"] == steps and scope["verified_saved_probe_images"] == 27,
                "saved recipe replay scope differs")
        require(digest(run/"checkpoint/weights.pt") == summary["checkpoint_sha256"] ==
                result["weights_sha256"] == scope["weights_sha256"] == manifest["weights_sha256"],
                "actual inference weights differ")
        pointer = read(run/"latest-training.json")
        folder = run/"training-snapshots"/pointer["generation"]
        require(folder.resolve().is_relative_to(run) and digest(folder/"manifest.json") == pointer["manifest_sha256"],
                "snapshot pointer differs")
        snapshot = read(folder/"manifest.json")
        require(snapshot["step"] == steps and snapshot["contract"] == summary["recovery_contract"] and
                digest(folder/"tensors.pt") == snapshot["tensors_sha256"], "actual training payload differs")
        require(set(snapshot["ledgers"]) == {"training.jsonl", "curve.jsonl", "evaluations.jsonl"}, "ledger set differs")
        for filename, cursor in snapshot["ledgers"].items():
            require((run/filename).stat().st_size == cursor["bytes"] and digest(run/filename) == cursor["sha256"],
                    "snapshot ledger cursor differs")
        originals = {path.name: path for path in run.iterdir() if path.is_file()}
        originals.update({"source-manifest.json": run/"code/source.json",
                          "checkpoint-manifest.json": run/"checkpoint/manifest.json",
                          "training-snapshot-manifest.json": folder/"manifest.json"})
        originals.update({name+suffix: ROOT/"runs"/(name+suffix) for suffix in ("-job.json", ".log", ".stderr.log")})
        for copy in saved.iterdir():
            if copy.name != "scope.json":
                require(copy.name in originals and digest(copy) == digest(originals[copy.name]), "curated bytes differ")
        for filename in ("result.json", "progress.json", "curation-job.json", "curation.log", "curation.stderr.log"):
            selected.append((pipe/filename, case+"/"+filename))
        selected += [(ROOT/"runs"/(name+"-pipeline"+suffix), case+"/pipeline"+suffix)
                     for suffix in ("-job.json", ".log", ".stderr.log")]
        records[case] = (summary, snapshot, run, parent, child)
    full, resumed, partial = (records[case] for case in ("full", "p2", "p1"))
    for field in ("config", "catalogue_sha256", "recovery_contract", "initial_training_probe",
                  "initial_development", "final_training_probe", "development", "checkpoint_sha256"):
        require(full[0][field] == resumed[0][field], "resumed summary differs: "+field)
    require(all(record[1]["contract"] == full[1]["contract"] for record in records.values()), "original recipe differs")
    require(full[1]["tensors_sha256"] == resumed[1]["tensors_sha256"] and
            full[1]["module_training"] == resumed[1]["module_training"] and
            full[1]["signature"] == resumed[1]["signature"], "model/Adam/RNG/modes differ")
    require(resumed[0]["resume"]["step"] == 1, "resume did not restore step1")
    for filename in ("training.jsonl", "evaluations.jsonl"):
        require((full[2]/filename).read_bytes() == (resumed[2]/filename).read_bytes(), "resumed ledger differs")
    require(rows(partial[2]/"training.jsonl") == rows(full[2]/"training.jsonl")[:1], "partial data prefix differs")
    def curve(run):
        return [{k: v for k, v in row.items() if k != "elapsed_seconds"} for row in rows(run/"curve.jsonl")]
    require(curve(full[2]) == curve(resumed[2]) and curve(partial[2]) == curve(full[2])[:1],
            "loss/terms/gradient trajectory differs")
    require(len(rows(full[2]/"training.jsonl")) == len(curve(full[2])) == 3, "trajectory length differs")
    evidence = ROOT/"docs/evidence/m3-controller-recovery-r1"
    evidence.mkdir(exist_ok=False)
    selected.append((Path(__file__), "record_controller_recovery.py"))
    hashes = {}
    for source, relative in selected:
        target = evidence/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[relative] = digest(source)
        require(digest(target) == hashes[relative], "proof copy differs")
    proof = {"status": "passed", "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
             "controller_sha256": CONTROLLER, "source_catalogue_sha256": CATALOGUE,
             "architecture": architecture, "component_capacity": 64, "model_kind": "pilot",
             "continuous_steps": 3, "partial_then_resumed_updates": [1, 2],
             "exact_data_loss_terms_gradients_evaluations": True,
             "byte_identical_model_optimizer_rng_payload": True, "exact_module_modes_signature": True,
             "replayed_training_rows_across_cases": 7, "verified_saved_probe_images_across_cases": 81,
             "weights_sha256": full[0]["checkpoint_sha256"], "training_payload_sha256": full[1]["tensors_sha256"],
             "cases": {case: {"parent_seconds": r[3]["elapsed_seconds"], "parent_peak_bytes": r[3]["peak_job_commit_bytes"],
                              "child_peak_bytes": r[4]["peak_job_commit_bytes"]} for case, r in records.items()},
             "artifact_sha256": hashes,
             "scope": "Native finite controller archive/routing/recovery/curation identity only; no convergence, long-shape memory or real accuracy claim."}
    (evidence/"scope.json").write_text(json.dumps(proof, indent=2)+"\n")
    print(json.dumps(proof), flush=True)


if __name__ == "__main__":
    main()
