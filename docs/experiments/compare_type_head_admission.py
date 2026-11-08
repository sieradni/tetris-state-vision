"""Compare actual first updates of two separately admitted optimizer recipes."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from probe_model import require_bounded_job
from source_snapshot import verify_source


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path):
    return json.loads(path.read_text())


def main():
    ceiling = require_bounded_job()
    import torch
    torch.set_num_threads(1)
    runs = [ROOT / "runs/m2-stream-recovery-gate-r8", ROOT / "runs/m2-stream-recovery-gate-r7"]
    summaries, states, manifests, sources = [], [], [], {}
    for factor, run in zip((1, 4), runs):
        proof = load_json(run / "proof.json")
        receipt = load_json(run.parent / (run.name + "-job.json"))
        summary = load_json(run / "continuous/summary.json")
        if (proof["status"] != "passed" or receipt["status"] != "passed" or receipt["exit_code"] != 0 or
                proof["type_head_rate_factor"] != factor or summary["recovery_contract"]["type_head_rate_factor"] != factor):
            raise ValueError("separate native admissions required")
        archive = verify_source(run / "continuous/code", proof["source_catalogue_sha256"])
        source = run / "continuous/training-snapshots"
        matches = list(source.glob("step-000003-*/manifest.json"))
        # The continuous worker snapshots at step3; profile snapshots step1.
        first = list((run / "profile/training-snapshots").glob("step-000001-*/manifest.json"))
        if len(first) != 1 or len(matches) != 1:
            raise ValueError("unique fixed-step snapshots required")
        manifest = load_json(first[0])
        tensor_path = first[0].parent / "tensors.pt"
        if manifest["step"] != 1 or digest(tensor_path) != manifest["tensors_sha256"]:
            raise ValueError("snapshot identity differs")
        state = torch.load(tensor_path, map_location="cpu", weights_only=True)
        summaries.append(summary)
        manifests.append(manifest)
        states.append(state)
        for path in (run / "proof.json", run.parent / (run.name + "-job.json"), first[0], tensor_path,
                     run / "continuous/summary.json", run / "continuous/curve.jsonl", run / "continuous/training.jsonl"):
            sources[path.relative_to(ROOT).as_posix()] = digest(path)
        if archive["catalogue_sha256"] != proof["source_catalogue_sha256"]:
            raise ValueError("source identity differs")
    a, b = summaries
    if (a["catalogue_sha256"] != b["catalogue_sha256"] or a["config"] != b["config"] or
            a["initial_training_probe"] != b["initial_training_probe"] or a["initial_development"] != b["initial_development"] or
            (runs[0]/"continuous/training.jsonl").read_bytes() != (runs[1]/"continuous/training.jsonl").read_bytes()):
        raise ValueError("matched source, initial observations or data differ")
    for run in runs:
        row = json.loads((run / "continuous/curve.jsonl").read_text().splitlines()[0])
        row.pop("elapsed_seconds")
        if run == runs[0]:
            first_record = row
        elif row != first_record:
            raise ValueError("first loss or gradient norm differs")
    head = {"layer_type.weight", "layer_type.bias"}
    astate, bstate = states
    if astate["model"].keys() != bstate["model"].keys() or any(
            not torch.equal(astate["model"][name], bstate["model"][name]) for name in astate["model"] if name not in head):
        raise ValueError("first-step non-head weights differ")
    if not all(not torch.equal(astate["model"][name], bstate["model"][name]) for name in head):
        raise ValueError("candidate head did not receive different update")
    groups = [manifest["signature"]["parameter_groups"] for manifest in manifests]
    if (len(groups[0]) != 1 or len(groups[1]) != 2 or set(groups[1][1]) != head or
            set(groups[0][0]) != set(groups[1][0]) | head or set(groups[1][0]) & head):
        raise ValueError("optimizer parameter ownership differs")
    settings = [[{key: value for key, value in group.items() if key != "params"}
                 for group in state["optimizer"]["param_groups"]] for state in states]
    if [group["lr"] for group in settings[0]] != [.001] or [group["lr"] for group in settings[1]] != [.001, .004]:
        raise ValueError("actual snapshot learning rates differ")
    if any({k: v for k, v in group.items() if k != "lr"} !=
           {k: v for k, v in settings[0][0].items() if k != "lr"} for group in settings[1]):
        raise ValueError("other optimizer settings differ")
    result = {"status": "passed", "source_catalogue_sha256": a["catalogue_sha256"], "sources": sources,
              "job_ceiling_bytes": ceiling, "initial_probe_metrics_identical": True,
              "first_update_loss_terms_gradient_norm_identical": True, "first_update_non_head_tensors_identical": True,
              "first_update_head_tensors_differ": True, "parameter_groups": groups, "optimizer_settings": settings,
              "scope": "Initial metrics and first update parity only; exact initial weight bytes were not saved; no model quality claim."}
    out = ROOT / "runs/type-head-matched-admission-r1.json"
    with out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"status": "passed", "scope": result["scope"]}))


if __name__ == "__main__":
    main()
