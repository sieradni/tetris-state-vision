"""Compare actual first updates of two separately admitted optimizer recipes."""
import hashlib
import argparse
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(not name.startswith("m2-stream-recovery-gate-r") or not name.removeprefix("m2-stream-recovery-gate-r").isdigit()
           for name in (args.control, args.candidate)):
        raise ValueError("owned named recovery gates required")
    output = args.output.resolve()
    if output.parent != ROOT / "runs" or output.exists():
        raise ValueError("new owned output required")
    import torch
    torch.set_num_threads(1)
    runs = [ROOT / "runs" / args.control, ROOT / "runs" / args.candidate]
    summaries, states, manifests, initial_states, sources = [], [], [], [], {}
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
        initial_path = run / "continuous/initial-weights.pt"
        if digest(initial_path) != summary["initial_weights_sha256"]:
            raise ValueError("initial tensor file identity differs")
        initial_states.append(torch.load(initial_path, map_location="cpu", weights_only=True))
        summaries.append(summary)
        manifests.append(manifest)
        states.append(state)
        for path in (run / "proof.json", run.parent / (run.name + "-job.json"), first[0], tensor_path,
                     initial_path, run / "continuous/summary.json", run / "continuous/curve.jsonl", run / "continuous/training.jsonl"):
            sources[path.relative_to(ROOT).as_posix()] = digest(path)
        if archive["catalogue_sha256"] != proof["source_catalogue_sha256"]:
            raise ValueError("source identity differs")
    a, b = summaries
    if initial_states[0].keys() != initial_states[1].keys() or any(
            initial_states[0][name].dtype != value.dtype or initial_states[0][name].shape != value.shape or
            not torch.equal(initial_states[0][name].contiguous().reshape(-1).view(torch.uint8),
                            value.contiguous().reshape(-1).view(torch.uint8))
            for name, value in initial_states[1].items()):
        raise ValueError("actual initial tensor bytes differ")
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
              "job_ceiling_bytes": ceiling, "initial_tensor_bytes_identical": True, "initial_probe_metrics_identical": True,
              "first_update_loss_terms_gradient_norm_identical": True, "first_update_non_head_tensors_identical": True,
              "first_update_head_tensors_differ": True, "parameter_groups": groups, "optimizer_settings": settings,
              "scope": "Actual initial tensors, initial metrics and first update parity only; no model quality claim."}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"status": "passed", "scope": result["scope"]}))


if __name__ == "__main__":
    main()
