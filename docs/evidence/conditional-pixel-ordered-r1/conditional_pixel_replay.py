"""Replay fixed pixels with a copied type head; preserve original ordinary run."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from probe_model import require_bounded_job
from pixel_replay_contract import verify_non_type_outputs
import probe_stream_errors as ordinary
from type_head_fit_contract import digest, load_bank, metrics, read


def main():
    ceiling = require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("ordered", "interleaved"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.parent != ROOT / "runs" or output.exists():
        raise ValueError("new owned output required")
    bank, model_path, identities = load_bank(args.study)
    head_dir = ROOT / "runs" / ("type-head-" + args.study + "-fit-r1")
    scope = read(ROOT / "docs/evidence" / head_dir.name / "scope.json")
    head_result = read(head_dir / "result.json")
    payload_path = head_dir / "head-state.pt"
    if (scope["status"] != "passed" or scope["inputs_sha256"] != identities or
            digest(head_dir / "result.json") != scope["result_sha256"] or
            digest(payload_path) != scope["artifact_sha256"]["head-state.pt"] or
            head_result["payload_sha256"] != digest(payload_path) or
            head_result["inputs_sha256"] != identities):
        raise ValueError("curated fitted-head identity differs")
    tracked = {str(path): digest(path) for path in (Path(__file__), ROOT / "scripts/pixel_replay_contract.py",
               ROOT / "scripts/probe_stream_errors.py", head_dir / "result.json", payload_path)}
    os.environ.update({"ONEDNN_PRIMITIVE_CACHE_CAPACITY": "16", "DNNL_PRIMITIVE_CACHE_CAPACITY": "16"})
    import torch
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    if payload["format"] != head_result["recipe"]["format"] or payload["step"] != 1024:
        raise ValueError("unexpected fitted payload")
    raw_weight = (payload["weight"] / payload["scale"]).float()
    raw_bias = (payload["bias"] - ((payload["weight"] / payload["scale"]) * payload["mean"]).sum(dim=1)).float()
    if (not torch.equal(raw_weight, torch.tensor(head_result["head"]["weight_raw32"])) or
            not torch.equal(raw_bias, torch.tensor(head_result["head"]["bias_raw32"]))):
        raise ValueError("payload conversion differs from audited raw head")
    rows = bank["features"]
    pending, changed_inputs, records, changed_logits = [], [], [], []
    current = {"inside": False, "changed": None}

    def hook(module, inputs, result):
        if isinstance(module, torch.nn.Linear) and tuple(result.shape) == (1, 2, 8):
            destination = changed_inputs if current["inside"] else pending
            destination.append((module, inputs[0].detach().clone(), result.detach().clone()))
        if type(module).__name__ != "TetrisStateNet" or current["inside"]:
            return None
        if len(records) >= 41 or len(pending) != len(result["stages"]) or current["changed"] is not None:
            raise ValueError("unexpected model forward order")
        index = len(records)
        if any(event[0] is not module.layer_type or not torch.equal(event[2], stage["layer_type"])
               for event, stage in zip(pending, result["stages"])):
            raise ValueError("actual type-head observation differs")
        if (pending[-1][1][0].tolist() != [row["features"] for row in rows[2*index:2*index+2]] or
                pending[-1][2][0].tolist() != [row["logits"] for row in rows[2*index:2*index+2]]):
            raise ValueError("original saved feature/logit replay differs")
        original_w, original_b = module.layer_type.weight.detach().clone(), module.layer_type.bias.detach().clone()
        try:
            module.layer_type.weight.copy_(raw_weight)
            module.layer_type.bias.copy_(raw_bias)
            current["inside"] = True
            changed = module(*inputs)
        finally:
            current["inside"] = False
            module.layer_type.weight.copy_(original_w)
            module.layer_type.bias.copy_(original_b)
        if (not torch.equal(original_w, module.layer_type.weight) or not torch.equal(original_b, module.layer_type.bias) or
                len(changed_inputs) != len(pending)):
            raise ValueError("head restore or stage count differs")
        for old, new in zip(pending, changed_inputs):
            if old[0] is not new[0] or not torch.equal(old[1], new[1]):
                raise ValueError("type-head inputs changed")
        verify_non_type_outputs(result, changed, torch)
        current["changed"] = changed["final"]
        changed_logits.extend(changed_inputs[-1][2][0].tolist())
        pending.clear()
        changed_inputs.clear()
        return None

    original_pack = ordinary.pack

    def capture_pack(final, target):
        if current["changed"] is None:
            raise ValueError("missing paired intervention")
        original = original_pack(final, target)
        changed = original_pack(current["changed"], target)
        if original[1] != changed[1]:
            raise ValueError("target changed")
        records.append({"original_prediction": original[0], "changed_prediction": changed[0], "target": original[1]})
        current["changed"] = None
        return original

    handle = torch.nn.modules.module.register_module_forward_hook(hook)
    previous = sys.argv
    ordinary.pack = capture_pack
    try:
        sys.argv = [str(ROOT / "scripts/probe_stream_errors.py"), "--run", model_path.name, "--output", str(output)]
        ordinary.main()
    finally:
        sys.argv = previous
        ordinary.pack = original_pack
        handle.remove()
    if len(records) != 41 or pending or changed_inputs or current["changed"] is not None:
        raise ValueError("finite paired population incomplete")
    observed = read(output / "result.json")
    reference = read(ROOT / "docs/evidence" / ("m2-stream-" + ("allclass1024" if args.study == "ordered" else "interleave1024") + "-features-r1") / "observed-ordinary.json")
    if observed["cases"] != reference["cases"] or observed["totals"] != reference["totals"]:
        raise ValueError("ordinary independent report differs")
    if any(digest(Path(path)) != value for path, value in tracked.items()):
        raise ValueError("input/source changed during replay")
    _, _, after = load_bank(args.study)
    if identities != after:
        raise ValueError("curated inputs changed during replay")
    vector = head_result["logits"]["raw32"]
    discrepancy = max(abs(a-b) for actual, expected in zip(changed_logits, vector) for a, b in zip(actual, expected))
    actual_metrics = metrics(rows, changed_logits)
    if any(actual_metrics[split]["correct"] != head_result["metrics"]["raw32"][split]["correct"]
           for split in ("training", "development")):
        raise ValueError("pixel and vector class counts differ")
    proof = {"status": "passed", "study": args.study, "model_run": model_path.name, "inputs_sha256": identities,
             "tracked_sha256": tracked, "job_ceiling_bytes": ceiling, "forwards": 82,
             "ordinary_report_sha256": digest(output / "result.json"), "head_payload_sha256": digest(payload_path),
             "pixel_vector_max_abs_difference": discrepancy, "type_metrics": actual_metrics,
             "changed_logits": changed_logits, "records": records}
    (output / "intervention.json").write_text(json.dumps(proof, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "passed", "forwards": 82, "pixel_vector_max_abs_difference": discrepancy}))


if __name__ == "__main__":
    main()
