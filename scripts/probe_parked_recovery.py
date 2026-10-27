"""Verify the saved state of a pressure-parked run without another forward/update."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

from probe_model import require_bounded_job
from source_snapshot import verify_source

ROOT = Path(__file__).resolve().parents[1]


def main():
    ceiling = require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if (run/"latest-training.json").stat().st_size > 64*1024:
        raise ValueError("recovery pointer byte budget exceeded")
    pointer = json.loads((run/"latest-training.json").read_text())
    if not re.fullmatch(r"step-[0-9]{6}-[0-9a-f]{32}", pointer["generation"]):
        raise ValueError("recovery generation invalid")
    manifest_path = run/"training-snapshots"/pointer["generation"]/"manifest.json"
    if (not manifest_path.resolve().is_relative_to(run) or manifest_path.stat().st_size > 64*1024 or
            hashlib.sha256(manifest_path.read_bytes()).hexdigest() != pointer["manifest_sha256"]):
        raise ValueError("recovery manifest path/size/identity differs")
    manifest = json.loads(manifest_path.read_text())
    archive = verify_source(run/"code", manifest["contract"]["catalogue_sha256"])
    job = json.loads((run.parent/(run.name+"-job.json")).read_text())
    if job["status"] != "memory_pressure" or job["exit_code"] != 125:
        raise ValueError("named real pressure-parked run required")
    sys.path[:0] = [str(run/"code/scripts"), str(run/"code")]
    sys.dont_write_bytecode = True
    import torch
    from tetris_vision.model import ModelConfig, TetrisStateNet
    from training_snapshot import load_training_snapshot
    if not Path(load_training_snapshot.__code__.co_filename).resolve().is_relative_to(run/"code"):
        raise ValueError("recovery probe import is not frozen")
    torch.set_num_threads(1)
    config = dict(manifest["contract"]["config"])
    config["widths"] = tuple(config["widths"])
    model = TetrisStateNet(ModelConfig(**config))
    optimizer = torch.optim.AdamW(model.parameters(), lr=manifest["contract"]["learning_rate"], weight_decay=.01)
    recovered = load_training_snapshot(run, model, optimizer, contract=manifest["contract"])
    if (recovered["step"] != manifest["step"] or len(optimizer.state) != sum(p.requires_grad for p in model.parameters()) or
            not all(float(state["step"]) == recovered["step"] for state in optimizer.state.values())):
        raise ValueError("parked optimizer step differs from committed cursor")
    for name in ("curve.jsonl", "training.jsonl"):
        rows = [json.loads(line) for line in recovered["ledger_prefixes"][name].decode().splitlines()]
        if [row["step"] for row in rows] != list(range(1, recovered["step"]+1)):
            raise ValueError("parked committed ledger prefix differs")
    result = {"status": "passed", "parked_run": str(run), "parked_exit_code": 125,
              "recovered_step": recovered["step"], "target_steps": manifest["contract"]["steps"],
              "optimizer_parameter_states": len(optimizer.state), "model_parameters": sum(p.numel() for p in model.parameters()),
              "source_catalogue_sha256": archive["catalogue_sha256"], "pointer": recovered["pointer"],
              "snapshot_tensors_sha256": manifest["tensors_sha256"], "job_ceiling_bytes": ceiling,
              "new_forwards": 0, "new_updates": 0, "final_run_summary_available": (run/"summary.json").exists(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "Pressure-parked model/Adam/RNG/ledger recovery only. No new optimizer update, parser accuracy, or completed target claim."}
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
