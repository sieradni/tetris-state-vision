"""Data-matched128-update diagnostics; not an adequately converged model ranking."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from probe_model import require_bounded_job
from source_snapshot import verify_source

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("images", "ancestry_groups", "exact_locked_boards", "cell_accuracy",
           "exact_known_layers", "known_layers", "mean_geometry_error_cells", "loss")


def read(path):
    if path.stat().st_size > 1024**2:
        raise ValueError("learning analysis JSON budget exceeded")
    return json.loads(path.read_text())


def rows(path):
    if path.stat().st_size > 16*1024**2:
        raise ValueError("learning ledger byte budget exceeded")
    values = [json.loads(line) for line in path.read_text().splitlines()]
    if len(values) > 4096:
        raise ValueError("learning record count exceeded")
    return values


def main():
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--reference", default="m2-stream-learn768-r3")
    args = parser.parse_args()
    for name in (args.run, args.reference):
        if not re.fullmatch(r"m2-stream-[a-z0-9-]{1,40}", name):
            raise ValueError("owned stream artifact required")
    run, reference = (ROOT/"runs"/name for name in (args.run, args.reference))
    summary, baseline = read(run/"summary.json"), read(reference/"summary.json")
    job = read(ROOT/"runs"/(args.run+"-job.json"))
    if summary["status"] != "complete" or summary["steps"] != 128 or job["status"] != "passed" or summary["model_kind"] != "accuracy":
        raise ValueError("native completed128 larger configuration required")
    verify_source(run/"code", summary["catalogue_sha256"])
    verify_source(reference/"code", summary["catalogue_sha256"])
    data, original = rows(run/"training.jsonl"), rows(reference/"training.jsonl")
    if [row["step"] for row in data] != list(range(1, 129)) or data != original[:128]:
        raise ValueError("128-step sample recipes/pixels/labels are not identical")
    matched = [row for row in rows(reference/"evaluations.jsonl") if row["step"] == 128]
    if len(matched) != 1:
        raise ValueError("reference needs its recorded128-update development probe")
    manifest = read(run/"checkpoint/manifest.json")
    if manifest["weights_sha256"] != summary["checkpoint_sha256"]:
        raise ValueError("larger inference manifest identity differs")
    populations = {}
    for split, field in (("training", "final_training_probe"), ("development", "development")):
        empty = nonempty = empty_right = nonempty_right = 0
        for case in summary[field]["cases"]:
            path = run/(split+"-probe")/(case["sample_id"]+".json")
            annotation = read(path)
            cells = [cell for row in annotation["board"]["rows"] for cell in row]
            if len(cells) != 200 or any(cell["occupied"] is None for cell in cells):
                raise ValueError("population requires200 supported locked-cell labels")
            is_empty = not any(cell["occupied"] for cell in cells)
            empty += int(is_empty)
            nonempty += int(not is_empty)
            empty_right += int(is_empty and case["exact_locked_board"])
            nonempty_right += int(not is_empty and case["exact_locked_board"])
        populations[split] = {"empty_locked_boards": empty, "exact_empty_locked_boards": empty_right,
                              "nonempty_locked_boards": nonempty, "exact_nonempty_locked_boards": nonempty_right}
    resume = summary["resume"]
    if resume:
        parent = Path(resume["source_run"])
        parent_summary = read(parent/"summary.json")
        parent_job = read(parent.parent/(parent.name+"-job.json"))
        carried = rows(parent/"curve.jsonl")[:resume["step"]]
        if carried != rows(run/"curve.jsonl")[:resume["step"]] or resume["step"]+summary["updates_this_run"] != 128:
            raise ValueError("carried optimizer records or new-update count differ")
        segments = [{"name": parent.name, "status": parent_summary["status"], "global_steps": parent_summary["steps"],
                     "updates_this_run": parent_summary["updates_this_run"], "native_elapsed_seconds": parent_job["elapsed_seconds"]},
                    {"name": args.run, "status": summary["status"], "global_steps": 128,
                     "updates_this_run": summary["updates_this_run"], "native_elapsed_seconds": job["elapsed_seconds"]}]
    else:
        segments = [{"name": args.run, "status": summary["status"], "global_steps": 128,
                     "updates_this_run": 128, "native_elapsed_seconds": job["elapsed_seconds"]}]
    result = {"status": "passed", "source_catalogue_sha256": summary["catalogue_sha256"],
              "weights_sha256": summary["checkpoint_sha256"], "parameters": manifest["parameters"], "steps": 128,
              "exact_data_prefix_records": 128, "matched_update_count": True, "segments": segments,
              "larger_training": {key: summary["final_training_probe"][key] for key in METRICS},
              "larger_development": {key: summary["development"][key] for key in METRICS},
              "pilot_development_at128": {key: matched[0]["development"][key] for key in METRICS},
              "locked_board_populations": populations, "initializations_differ_by_architecture": True,
              "adequately_converged_model_ranking_allowed": False, "complete_state_accuracy_established": False,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "Native larger128 diagnostic, exact data exposure and carried93+new35 records. Equal step count/one seed does not tune optimization, establish convergence, select architecture, or prove real/complete-state accuracy."}
    with (run/"analysis.json").open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
