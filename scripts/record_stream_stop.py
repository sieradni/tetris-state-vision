"""Curate an ancestor Job's real commit-pressure stop and banked recovery cursor.

The child receipt may still say running after the ancestor kills its whole tree.
Preserve it verbatim alongside the terminating ancestor; never repair history.
This verifies hashes/prefixes, not semantic optimizer restoration or completion.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

from probe_model import require_bounded_job
from probe_stream_recovery import verify_dead
from source_snapshot import verify_source

ROOT = Path(__file__).resolve().parents[1]


def read(path, budget=64*1024):
    if path.stat().st_size > budget:
        raise ValueError("stop evidence JSON budget exceeded")
    return json.loads(path.read_text())


def digest(path, budget):
    count, sha = 0, hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            count += len(block)
            if count > budget:
                raise ValueError("stop evidence artifact budget exceeded")
            sha.update(block)
    return sha.hexdigest()


def main():
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"m2-stream-[a-z0-9-]{1,40}", args.name):
        raise ValueError("owned bounded stream name required")
    run = ROOT/"runs"/args.name
    ancestor_path = ROOT/"runs"/(args.name+"-pipeline-job.json")
    child_path = ROOT/"runs"/(args.name+"-job.json")
    ancestor, child, progress = read(ancestor_path), read(child_path), read(run/"progress.json")
    if (ancestor["status"] != "memory_pressure" or ancestor["exit_code"] != 125 or
            ancestor["system_pressure"]["reason"] != "os_commit_pressure" or
            ancestor["last_system_memory"]["commit_bytes"] < ancestor["system_pressure"]["stop_commit_bytes"]):
        raise ValueError("named actual ancestor commit-pressure stop required")
    pids = sorted({ancestor["supervisor_pid"], ancestor["child_pid"], child["supervisor_pid"], child["child_pid"], progress["pid"]})
    for pid in pids:
        verify_dead(pid)
    if (run/"summary.json").exists() or (run/"checkpoint/weights.pt").exists():
        raise ValueError("this stop curator requires no final inference artifact")
    pointer = read(run/"latest-training.json")
    if not re.fullmatch(r"step-[0-9]{6}-[0-9a-f]{32}", pointer["generation"]):
        raise ValueError("banked generation invalid")
    folder = run/"training-snapshots"/pointer["generation"]
    if (not folder.resolve().is_relative_to(run) or
            digest(folder/"manifest.json", 64*1024) != pointer["manifest_sha256"]):
        raise ValueError("banked manifest identity differs")
    snapshot = read(folder/"manifest.json")
    archive = verify_source(run/"code", snapshot["contract"]["catalogue_sha256"])
    if digest(folder/"tensors.pt", 256*1024**2) != snapshot["tensors_sha256"]:
        raise ValueError("banked model/optimizer artifact differs")
    if set(snapshot["ledgers"]) != {"curve.jsonl", "training.jsonl", "evaluations.jsonl"}:
        raise ValueError("banked ledger set differs")
    committed = {}
    for name, cursor in snapshot["ledgers"].items():
        size = cursor["bytes"]
        if type(size) is not int or not 0 <= size <= 16*1024**2:
            raise ValueError("banked cursor budget invalid")
        with (run/name).open("rb") as stream:
            prefix = stream.read(size)
        if len(prefix) != size or hashlib.sha256(prefix).hexdigest() != cursor["sha256"]:
            raise ValueError("banked prefix changed/truncated")
        committed[name] = len(prefix.splitlines())
    if not 1 <= snapshot["step"] <= progress["step"] <= snapshot["contract"]["steps"] <= 4096:
        raise ValueError("banked/logged/target cursor differs")
    if any(committed[name] != snapshot["step"] for name in ("curve.jsonl", "training.jsonl")):
        raise ValueError("banked optimizer/data rows differ from cursor")
    evidence = ROOT/"docs/evidence"/(args.name+"-stop")
    copies = [(ancestor_path, "ancestor-job.json"), (child_path, "child-job.json"),
              (run/"progress.json", "training-progress.json"), (run/"latest-training.json", "latest-training.json"),
              (run/"code/source.json", "source-manifest.json"), (folder/"manifest.json", "snapshot-manifest.json")]
    for suffix in (".log", ".stderr.log"):
        copies += [(ROOT/"runs"/(args.name+"-pipeline"+suffix), "ancestor"+suffix),
                   (ROOT/"runs"/(args.name+suffix), "child"+suffix)]
    copies += [(run/name, name) for name in ("curve.jsonl", "training.jsonl", "evaluations.jsonl", "memory.jsonl")]
    hashes = {name: digest(path, 16*1024**2) for path, name in copies}
    evidence.mkdir(exist_ok=False)
    for path, name in copies:
        shutil.copyfile(path, evidence/name)
        if digest(evidence/name, 16*1024**2) != hashes[name]:
            raise ValueError("curated stop bytes differ")
    result = {"status": "pressure_stop_hashes_verified", "run": args.name, "terminating_ancestor_status": ancestor["status"],
              "terminating_ancestor_exit_code": 125, "child_receipt_status": child["status"],
              "child_receipt_terminal": child["exit_code"] is not None, "verified_terminal_saved_pids": pids,
              "logged_step": progress["step"], "banked_step": snapshot["step"], "target_steps": snapshot["contract"]["steps"],
              "uncheckpointed_updates_not_restored": progress["step"]-snapshot["step"],
              "committed_ledger_rows": committed, "source_catalogue_sha256": archive["catalogue_sha256"],
              "snapshot_tensors_sha256": snapshot["tensors_sha256"], "artifact_sha256": hashes,
              "semantic_restore_verified": False, "target_complete": False,
              "curator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "Real ancestor OS-pressure stop; saved PIDs dead, frozen sources and banked tensor/ledger hashes verified. Raw stale child receipt retained. No semantic restoration, final inference artifact, completed target or accuracy claim."}
    (evidence/"scope.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "artifact_sha256"}), flush=True)


if __name__ == "__main__":
    main()
