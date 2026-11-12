"""Verify/replay finite stream ledgers from their archived source and curate evidence."""
import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import re
import shutil
import subprocess
import sys

from probe_model import require_bounded_job
from source_snapshot import verify_source

ROOT = Path(__file__).resolve().parents[1]

# Owned, source-frozen finite runs. Acceleration arms keep their own family
# because they are named for the predeclared policy, not for a stream study.
OWNED_RUN_PATTERN = re.compile(r"(?:m[23]-stream-[a-z0-9-]{1,40}|accumulation-arm-[a-z0-9]+(?:-[a-z0-9]+)*"
                               r"|m3-identity-factor-[a-z0-9-]{1,40}|m3-occsched-[a-z0-9-]{1,40}"
                               r"|m3-fit-[a-z0-9-]{1,40}"
                               r"|m3-percell-[a-z0-9-]{1,40}|m3-sweep-[a-z0-9-]{1,40}"
                               r"|m3-occctx-[a-z0-9-]{1,40}|m3-occdetail-[a-z0-9-]{1,40}"
                               r"|m3-occdstencil-[a-z0-9-]{1,40}|m3-occres-[a-z0-9-]{1,40}"
                               r"|m3-occres2-[a-z0-9-]{1,40}|m3-occdeform-[a-z0-9-]{1,40}"
                               r"|m3-occstack-[a-z0-9-]{1,40}|m3-occpix-[a-z0-9-]{1,40}"
                               r"|m3-occscale-[a-z0-9-]{1,40}|m3-occgeom-[a-z0-9-]{1,40}"
                               r"|m3-occextent-[a-z0-9-]{1,40}"
                               r"|m3-occgeocap-[a-z0-9-]{1,40}|m3-occlatres-[a-z0-9-]{1,40}|m3-occlatfree-[a-z0-9-]{1,40}|m3-occbref-[a-z0-9-]{1,40}|m3-occfit-[a-z0-9-]{1,40}"
                               r"|m3-occorigin-[a-z0-9-]{1,40}|m3-trunkwidth-[a-z0-9-]{1,40})")


def require_owned_run(name):
    """Refuse anything outside the owned finite run families."""
    if type(name) is not str or not OWNED_RUN_PATTERN.fullmatch(name):
        raise ValueError("owned finite run name required")
    return name


def validate_completion(summary, job, *, allow_partial=False):
    partial = summary.get("status") in {"wall_partial", "step_partial"}
    steps, target = summary.get("steps"), summary.get("requested_steps")
    if (type(steps) is not int or type(target) is not int or not 1 <= steps <= target <= 4096 or
            not summary.get("checkpoint_round_trip_exact")):
        raise ValueError("finite saved checkpoint and exact reload required")
    if partial:
        if not allow_partial or steps >= target or job.get("status") != "failed" or job.get("exit_code") != 2:
            raise ValueError("explicit controlled partial with saved state required")
    elif summary.get("status") != "complete" or steps != target or job.get("status") != "passed" or job.get("exit_code") != 0:
        raise ValueError("complete native run required")
    updates = summary.get("updates_this_run", steps)
    if type(updates) is not int or not 0 < updates <= job.get("work_items", 0):
        raise ValueError("new-update count exceeds declared cap")
    return partial


@lru_cache(maxsize=1)
def working_tree_model():
    """The present repository's model module, loaded by path under a private package name.

    Architecture identity must not depend on `sys.path` order: replay instruments insert a run's frozen
    archive ahead of the working tree, and that archive may predate the derived-name helper. Loading the
    current package from `src/` under its own name makes the identity rule deterministic wherever this is
    called, so a consumer never has to sequence its calls around the archive (V-090).
    """
    import importlib.util
    package = ROOT/"src"/"tetris_vision"
    name = "tetris_vision_identity"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, package/"__init__.py",
                                                      submodule_search_locations=[str(package)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    from tetris_vision_identity.model import ModelConfig, architecture_name
    return ModelConfig, architecture_name


def checkpoint_architecture(summary, manifest):
    """Bind the report's model identity to the actual checkpoint configuration."""
    if type(summary["config"].get("layer_evidence", False)) is not bool:
        raise ValueError("checkpoint layer evidence identity must be boolean")
    capacity = summary["config"].get("component_capacity", 0)
    declared = summary.get("component_capacity", 0)
    if type(capacity) is not int or not 0 <= capacity <= 64 or type(declared) is not int or declared != capacity:
        raise ValueError("checkpoint component capacity identity differs")
    if "curriculum" in summary:
        curriculum = summary["curriculum"]
        if (type(curriculum) is not str or curriculum not in ("stream-v1", "all-class-fit-v1", "all-class-interleaved-v1") or
                curriculum in ("all-class-fit-v1", "all-class-interleaved-v1") and capacity or
                summary.get("recovery_contract", {}).get("curriculum") != curriculum or
                manifest.get("metadata", {}).get("recovery_contract", {}).get("curriculum") != curriculum):
            raise ValueError("checkpoint curriculum identity differs")
    expected = "TetrisStateNet-M3" if capacity else "TetrisStateNet-M2"
    # The declared trunk shape is part of the identity, derived from the verified config so a
    # widened configuration is never reported under the published name.
    ModelConfig, architecture_name = working_tree_model()
    settings = dict(summary["config"])
    derived = expected
    if "widths" in settings:
        # Only a config that records the trunk shape can name a trunk; legacy snippets that carry
        # neither shape field keep the published identity they were written under.
        settings["widths"] = tuple(settings["widths"])
        derived = architecture_name(ModelConfig(**settings))
    if not derived.startswith(expected) or manifest["architecture"] != derived or manifest["config"] != summary["config"]:
        raise ValueError("checkpoint architecture/configuration identity differs")
    return derived


def digest(path, maximum=64*1024**2):
    value, consumed = hashlib.sha256(), 0
    with path.open("rb") as data:
        for chunk in iter(lambda: data.read(65536), b""):
            consumed += len(chunk)
            if consumed > maximum:
                raise ValueError("evidence file byte budget exceeded")
            value.update(chunk)
    return value.hexdigest()


def main():
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--allow-partial", action="store_true", help="preserve controlled wall/step partial status; never a completion claim")
    args = parser.parse_args()
    require_owned_run(args.name)
    run = ROOT/"runs"/args.name
    summary = json.loads((run/"summary.json").read_text())
    job = json.loads((ROOT/"runs"/(args.name+"-job.json")).read_text())
    partial = validate_completion(summary, job, allow_partial=args.allow_partial)
    code = run/"code"
    archive = verify_source(code, summary["catalogue_sha256"])
    sys.path[:0] = [str(code/"scripts"), str(code)]
    sys.dont_write_bytecode = True
    curriculum = summary.get("curriculum", "stream-v1")
    if type(curriculum) is not str or curriculum not in ("stream-v1", "all-class-fit-v1", "all-class-interleaved-v1"):
        raise ValueError("unsupported recorded curriculum")
    new_curriculum_contract = "curriculum" in summary
    factor = summary.get("arguments", {}).get("type_head_rate_factor", 1)
    if (type(factor) is not int or factor not in (1, 4) or
            summary.get("recovery_contract", {}).get("type_head_rate_factor", 1) != factor or
            factor == 4 and (curriculum != "all-class-fit-v1" or summary["config"].get("component_capacity", 0) or
                             summary["config"].get("layer_evidence", False) or summary["arguments"]["learning_rate"] != .001)):
        raise ValueError("recorded type-head optimizer identity differs")
    if new_curriculum_contract:
        from m2_curriculum import probe_specs, sample_for, training_case
        from m2_fit_contract import summarize_fit
        if any(not Path(function.__code__.co_filename).resolve().is_relative_to(code)
               for function in (probe_specs, sample_for, training_case, summarize_fit)):
            raise ValueError("curriculum implementation is not frozen")
        if summary["recovery_contract"].get("curriculum") != curriculum:
            raise ValueError("curriculum recovery identity differs")
        if curriculum == "all-class-interleaved-v1":
            from m2_interleaved_order import distinct_groups, interleaved_spec
            if any(not Path(function.__code__.co_filename).resolve().is_relative_to(code)
                   for function in (distinct_groups, interleaved_spec)):
                raise ValueError("interleaved order implementation is not frozen")
    if summary.get("resource_profile_version", 0) >= 2 and summary["profile_square"]:
        from train_m2_stream import validate_admission
        validate_admission(summary, job, catalogue=archive["catalogue_sha256"],
                           model=summary["model_kind"], config=summary["config"],
                           ceiling=summary["job_ceiling_bytes"], development_groups=summary["development_groups"],
                           supervisor_sha=archive["files"]["scripts/run_bounded.py"],
                           **({"curriculum": curriculum} if new_curriculum_contract else {}),
                           **({"type_head_rate_factor": factor} if "type_head_rate_factor" in summary.get("arguments", {}) else {}))
    from PIL import Image
    from m2_stream_data import sample_case
    if not Path(sample_case.__code__.co_filename).resolve().is_relative_to(code):
        raise ValueError("sampler did not import from archived source")
    manifest = json.loads((run/"checkpoint/manifest.json").read_text())
    architecture = checkpoint_architecture(summary, manifest)
    if "initial_weights_sha256" in summary and (
            manifest["metadata"].get("initial_weights_sha256") != summary["initial_weights_sha256"] or
            digest(run/"initial-weights.pt", 256*1024**2) != summary["initial_weights_sha256"]):
        raise ValueError("actual initial tensor artifact differs")
    origin = summary.get("initial_weights_from")
    if origin is not None:
        # A staged run's start is another saved artifact, so its bytes must be
        # re-read from that run rather than trusted as a printed digest.
        source = Path(origin["run"]).resolve()
        if not source.is_relative_to(ROOT/"runs") or source == run.resolve():
            raise ValueError("initial-weight source escaped owned runs")
        source_summary = json.loads((source/"summary.json").read_text())
        source_manifest = json.loads((source/"checkpoint/manifest.json").read_text())
        if (source_manifest["weights_sha256"] != origin["weights_sha256"] or
                source_summary["checkpoint_sha256"] != origin["weights_sha256"] or
                digest(source/"checkpoint/weights.pt") != origin["weights_sha256"] or
                source_summary["catalogue_sha256"] != origin["catalogue_sha256"] or
                summary["initial_weights_sha256"] != origin["weights_sha256"] or
                source_summary.get("curriculum") != summary.get("curriculum") or
                summary.get("recovery_contract", {}).get("initial_weights_from") != origin["run"] or
                manifest["metadata"].get("initial_weights_from") != origin):
            raise ValueError("initial-weight origin differs from its saved artifact")
    if summary.get("freeze_backbone", False) is not summary.get("recovery_contract", {}).get("freeze_backbone", False):
        raise ValueError("staged freeze identity differs between summary and contract")
    if digest(run/"checkpoint/weights.pt") != summary["checkpoint_sha256"] or manifest["weights_sha256"] != summary["checkpoint_sha256"]:
        raise ValueError("weight checksum mismatch")
    recovery_manifest = None
    resume_parent = None
    resume_files = []
    if summary.get("resource_profile_version", 0) >= 3:
        pointer = json.loads((run/"latest-training.json").read_text())
        generation = pointer.get("generation", "")
        if (not re.fullmatch(r"step-[0-9]{6}-[0-9a-f]{32}", generation) or
                pointer != summary["latest_training_snapshot"] or pointer != manifest["metadata"]["latest_training_snapshot"]):
            raise ValueError("training recovery pointer identity differs")
        folder = run/"training-snapshots"/generation
        if not folder.resolve().is_relative_to(run) or digest(folder/"manifest.json", 64*1024) != pointer["manifest_sha256"]:
            raise ValueError("training recovery manifest identity differs")
        recovery_manifest = json.loads((folder/"manifest.json").read_text())
        if (recovery_manifest["step"] != summary["steps"] or recovery_manifest["contract"] != summary["recovery_contract"] or
                digest(folder/"tensors.pt", 256*1024**2) != recovery_manifest["tensors_sha256"]):
            raise ValueError("training recovery artifact/recipe identity differs")
        if set(recovery_manifest["ledgers"]) != {"curve.jsonl", "training.jsonl", "evaluations.jsonl"}:
            raise ValueError("training recovery ledger set differs")
        for name, cursor in recovery_manifest["ledgers"].items():
            if (run/name).stat().st_size != cursor["bytes"] or digest(run/name, 16*1024**2) != cursor["sha256"]:
                raise ValueError("complete-run recovery ledger prefix differs")
        if not 0 < summary["updates_this_run"] <= job["work_items"]:
            raise ValueError("native new-update count exceeds declared segment cap")
        if summary["resume"] is not None:
            resume = summary["resume"]
            parent = Path(resume["source_run"]).resolve()
            if not parent.is_relative_to(ROOT/"runs"):
                raise ValueError("resume evidence parent escaped owned runs")
            verify_source(parent/"code", summary["catalogue_sha256"])
            parent_pointer = resume["pointer"]
            if not re.fullmatch(r"step-[0-9]{6}-[0-9a-f]{32}", parent_pointer["generation"]):
                raise ValueError("resume parent generation invalid")
            parent_folder = parent/"training-snapshots"/parent_pointer["generation"]
            if not parent_folder.resolve().is_relative_to(parent) or digest(parent_folder/"manifest.json", 64*1024) != parent_pointer["manifest_sha256"]:
                raise ValueError("resume parent manifest identity differs")
            parent_manifest = json.loads((parent_folder/"manifest.json").read_text())
            if (parent_manifest["step"] != resume["step"] or parent_manifest["contract"] != summary["recovery_contract"] or
                    resume["step"]+summary["updates_this_run"] != summary["steps"] or
                    digest(parent_folder/"tensors.pt", 256*1024**2) != parent_manifest["tensors_sha256"] or
                    set(parent_manifest["ledgers"]) != {"curve.jsonl", "training.jsonl", "evaluations.jsonl"}):
                raise ValueError("resume parent tensor/step/recipe identity differs")
            for name, cursor in parent_manifest["ledgers"].items():
                size = cursor["bytes"]
                if type(size) is not int or not 0 <= size <= 16*1024**2 or cursor["sha256"] != resume["ledger_prefix_sha256"][name]:
                    raise ValueError("resume prefix cursor differs")
                for file in (parent/name, run/name):
                    with file.open("rb") as stream:
                        prefix = stream.read(size)
                    if len(prefix) != size or hashlib.sha256(prefix).hexdigest() != cursor["sha256"]:
                        raise ValueError("parent/resumed committed ledger prefix differs")
            parent_job = json.loads((parent.parent/(parent.name+"-job.json")).read_text())
            resume_parent = {"step": resume["step"], "run": str(parent), "native_status": parent_job["status"],
                             "native_exit_code": parent_job["exit_code"], "manifest_sha256": parent_pointer["manifest_sha256"],
                             "tensors_sha256": parent_manifest["tensors_sha256"]}
            resume_files = [(parent_folder/"manifest.json", "resume-parent-snapshot-manifest.json"),
                            (parent/"code/source.json", "resume-parent-source-manifest.json")]
            for suffix in ("-job.json", ".log", ".stderr.log"):
                resume_files.append((parent.parent/(parent.name+suffix), "resume-parent"+suffix))
            enclosing = parent.parent/(parent.name+"-pipeline-job.json")
            if enclosing.exists():
                if enclosing.stat().st_size > 64*1024:
                    raise ValueError("resume enclosing receipt budget exceeded")
                enclosing_job = json.loads(enclosing.read_text())
                resume_parent.update(enclosing_native_status=enclosing_job["status"],
                                     enclosing_native_exit_code=enclosing_job["exit_code"],
                                     child_receipt_stale_after_ancestor_stop=parent_job["exit_code"] is None and enclosing_job["exit_code"] is not None)
                for suffix in ("-job.json", ".log", ".stderr.log"):
                    resume_files.append((parent.parent/(parent.name+"-pipeline"+suffix), "resume-parent-enclosing"+suffix))
            if (parent/"summary.json").exists():
                resume_files.append((parent/"summary.json", "resume-parent-summary.json"))
    ledger = run/"training.jsonl"
    if digest(ledger, 16*1024**2) != summary["stream_ledger_sha256"] or not 1 <= summary["steps"] <= 4096:
        raise ValueError("stream ledger budget/identity mismatch")
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    if [row["step"] for row in rows] != list(range(1, summary["steps"]+1)):
        raise ValueError("stream ledger incomplete")
    if new_curriculum_contract and len({row["group_id"] for row in rows}) != summary["fresh_parent_groups"]:
        raise ValueError("distinct training ancestry count differs")
    for row in rows:
        if bool(row.get("final_capture", False)) != bool(summary.get("component_capacity", 0)):
            raise ValueError("stream capture policy differs from joint model contract")
        if new_curriculum_contract:
            replay = training_case(row["step"], summary["arguments"]["seed"], curriculum=curriculum,
                                   component_capacity=summary.get("component_capacity", 0), profile_square=summary["profile_square"])
        else:
            extra = {"final_capture": True} if row.get("final_capture") else {}
            replay = sample_case(row["group_index"], row["view"], row["split"], row["base_seed"], profile_square=row["profile_square"], **extra)
        if {key: value for key, value in row.items() if key != "step"} != json.loads(json.dumps(replay.record)):
            raise ValueError("stream source does not reproduce the recorded pixels/labels/recipe")
    curve = [json.loads(line) for line in (run/"curve.jsonl").read_text().splitlines()]
    if [row["sample_id"] for row in curve] != [row["sample_id"] for row in rows]:
        raise ValueError("optimizer steps and source ledger differ")
    probe_images = 0
    for split in ("training", "development"):
        lines = (run/(split+"-probe.jsonl")).read_text().splitlines()
        if new_curriculum_contract:
            specs = probe_specs(split, 3 if split == "training" else summary["development_groups"], curriculum)
            if len(lines) != len(specs):
                raise ValueError("declared probe population differs")
        for index, line in enumerate(lines):
            row = json.loads(line)
            if bool(row.get("final_capture", False)) != bool(summary.get("component_capacity", 0)):
                raise ValueError("probe capture policy differs from joint model contract")
            if new_curriculum_contract:
                group, view = specs[index]
                replay = sample_for(group, view, split, summary["arguments"]["seed"], curriculum=curriculum,
                                    component_capacity=summary.get("component_capacity", 0))
            else:
                extra = {"final_capture": True} if row.get("final_capture") else {}
                replay = sample_case(row["group_index"], row["view"], row["split"], row["base_seed"], **extra)
            if json.loads(json.dumps(replay.record)) != row:
                raise ValueError("probe metadata/pixels changed")
            folder = run/(split+"-probe")
            with Image.open(folder/(row["sample_id"]+".png")) as stored:
                if stored.mode != replay.frame.image.mode or stored.size != replay.frame.image.size or stored.tobytes() != replay.frame.image.tobytes():
                    raise ValueError("saved probe raster changed")
            if digest(folder/(row["sample_id"]+".json")) != row["state_sha256"]:
                raise ValueError("saved probe annotation changed")
            probe_images += 1
    if curriculum in ("all-class-fit-v1", "all-class-interleaved-v1"):
        for field in ("initial_training_probe", "final_training_probe"):
            score = summary[field]
            actual_fit = summarize_fit(score["cases"])
            if (actual_fit != score.get("fit") or score["images"] != 32 or score["ancestry_groups"] != 16 or
                    score["exact_locked_boards"] != actual_fit["exact_locked_boards"] or
                    score["exact_known_layers"] != actual_fit["exact_known_layers"] or score["known_layers"] != 64 or
                    score["mean_geometry_error_cells"] != actual_fit["mean_geometry_error_cells"]):
                raise ValueError("fit aggregates differ from32 declared cases")
    evidence = ROOT/"docs/evidence"/args.name
    evidence.mkdir(exist_ok=False)
    for name in ("summary.json", "progress.json", "training.jsonl", "curve.jsonl", "evaluations.jsonl", "training-probe.jsonl", "development-probe.jsonl"):
        shutil.copyfile(run/name, evidence/name)
    if summary.get("resource_profile_version", 0) >= 3:
        if (run/"memory.jsonl").stat().st_size > 1024**2:
            raise ValueError("memory trajectory byte budget exceeded")
        shutil.copyfile(run/"memory.jsonl", evidence/"memory.jsonl")
    shutil.copyfile(code/"source.json", evidence/"source-manifest.json")
    shutil.copyfile(run/"checkpoint/manifest.json", evidence/"checkpoint-manifest.json")
    if recovery_manifest is not None:
        shutil.copyfile(run/"latest-training.json", evidence/"latest-training.json")
        (evidence/"training-snapshot-manifest.json").write_text(json.dumps(recovery_manifest, indent=2)+"\n")
    for source, name in resume_files:
        shutil.copyfile(source, evidence/name)
    for suffix in ("-job.json", ".log", ".stderr.log"):
        source = ROOT/"runs"/(args.name+suffix)
        shutil.copyfile(source, evidence/source.name)
    comparison_status = None
    if (run/"analysis.json").exists():
        if (run/"analysis.json").stat().st_size > 64*1024:
            raise ValueError("comparison analysis byte budget exceeded")
        comparison = json.loads((run/"analysis.json").read_text())
        if comparison["weights_sha256"] != summary["checkpoint_sha256"] or comparison["source_catalogue_sha256"] != summary["catalogue_sha256"]:
            raise ValueError("comparison is for another artifact/source")
        shutil.copyfile(run/"analysis.json", evidence/"analysis.json")
        comparison_status = comparison["status"]
    scope = {"status": "verified_partial" if partial else "passed", "run_status": summary["status"],
             "steps": summary["steps"], "requested_steps": summary["requested_steps"],
             "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
             "source_catalogue_sha256": archive["catalogue_sha256"], "weights_sha256": summary["checkpoint_sha256"],
             "replayed_training_samples": len(rows), "verified_saved_probe_images": probe_images,
             "native_peak_job_commit_bytes": job["peak_job_commit_bytes"], "all_final_tensors_reload_exact": True,
             "source_archive": str(code), "scope": "Bounded curriculum plumbing/source binding/resource admission, not model accuracy.",
             "test_gate_scope": "Separate named, source-pinned fixture receipts; this curator does not attach historical fixture logs to new runs.",
             "training_snapshot_identity_verified": recovery_manifest is not None,
             "comparison_status": comparison_status,
             "updates_this_run": summary.get("updates_this_run", summary["steps"]), "resume_parent": resume_parent,
             "curator_sha256": digest(Path(__file__)), "core_architecture": architecture,
             "freeze_backbone": summary.get("freeze_backbone", False),
             "initial_weights_from": summary.get("initial_weights_from"),
             "component_capacity": summary["config"].get("component_capacity", 0),
             "layer_evidence": summary["config"].get("layer_evidence", False), "curriculum": curriculum,
             "type_head_rate_factor": factor,
             "distinct_training_ancestries": len({row["group_id"] for row in rows})}
    (evidence/"scope.json").write_text(json.dumps(scope, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": scope["status"], "run_status": summary["status"], "steps": summary["steps"],
                      "requested_steps": summary["requested_steps"], "replayed_samples": len(rows),
                      "verified_probe_images": probe_images, "scope": str(evidence/"scope.json")}))


if __name__ == "__main__":
    main()


RECEIPT_KEYS = {
    "supervisor_pid", "command", "memory_bytes", "wall_seconds", "work_items", "cpu_library_threads",
    "child_executable", "job_name", "guard", "supervisor_script_sha256", "system_pressure_guard",
    "system_at_admission", "system_pressure", "status", "exit_code", "elapsed_seconds", "child_pid",
    "last_system_memory", "peak_job_commit_bytes",
}


def job_receipt_problems(receipt):
    """Named problems with a supervisor receipt; empty only for a clean passed job."""
    if type(receipt) is not dict:
        return {"receipt": "object required"}
    if set(receipt) != RECEIPT_KEYS:
        return {"keys": "expected %s, got %s" % (sorted(RECEIPT_KEYS), sorted(receipt))}
    problems = {}
    if receipt["status"] != "passed":
        problems["status"] = "status %r" % (receipt["status"],)
    if (type(receipt["exit_code"]) is not int or isinstance(receipt["exit_code"], bool) or
            receipt["exit_code"] != 0):
        problems["exit_code"] = "exit_code %r" % (receipt["exit_code"],)
    if type(receipt["elapsed_seconds"]) is not float or not receipt["elapsed_seconds"] > 0:
        problems["elapsed_seconds"] = "elapsed_seconds %r" % (receipt["elapsed_seconds"],)
    memory = receipt["memory_bytes"]
    if type(memory) is not int or isinstance(memory, bool) or not 0 < memory <= 8*1024**3:
        problems["memory_bytes"] = "memory_bytes %r" % (memory,)
    peak = receipt["peak_job_commit_bytes"]
    if type(peak) is not int or isinstance(peak, bool) or not 0 < peak <= memory:
        problems["peak_job_commit_bytes"] = "peak %r against ceiling %r" % (peak, memory)
    command = receipt["command"]
    if (type(command) is not list or len(command) < 2 or
            not all(type(part) is str for part in command) or "unittest" not in command or
            "discover" not in command):
        problems["command"] = "command %r" % (command,)
    threads = receipt["cpu_library_threads"]
    if type(threads) is not dict or not threads or any(value != "1" for value in threads.values()):
        problems["cpu_library_threads"] = "thread policy %r" % (threads,)
    pressure = receipt["system_pressure"]
    if type(pressure) is not dict or pressure.get("reason") is not None:
        problems["system_pressure"] = "pressure reason %r" % (pressure,)
    if type(receipt["work_items"]) is not int or receipt["work_items"] <= 0:
        problems["work_items"] = "work_items %r" % (receipt["work_items"],)
    if type(receipt["wall_seconds"]) is not float or not receipt["wall_seconds"] > 0:
        problems["wall_seconds"] = "wall_seconds %r" % (receipt["wall_seconds"],)
    return problems
