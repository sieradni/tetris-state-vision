"""Finite, source-frozen CPU curriculum. Resource profile required before a study."""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

from probe_model import bounded_job_snapshot, require_bounded_job
from source_snapshot import capture_source, verify_source
from atomic_write import replace_retry

ROOT = Path(__file__).resolve().parents[1]
NATIVE_CACHE_LIMITS = {"ONEDNN_PRIMITIVE_CACHE_CAPACITY": "16", "DNNL_PRIMITIVE_CACHE_CAPACITY": "16"}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=("pilot", "accuracy"), required=True)
    parser.add_argument("--component-capacity", type=int, default=0,
                        help="0 keeps M2;10..64 enables joint heads and final-capture data")
    parser.add_argument("--layer-evidence", action="store_true", help="optional independent visible-cell layer supervision/readout")
    parser.add_argument("--curriculum", choices=("stream-v1", "all-class-fit-v1"), default="stream-v1")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--wall-seconds", type=float, default=300)
    parser.add_argument("--evaluate-every", type=int, default=128)
    parser.add_argument("--development-groups", type=int, default=6)
    parser.add_argument("--profile-square", action="store_true")
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=64)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--max-new-steps", type=int)
    args = parser.parse_args(argv)
    if not 1 <= args.steps <= 4096 or not 0 <= args.seed <= 2**31-1:
        parser.error("steps 1..4096 and finite nonnegative seed required")
    if args.component_capacity != 0 and not 10 <= args.component_capacity <= 64:
        parser.error("joint stream component capacity must be10..64 to cover the renderer's10 slots;0 keeps M2")
    if args.curriculum == "all-class-fit-v1" and args.component_capacity:
        parser.error("all-class fit is a capacity0 diagnostic")
    if not math.isfinite(args.wall_seconds) or not 0 < args.wall_seconds <= 1800 or not math.isfinite(args.learning_rate) or not 0 < args.learning_rate <= .01:
        parser.error("wall budget (0,1800] and learning rate (0,.01] required")
    if not 1 <= args.evaluate_every <= 1024 or not 3 <= args.development_groups <= 12:
        parser.error("evaluation interval 1..1024 and development groups 3..12 required")
    if not 1 <= args.checkpoint_every <= 1024 or args.max_new_steps is not None and not 1 <= args.max_new_steps <= 4096:
        parser.error("checkpoint interval1..1024 and new-update cap1..4096 required")
    if args.profile_square and (args.steps != 2 or args.admission is not None or args.resume_from is not None or args.max_new_steps is not None):
        parser.error("square resource profile requires two optimizer steps without prior admission")
    if not args.profile_square and args.admission is None:
        parser.error("a matching warm-optimizer square resource admission is required")
    return args


def validate_admission(summary, job, *, catalogue, model, config, ceiling, development_groups, supervisor_sha, curriculum="stream-v1"):
    if (type(curriculum) is not str or curriculum not in ("stream-v1", "all-class-fit-v1") or
            summary.get("curriculum", "stream-v1") != curriculum):
        raise ValueError("curriculum differs from resource admission")
    if "curriculum" in summary and (summary.get("arguments", {}).get("curriculum") != curriculum or
                                    summary.get("recovery_contract", {}).get("curriculum") != curriculum):
        raise ValueError("resource arguments/recovery curriculum identity differs")
    warm = summary.get("warm_optimizer_backward", {})
    if (summary.get("status") != "complete" or summary.get("steps") != 2 or not summary.get("profile_square") or
            summary.get("resource_profile_version") != 3 or not summary.get("training_recovery_profile_exact") or
            warm.get("step") != 2 or warm.get("input_pixels") != 512**2 or
            not warm.get("all_parameters_initialized") or not warm.get("backward_completed") or warm.get("state_bytes", 0) <= 0 or
            not summary.get("checkpoint_round_trip_exact") or summary.get("max_input_pixels_seen") != 512**2 or
            summary.get("native_cache_limits") != NATIVE_CACHE_LIMITS or
            any(summary.get(key) != value for key, value in {"max_side": 512, "threads": 1, "workers": 0, "prefetch": 0, "tensor_cache_bytes": 0}.items()) or
            job.get("status") != "passed" or job.get("exit_code") != 0 or job.get("work_items") != 2):
        raise ValueError("complete two-step square warm-optimizer resource admission required")
    if (summary["catalogue_sha256"] != catalogue or summary["model_kind"] != model or summary["config"] != config or
            summary["job_ceiling_bytes"] != ceiling or job["memory_bytes"] != ceiling or
            summary["development_groups"] < development_groups or not summary["cpu_only"] or
            summary.get("system_pressure_guard") != "GetPerformanceInfo/0.2s" or job.get("system_pressure_guard") != "GetPerformanceInfo/0.2s" or
            summary.get("supervisor_script_sha256") != supervisor_sha or job.get("supervisor_script_sha256") != supervisor_sha):
        raise ValueError("source/config/input/CPU/memory envelope differs from resource admission")


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    replace_retry(temporary, path)


def equal_tree(left, right):
    """Exact nested tensor parity for warm recovery and M2/M3 final outputs."""
    import torch
    if torch.is_tensor(left):
        return (torch.is_tensor(right) and left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right))
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(equal_tree(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(equal_tree(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def main(argv=None, *, _archive=None, _started=None):
    args = arguments(argv)
    ceiling = require_bounded_job()
    args.output = args.output.resolve()
    if _archive is None:
        args.output.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        atomic_json(args.output/"progress.json", {"status": "capturing_source", "pid": os.getpid(), "step": 0})
        code = args.output/"code"
        archive = capture_source(ROOT, code)
        # Execute the driver itself from the verified archive, so tracebacks
        # cannot read later parent edits. This stays in the same native Job/PID.
        sys.path[:0] = [str(code/"scripts"), str(code)]
        sys.dont_write_bytecode = True
        for name in ("probe_model", "source_snapshot", "atomic_write", "run_bounded"):
            sys.modules.pop(name, None)
        spec = importlib.util.spec_from_file_location("frozen_training_driver", code/"scripts/train_m2_stream.py")
        driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(driver)
        return driver.main(argv, _archive=archive, _started=started)
    started = _started
    progress = {"status": "capturing_source", "pid": os.getpid(), "step": 0, "requested_steps": args.steps}
    atomic_json(args.output/"progress.json", progress)
    code = args.output/"code"
    archive = verify_source(code, _archive["catalogue_sha256"])
    if not Path(__file__).resolve().is_relative_to(code):
        raise RuntimeError("trainer main did not execute from archived source")
    supervisor_sha = archive["files"]["scripts/run_bounded.py"]
    if os.environ.get("TETRIS_VISION_SYSTEM_GUARD") != "GetPerformanceInfo/0.2s" or os.environ.get("TETRIS_VISION_SUPERVISOR_SHA256") != supervisor_sha:
        raise RuntimeError("supervisor source/OS pressure guard does not match the archived runner")
    sys.path[:0] = [str(code/"scripts"), str(code)]
    sys.dont_write_bytecode = True
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ.update(NATIVE_CACHE_LIMITS)
    import torch
    from dataclasses import asdict, replace
    from m2_curriculum import ALL_CLASS, distinct_training_groups, probe_specs, sample_for, training_case
    from m2_fit_contract import summarize_fit
    from training_snapshot import load_training_snapshot, save_training_snapshot
    from tetris_vision.checkpoint import file_digest, load_checkpoint, save_checkpoint
    from tetris_vision.dataset import source_digest
    from tetris_vision.losses import PIECE_NAMES, diagnostic_metrics, joint_loss, targets_for
    from tetris_vision.model import ModelConfig, TetrisStateNet
    from tetris_vision.preprocess import prepare_image
    import tetris_vision.model as model_module
    if (not Path(model_module.__file__).resolve().is_relative_to(code) or
            any(not Path(function.__code__.co_filename).resolve().is_relative_to(code) for function in (sample_for, training_case, summarize_fit))):
        raise RuntimeError("runtime did not import from the verified source archive")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    config = ModelConfig.pilot() if args.model == "pilot" else ModelConfig()
    config = replace(config, component_capacity=args.component_capacity, layer_evidence=args.layer_evidence)
    config_json = json.loads(json.dumps(asdict(config)))
    if args.admission is not None:
        admit = args.admission.resolve()
        validate_admission(json.loads((admit/"summary.json").read_text()),
                           json.loads((admit.parent/(admit.name+"-job.json")).read_text()),
                           catalogue=archive["catalogue_sha256"], model=args.model, config=config_json,
                           ceiling=ceiling, development_groups=args.development_groups, supervisor_sha=supervisor_sha, curriculum=args.curriculum)
    model = TetrisStateNet(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
    maximum_pixels, evaluation_identities = 0, {}
    warm_optimizer_backward = None
    recovery_profile_exact = False

    def sample(group, view, split, *, profile_square=False):
        return sample_for(group, view, split, args.seed, profile_square=profile_square,
                          curriculum=args.curriculum, component_capacity=args.component_capacity)

    def prepare(case):
        nonlocal maximum_pixels
        prepared = prepare_image(case.frame.image, max_side=512)
        maximum_pixels = max(maximum_pixels, prepared.pixels.shape[-2]*prepared.pixels.shape[-1])
        if maximum_pixels > 512**2:
            raise ValueError("512-square input budget exceeded")
        target = targets_for(case.frame.state, prepared, component_capacity=args.component_capacity,
                             layer_evidence=args.layer_evidence)
        if int(target["known"].sum()) != 200:
            raise ValueError("curriculum input admission requires all 200 visible locked cells")
        return prepared, target

    @torch.no_grad()
    def evaluate(split, groups, persist=False):
        model.eval()
        records, per_case = [], []
        for group, view in probe_specs(split, groups, args.curriculum):
            case = sample(group, view, split)
            prepared, target = prepare(case)
            output = model(prepared.pixels[None], prepared.valid[None])
            loss, _ = joint_loss(output, target)
            layers_right = layers_known = 0
            fit_roles = {}
            for layer, role in enumerate(("active", "ghost")):
                known = bool(target["layer_known"][0, layer])
                observed = bool(target["pose_known"][0, layer])
                right = None
                if known:
                    layers_known += 1
                    final = output["final"]
                    right = int(final["layer_type"][0, layer].argmax()) == int(target["layer_type"][0, layer]) and bool(final["layer_known"][0, layer] > 0)
                    if observed:
                        right = (right and int(final["pose_column"][0, layer].argmax()) == int(target["column"][0, layer]) and
                                 int(final["pose_row"][0, layer].argmax()) == int(target["row"][0, layer]) and
                                 bool(target["rotations"][0, layer, final["pose_rotation"][0, layer].argmax()]))
                    layers_right += int(right)
                if args.curriculum == ALL_CLASS and split == "training":
                    kind = int(target["layer_type"][0, layer])
                    fit_roles[role] = {"known": known, "observed": observed, "exact": right,
                                       "class": (PIECE_NAMES[kind-1] if kind else "absent") if known else "unknown"}
            per_case.append({"sample_id": case.record["sample_id"], "loss": float(loss),
                             "exact_known_layers": layers_right, "known_layers": layers_known,
                             **diagnostic_metrics(output, target)})
            if args.curriculum == ALL_CLASS and split == "training":
                typed = target["typed"]
                correct = (output["final"]["semantic"][:, :20].argmax(-1) == target["semantic"]) & typed
                per_case[-1].update(group_id=case.record["group_id"],
                                    fit_case={"semantic_known": int(typed.sum()), "semantic_correct": int(correct.sum()), "roles": fit_roles})
            if args.component_capacity:
                from tetris_vision.components import component_metrics
                per_case[-1]["components"] = component_metrics(output["final"]["components"], target["components"])[0]
            records.append(case.record)
            if persist:
                folder = args.output/(split+"-probe")
                folder.mkdir(exist_ok=True)
                case.frame.image.save(folder/(case.record["sample_id"]+".png"))
                (folder/(case.record["sample_id"]+".json")).write_bytes(case.frame.state.to_json().encode())
        identity = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
        if split in evaluation_identities and evaluation_identities[split] != identity:
            raise ValueError("evaluation cases changed during training")
        evaluation_identities[split] = identity
        if persist:
            (args.output/(split+"-probe.jsonl")).write_text("".join(json.dumps(row)+"\n" for row in records))
        result = {"images": len(per_case), "ancestry_groups": len({row["group_id"] for row in records}), "loss": sum(row["loss"] for row in per_case)/len(per_case),
                "exact_locked_boards": sum(row["exact_locked_board"] for row in per_case),
                "cell_accuracy": sum(row["cell_accuracy"] for row in per_case)/len(per_case),
                "mean_geometry_error_cells": sum(row["mean_geometry_error_cells"] for row in per_case)/len(per_case),
                "exact_known_layers": sum(row["exact_known_layers"] for row in per_case),
                "known_layers": sum(row["known_layers"] for row in per_case), "cases": per_case}
        if args.layer_evidence:
            for key in ("dense_layer_grids_known", "dense_layer_grids_exact",
                        "dense_observed_visible_grids_known", "dense_observed_visible_grids_exact"):
                result[key] = sum(row[key] for row in per_case)
        if args.component_capacity:
            result.update(readable_component_cases=sum(row["components"]["component_state_readable"] for row in per_case),
                          exact_readable_components=sum(row["components"]["exact_readable_components"] is True for row in per_case),
                          supported_components_exact=sum(row["components"]["supported_components_exact"] for row in per_case))
        if args.curriculum == ALL_CLASS and split == "training":
            result["fit"] = summarize_fit(per_case)
        return result

    initial_train, initial_dev = evaluate("training", 3, True), evaluate("development", args.development_groups, True)
    completed = 0
    recovery_contract = {"catalogue_sha256": archive["catalogue_sha256"], "config": config_json,
                         "seed": args.seed, "learning_rate": args.learning_rate, "weight_decay": .01,
                         "steps": args.steps, "max_side": 512, "development_groups": args.development_groups,
                         "evaluate_every": args.evaluate_every, "checkpoint_every": args.checkpoint_every,
                         "profile_square": args.profile_square, "threads": 1, "rng": "torch_cpu_only"}
    recovery_contract["native_cache_limits"] = NATIVE_CACHE_LIMITS
    recovery_contract["input_policy"] = "M3-final-capture-v1" if args.component_capacity else "M2-prepared-resize-v1"
    recovery_contract["curriculum"] = args.curriculum
    ledger_names = ("curve.jsonl", "training.jsonl", "evaluations.jsonl")
    resume_record = None
    if args.resume_from is not None:
        recovery = load_training_snapshot(args.resume_from, model, optimizer, contract=recovery_contract)
        completed = recovery["step"]
        if completed >= args.steps or recovery["ledger_prefixes"].keys() != set(ledger_names):
            raise ValueError("resume requires an unfinished compatible stream and all three committed ledgers")
        for name, data in recovery["ledger_prefixes"].items():
            (args.output/name).write_bytes(data)
        # The source cursor is authoritative; later uncheckpointed appends are
        # retained in the parent run and never replayed into the resumed stream.
        for name in ("curve.jsonl", "training.jsonl"):
            records = [json.loads(line) for line in (args.output/name).read_text().splitlines()]
            if [row["step"] for row in records] != list(range(1, completed+1)):
                raise ValueError("snapshot ledger does not match its global step")
        resume_record = {"source_run": str(args.resume_from.resolve()), "step": completed, "pointer": recovery["pointer"],
                         "ledger_prefix_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in recovery["ledger_prefixes"].items()}}
    started_step = completed
    final_step = min(args.steps, started_step+(args.max_new_steps or args.steps))
    latest_snapshot = None
    mode = "a" if resume_record else "x"
    with (args.output/"curve.jsonl").open(mode, encoding="utf-8", newline="\n") as curve, (args.output/"training.jsonl").open(mode, encoding="utf-8", newline="\n") as ledger, (args.output/"evaluations.jsonl").open(mode, encoding="utf-8", newline="\n") as evaluations, (args.output/"memory.jsonl").open("x", encoding="utf-8") as memory_log:
        if not resume_record:
            evaluations.write(json.dumps({"step": 0, "training": initial_train, "development": initial_dev})+"\n")
        evaluations.flush()

        def snapshot():
            for writer in (curve, ledger, evaluations):
                writer.flush()
            return save_training_snapshot(args.output, model, optimizer, step=completed,
                                          contract=recovery_contract, ledgers=ledger_names)

        for step in range(started_step+1, final_step+1):
            if time.monotonic()-started >= args.wall_seconds:
                break
            # Both characterization steps must use the maximum shape. Step two
            # includes Adam's moment tensors, allocated lazily by the first step.
            case = training_case(step, args.seed, curriculum=args.curriculum, component_capacity=args.component_capacity, profile_square=args.profile_square)
            prepared, target = prepare(case)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            if args.profile_square and step == 2:
                parameters = [p for p in model.parameters() if p.requires_grad]
                initialized = all(
                    p in optimizer.state and all(
                        key in optimizer.state[p] and optimizer.state[p][key].shape == p.shape
                        for key in ("exp_avg", "exp_avg_sq"))
                    and float(optimizer.state[p]["step"]) == 1 for p in parameters)
                if not parameters or not initialized:
                    raise RuntimeError("resource profile requires resident Adam state for every trainable parameter")
                warm_optimizer_backward = {
                    "step": step, "input_pixels": prepared.pixels.shape[-2]*prepared.pixels.shape[-1],
                    "parameter_tensors": len(parameters), "all_parameters_initialized": initialized,
                    "state_bytes": sum(value.numel()*value.element_size() for state in optimizer.state.values()
                                       for value in state.values() if torch.is_tensor(value)),
                    "backward_completed": False}
            output = model(prepared.pixels[None], prepared.valid[None])
            loss, terms = joint_loss(output, target)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite curriculum loss")
            loss.backward()
            if args.profile_square and step == 2:
                warm_optimizer_backward["backward_completed"] = True
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            completed = step
            item = {"step": step, "sample_id": case.record["sample_id"], "loss": float(loss.detach()),
                    "terms": {key: float(value.detach()) for key, value in terms.items()},
                    "gradient_norm": float(norm), "elapsed_seconds": time.monotonic()-started}
            curve.write(json.dumps(item, allow_nan=False)+"\n")
            ledger.write(json.dumps({"step": step, **case.record}, allow_nan=False)+"\n")
            curve.flush()
            ledger.flush()
            if step == started_step+1 or step % 16 == 0 or step == final_step:
                memory_log.write(json.dumps({"step": step, "input_shape": list(prepared.pixels.shape),
                                             "phase": "post_update", **bounded_job_snapshot()})+"\n")
                memory_log.flush()
            progress.update(status="training", step=step, loss=item["loss"], elapsed_seconds=item["elapsed_seconds"])
            atomic_json(args.output/"progress.json", progress)
            print(json.dumps(item), flush=True)
            if step % args.evaluate_every == 0:
                evaluations.write(json.dumps({"step": step, "development": evaluate("development", args.development_groups)})+"\n")
                evaluations.flush()
            if step % args.checkpoint_every == 0 or step == final_step or args.profile_square:
                latest_snapshot = snapshot()
                progress["recovery_step"] = completed
                atomic_json(args.output/"progress.json", progress)
            if args.profile_square and step == 1:
                # Characterize actual recovery allocation as well as warm
                # backward. Deep copies intentionally make this a conservative
                # peak; the native cap remains authoritative.
                expected_state = copy.deepcopy((model.state_dict(), optimizer.state_dict(), torch.get_rng_state()))
                recovery = load_training_snapshot(args.output, model, optimizer, contract=recovery_contract)
                recovery_profile_exact = recovery["step"] == 1 and equal_tree(expected_state, (model.state_dict(), optimizer.state_dict(), torch.get_rng_state()))
                del expected_state, recovery
                if not recovery_profile_exact:
                    raise RuntimeError("training snapshot changed model/optimizer/RNG")
        if completed > started_step and (latest_snapshot is None or progress.get("recovery_step") != completed):
            latest_snapshot = snapshot()
            progress["recovery_step"] = completed
    final_train, final_dev = evaluate("training", 3), evaluate("development", args.development_groups)
    verify_source(code, archive["catalogue_sha256"])
    metadata = {"trained_steps": completed, "catalogue_sha256": archive["catalogue_sha256"],
                "supervisor_script_sha256": supervisor_sha, "system_pressure_guard": "GetPerformanceInfo/0.2s",
                "runtime_source_sha256": source_digest(), "model_kind": args.model,
                "component_capacity": args.component_capacity, "input_policy": recovery_contract["input_policy"],
                "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "stream_ledger_sha256": file_digest(args.output/"training.jsonl"), "cpu_only": torch.version.cuda is None,
                "resume": resume_record, "latest_training_snapshot": latest_snapshot,
                "driver_source_path": str(Path(__file__).resolve()),
                "native_cache_limits": NATIVE_CACHE_LIMITS,
                "recovery_contract": recovery_contract, "updates_this_run": completed-started_step}
    checkpoint = save_checkpoint(args.output/"checkpoint", model.eval(), metadata)
    restored, _ = load_checkpoint(args.output/"checkpoint")
    example, _ = prepare(sample(0, 0, "development"))
    with torch.no_grad():
        expected = model(example.pixels[None], example.valid[None])["final"]
        actual = restored.eval()(example.pixels[None], example.valid[None])["final"]
    if not equal_tree(expected, actual):
        raise RuntimeError("saved checkpoint changes a final output tensor")
    status = "complete" if completed == args.steps else "step_partial" if completed == final_step else "wall_partial"
    summary = {"status": status, "steps": completed, "requested_steps": args.steps,
               "resource_profile_version": 3, "training_recovery_profile_exact": recovery_profile_exact,
               "warm_optimizer_backward": warm_optimizer_backward,
               "profile_square": args.profile_square, "max_input_pixels_seen": maximum_pixels, "max_side": 512,
               "job_ceiling_bytes": ceiling, "development_groups": args.development_groups, "config": config_json,
               "threads": 1, "workers": 0, "prefetch": 0, "tensor_cache_bytes": 0, "fresh_parent_groups": distinct_training_groups(completed, args.curriculum),
               "curriculum": args.curriculum,
               "initial_training_probe": initial_train, "initial_development": initial_dev,
               "final_training_probe": final_train, "development": final_dev, "checkpoint_round_trip_exact": True,
               "checkpoint_sha256": checkpoint["weights_sha256"], "elapsed_seconds": time.monotonic()-started,
               "claim": "Finite synthetic curriculum or resource characterization, not complete-state or real-image acceptance.", **metadata}
    atomic_json(args.output/"summary.json", summary)
    progress.update(status=summary["status"], step=completed, summary="summary.json", checkpoint="checkpoint/manifest.json")
    atomic_json(args.output/"progress.json", progress)
    print(json.dumps({"status": summary["status"], "steps": completed, "max_input_pixels_seen": maximum_pixels,
                      "checkpoint_round_trip_exact": True, "catalogue_sha256": archive["catalogue_sha256"]}), flush=True)
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
