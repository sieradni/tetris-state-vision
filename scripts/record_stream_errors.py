"""Curate bounded ordinary M2/M3 saved-probe diagnostics; no quality claim."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

from m2_error_analysis import fit_metrics_from_diagnostics, recipe_strata, summarize
from m3_error_analysis import summarize_components
from probe_model import require_bounded_job
from probe_stream_errors import CANONICAL_DEVELOPMENT_GROUPS, extension_specs
from record_stream_run import checkpoint_architecture, require_owned_run, validate_completion
from source_snapshot import verify_source

ROOT = Path(__file__).resolve().parents[1]
REPORT_SOURCES = tuple("scripts/"+name for name in
    ("probe_stream_errors.py", "m2_error_analysis.py", "m3_error_analysis.py", "record_stream_run.py", "source_snapshot.py", "probe_model.py"))
# The sidecars this curator copies out of `runs/`. A probe receipt is `<name>-job.json`,
# but its captured stdout/stderr are `<name>.log` and `<name>.stderr.log`: a launch that
# names every sidecar `<name>-job.*` leaves the canonical log missing and the curation
# dies after the replay has already passed (V-053).
NATIVE_SIDECARS = ("-job.json", ".log", ".stderr.log")


def read(path, maximum=2*1024**2):
    if path.stat().st_size > maximum:
        raise ValueError("error evidence JSON byte budget exceeded")
    return json.loads(path.read_text())


def digest(path):
    value, consumed = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            consumed += len(block)
            if consumed > 64*1024**2:
                raise ValueError("error evidence artifact byte budget exceeded")
            value.update(block)
    return value.hexdigest()


def verify_extension(extension, *, recorded_groups, saved_recipes):
    """An extension is the canonical set, prefix-completed from saved cases."""
    if extension is None:
        return {}
    if type(extension) is not dict or type(recorded_groups) is not int or not saved_recipes:
        raise ValueError("development extension metadata required")
    base_seed = saved_recipes[0]["base_seed"]
    specs = extension_specs(base_seed, recorded_groups, CANONICAL_DEVELOPMENT_GROUPS)
    if (set(extension) != {"saved_groups", "evaluated_groups", "canonical_groups", "base_seed",
                           "saved_cases", "extension_cases", "recipes"} or
            extension["saved_groups"] != recorded_groups or
            extension["evaluated_groups"] != CANONICAL_DEVELOPMENT_GROUPS or
            extension["canonical_groups"] != CANONICAL_DEVELOPMENT_GROUPS or
            extension["base_seed"] != base_seed or extension["saved_cases"] != len(saved_recipes) or
            type(extension["recipes"]) is not list or len(extension["recipes"]) != len(specs) or
            extension["extension_cases"] != len(specs)):
        raise ValueError("development extension metadata differs")
    for spec, recipe in zip(specs, extension["recipes"]):
        if (type(recipe) is not dict or recipe.get("sample_id") != spec["sample_id"] or
                recipe.get("group_id") != spec["group_id"] or recipe.get("view") != spec["view"] or
                recipe.get("group_index") != spec["group_index"] or recipe.get("base_seed") != spec["base_seed"] or
                recipe.get("split") != "development"):
            raise ValueError("development extension recipe identity differs")
    return {recipe["sample_id"]: recipe for recipe in extension["recipes"]}


def validate_report(result, job, summary, manifest, recipes, *, name):
    """Pure metadata gate; neural/pixel replay belongs to the recorded worker."""
    curriculum = summary.get("curriculum", "stream-v1")
    max_cases = 68 if curriculum in ("all-class-fit-v1", "all-class-interleaved-v1") else 45
    if (result["status"] != "passed" or result.get("diagnostic_format_version") != 2 or
            job["status"] != "passed" or job["exit_code"] != 0 or
            type(result["forwards"]) is not int or not 1 <= result["forwards"] <= max_cases or
            job["work_items"] != result["forwards"] or result["forwards"] != len(result["cases"]) or
            result["weights_unchanged"] is not True or result["automatic_metrics_reproduced"] is not True or
            result["label_inputs_to_model"] is not False):
        raise ValueError("complete finite ordinary diagnostic gate required")
    command = job["command"]
    declared = result.get("development_extension")
    expected_tail = [] if declared is None else ["--development-groups", str(declared.get("evaluated_groups"))]
    if (len(command) != 6+len(expected_tail) or Path(command[1]).resolve() != ROOT/"scripts/probe_stream_errors.py" or
            command[2] != "--run" or command[3] != result["run"] or command[4] != "--output" or
            Path(command[5]).resolve() != ROOT/"runs"/name or command[6:] != expected_tail):
        raise ValueError("native report command/output identity differs")
    architecture = checkpoint_architecture(summary, manifest)
    if (result["run_status"] != summary["status"] or result["global_steps"] != summary["steps"] or
            result["target_steps"] != summary["requested_steps"] or
            result["weights_sha256"] != summary["checkpoint_sha256"] or manifest["weights_sha256"] != result["weights_sha256"] or
            result["source_catalogue_sha256"] != summary["catalogue_sha256"] or
            result["core_architecture"] != architecture or result["config"] != manifest["config"] or
            result["component_capacity"] != summary["config"].get("component_capacity", 0) or
            result.get("component_identity_factor", 1) != summary.get("arguments", {}).get("component_identity_factor", 1) or
            result.get("component_endpoint_term", "joint") != summary.get("arguments", {}).get("component_endpoint_term", "joint") or
            result.get("occupancy_factor", 1) != summary.get("arguments", {}).get("occupancy_factor", 1) or
            result["job_ceiling_bytes"] != job["memory_bytes"] or result.get("curriculum", "stream-v1") != curriculum):
        raise ValueError("report checkpoint/partial-status identity differs")
    capacity = result["component_capacity"]
    extension_recipes = verify_extension(result.get("development_extension"),
                                         recorded_groups=summary["development_groups"],
                                         saved_recipes=recipes["development"])
    totals = summarize(result["cases"], max_cases=max_cases)
    for split, field in (("training", "final_training_probe"), ("development", "development")):
        expected = {row["sample_id"]: row for row in summary[field]["cases"]}
        labels = {row["sample_id"]: row for row in recipes[split]}
        cases = [row for row in result["cases"] if row["split"] == split]
        saved_rows = [row for row in cases if row.get("saved", True)]
        extension_rows = [row for row in cases if row.get("saved") is False]
        split_limit = (32 if curriculum in ("all-class-fit-v1", "all-class-interleaved-v1") else 9) if split == "training" else 36
        if (not 1 <= len(saved_rows) <= split_limit or len(cases) > split_limit or
                len(expected) != len(summary[field]["cases"]) or len(labels) != len(recipes[split]) or
                len(saved_rows) != len(expected) or {row["sample_id"] for row in saved_rows} != set(expected) or
                set(labels) != set(expected)):
            raise ValueError("saved diagnostic population differs")
        if extension_recipes and split == "development":
            if len(extension_rows) != len(extension_recipes) or {row["sample_id"] for row in extension_rows} != set(extension_recipes):
                raise ValueError("development extension population differs")
        elif extension_rows:
            raise ValueError("unsaved diagnostic cases outside the development extension")
        for row in saved_rows:
            recipe = labels[row["sample_id"]]
            density, skin = recipe_strata(recipe)
            if row["group_id"] != recipe["group_id"] or row["view"] != recipe["view"] or row["skin"] != skin or row["density_recipe"] != density:
                raise ValueError("saved sample ancestry/recipe differs")
            ordinary = row["ordinary_metrics"]
            if ordinary != {key: value for key, value in expected[row["sample_id"]].items() if key != "sample_id"}:
                raise ValueError("saved ordinary metrics differ")
            diagnostic = row["diagnostics"]
            if curriculum in ("all-class-fit-v1", "all-class-interleaved-v1") and split == "training":
                if ordinary.get("fit_case") != fit_metrics_from_diagnostics(diagnostic) or ordinary.get("group_id") != recipe["group_id"]:
                    raise ValueError("fit raw-head decomposition differs")
            layers = [value for value in diagnostic["layers"].values() if value["target_known"]]
            if (ordinary["known_cells"] != diagnostic["locked"]["known_cells"] or
                    ordinary["exact_locked_board"] != diagnostic["locked"]["exact_board"] or
                    not math.isclose(ordinary["cell_accuracy"], diagnostic["locked"]["cell_accuracy"], abs_tol=1e-6) or
                    not math.isclose(ordinary["mean_geometry_error_cells"], diagnostic["geometry"]["mean_cells"], rel_tol=1e-5, abs_tol=1e-5) or
                    ordinary["known_layers"] != len(layers) or ordinary["exact_known_layers"] != sum(layer["exact"] for layer in layers)):
                raise ValueError("decomposition disagrees with ordinary metrics")
            if bool(capacity) != ("components" in diagnostic):
                raise ValueError("component diagnostics differ from checkpoint capacity")
            if capacity and any(diagnostic["components"][key] != value for key, value in ordinary["components"].items()):
                raise ValueError("component decomposition disagrees with ordinary metrics")
        for row in extension_rows:
            recipe = extension_recipes[row["sample_id"]]
            density, skin = recipe_strata(recipe)
            ordinary, diagnostic = row["ordinary_metrics"], row["diagnostics"]
            if (row["group_id"] != recipe["group_id"] or row["view"] != recipe["view"] or
                    row["skin"] != skin or row["density_recipe"] != density):
                raise ValueError("extension sample ancestry/recipe differs")
            layers = [value for value in diagnostic["layers"].values() if value["target_known"]]
            if (ordinary["known_cells"] != diagnostic["locked"]["known_cells"] or
                    ordinary["exact_locked_board"] != diagnostic["locked"]["exact_board"] or
                    not math.isclose(ordinary["cell_accuracy"], diagnostic["locked"]["cell_accuracy"], abs_tol=1e-6) or
                    not math.isclose(ordinary["mean_geometry_error_cells"], diagnostic["geometry"]["mean_cells"], rel_tol=1e-5, abs_tol=1e-5) or
                    ordinary["known_layers"] != len(layers) or ordinary["exact_known_layers"] != sum(layer["exact"] for layer in layers)):
                raise ValueError("extension decomposition disagrees with ordinary metrics")
            if bool(capacity) != ("components" in diagnostic):
                raise ValueError("extension component diagnostics differ from checkpoint capacity")
            if capacity and any(diagnostic["components"][key] != value for key, value in ordinary["components"].items()):
                raise ValueError("extension component decomposition disagrees with ordinary metrics")
    if capacity:
        for split, values in summarize_components(result["cases"]).items():
            totals[split]["components"] = values
    if result["totals"] != totals:
        raise ValueError("diagnostic aggregate differs")
    return architecture


def main():
    require_bounded_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    require_owned_run(args.name)
    run = ROOT/"runs"/args.name
    result = read(run/"result.json")
    require_owned_run(result["run"])
    source = ROOT/"runs"/result["run"]
    summary, manifest = read(source/"summary.json"), read(source/"checkpoint/manifest.json")
    source_job = read(ROOT/"runs"/(result["run"]+"-job.json"))
    validate_completion(summary, source_job, allow_partial=True)
    verify_source(source/"code", result["source_catalogue_sha256"])
    if digest(source/"checkpoint/weights.pt") != result["weights_sha256"]:
        raise ValueError("input weight bytes differ")
    job = read(ROOT/"runs"/(args.name+"-job.json"))
    if (job["supervisor_script_sha256"] != digest(ROOT/"scripts/run_bounded.py") or
            job["system_pressure_guard"] != "GetPerformanceInfo/0.2s"):
        raise ValueError("native supervisor source/guard differs")
    recipes = {}
    for split in ("training", "development"):
        path = source/(split+"-probe.jsonl")
        if path.stat().st_size > 1024**2:
            raise ValueError("probe ledger byte budget exceeded")
        recipes[split] = [json.loads(line) for line in path.read_text().splitlines()]
    architecture = validate_report(result, job, summary, manifest, recipes, name=args.name)
    if result["sources"] != {name: digest(ROOT/name) for name in REPORT_SOURCES}:
        raise ValueError("diagnostic execution source changed")
    relative = list(REPORT_SOURCES)+[Path(__file__).relative_to(ROOT).as_posix()]
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *relative], cwd=ROOT, check=True, capture_output=True)
    base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    evidence = ROOT/"docs/evidence"/args.name
    evidence.mkdir(exist_ok=False)
    selected = [(run/"result.json", "result.json"), (run/"progress.json", "progress.json"),
                (source/"summary.json", "input-summary.json"), (source/"code/source.json", "source-manifest.json"),
                (source/"checkpoint/manifest.json", "checkpoint-manifest.json"),
                (ROOT/"runs"/(result["run"]+"-job.json"), "input-run-job.json")]
    selected += [(source/(split+"-probe.jsonl"), split+"-probe.jsonl") for split in ("training", "development")]
    selected += [(ROOT/"runs"/(args.name+suffix), "native"+suffix) for suffix in NATIVE_SIDECARS]
    hashes = {}
    for path, name in selected:
        hashes[name] = digest(path)
        shutil.copyfile(path, evidence/name)
        if digest(evidence/name) != hashes[name]:
            raise ValueError("error evidence copy differs")
    scope = {"status": "passed", "source_base_commit": base, "core_architecture": architecture,
             **{key: result[key] for key in ("source_catalogue_sha256", "weights_sha256", "global_steps", "target_steps",
                 "run_status", "forwards", "component_capacity", "automatic_metrics_reproduced", "weights_unchanged")},
             "native_peak_job_commit_bytes": job["peak_job_commit_bytes"], "native_elapsed_seconds": job["elapsed_seconds"],
             "diagnostic_sources": result["sources"], "artifact_sha256": hashes,
             "curator_sha256": digest(Path(__file__)), "scope": result["scope"]}
    (evidence/"scope.json").write_text(json.dumps(scope, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "forwards": result["forwards"], "scope": str(evidence/"scope.json")}))


if __name__ == "__main__":
    main()
