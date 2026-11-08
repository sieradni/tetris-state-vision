"""Clean-room audit of the committed segmented-arm evidence.

Reads only artifacts inside docs/evidence/accumulation-arm-r1/ (summaries,
supervisor receipts, journals, boundary manifests, profile artifacts and reload
receipts) plus the two documents that publish their numbers, and re-derives
every per-segment and per-arm figure with plain arithmetic. Nothing here imports
the trainer or the evidence builder; the only inputs are the committed bytes.

    python audit/audit_arm_evidence.py --seal   # derive, compare, write audit.json,
                                                # refresh the derived scope fields/hashes
    python audit/audit_arm_evidence.py          # same comparison, no writes; exit 1 on mismatch

A plain run rewrites audit.json with byte-identical content when nothing changed,
so the hashes recorded in scope.json stay valid.
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
SCOPE = HERE / "scope.json"
AUDIT = HERE / "audit" / "audit.json"
DOC = ROOT / "docs" / "BALANCED_ACCUMULATION.md"
PROFILE_DOC_SCOPE = ROOT / "docs" / "evidence" / "accumulation-profile-r1"
DOC_ARM_LABEL = {"balanced-eight": "balanced-eight-v1", "online interleaved": "online-interleaved-v1"}

mismatches = []
checks = {}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def check(name, condition, detail=""):
    ok = bool(condition)
    checks[name] = ok
    if not ok:
        mismatches.append(f"{name}: {detail}")
    return ok


def compare(name, derived, published):
    if isinstance(derived, float) or isinstance(published, float):
        if published is None or abs(float(derived) - float(published)) > 1e-9:
            mismatches.append(f"{name}: derived {derived!r} != published {published!r}")
            checks[name] = False
            return
    elif derived != published:
        mismatches.append(f"{name}: derived {derived!r} != published {published!r}")
        checks[name] = False
        return
    checks[name] = True


def arms():
    found = {}
    for path in sorted(HERE.glob("*-segment1-summary.json")):
        policy = path.name[: -len("-segment1-summary.json")]
        segments = sorted(HERE.glob(f"{policy}-segment*-summary.json"))
        found[policy] = [p.name[: -len("-summary.json")] for p in segments]
    return found


def measure(policy, runs):
    rows, inputs = [], []
    summaries, receipts, journals, manifests = [], [], [], []
    for index, run in enumerate(runs, start=1):
        summary_path = HERE / f"{run}-summary.json"
        receipt_path = HERE / f"{run}-job.json"
        journal_path = HERE / f"{run}-groups.jsonl"
        manifest_path = HERE / f"{run}-manifest.json"
        for path in (summary_path, receipt_path, journal_path, manifest_path):
            inputs.append(path)
        summary, receipt = load(summary_path), load(receipt_path)
        run_name = Path(summary["arguments"]["output"]).name
        journal = journal_path.read_bytes()
        manifest = load(manifest_path)
        lines = journal.splitlines()
        summaries.append((summary_path, summary))
        receipts.append((receipt_path, receipt))
        journals.append((journal_path, journal))
        manifests.append((manifest_path, manifest))

        before = summary["resumed_from_update"]
        after = summary["optimizer_updates"]
        ceiling, peak = receipt["memory_bytes"], receipt["peak_job_commit_bytes"]
        margin = ceiling - peak
        images_per_update = summary["images_per_update"]
        journal_images = sum(len(json.loads(line)["images"]) for line in lines)
        prior_images = (sum(len(json.loads(line)["images"])
                            for line in journals[0][1].splitlines()) if index > 1 else 0)
        boundary = summary["accumulation"]["latest_boundary"]
        row = {
            "segment": index,
            "run": run_name,
            "status": summary["status"],
            "job_status": receipt["status"],
            "job_exit_code": receipt["exit_code"],
            "optimizer_updates_before": before,
            "optimizer_updates_after": after,
            "updates_this_segment": after - before,
            "images_per_update": images_per_update,
            "images_this_segment": summary["accumulation"]["images_this_segment"],
            "images_seen": summary["images_seen"],
            "wall_seconds": summary["elapsed_seconds"],
            "job_wall_seconds": receipt["elapsed_seconds"],
            "supervisor_minus_child_seconds": round(receipt["elapsed_seconds"] - summary["elapsed_seconds"], 3),
            "segment_updates_per_second": round((after - before) / summary["elapsed_seconds"], 6),
            "cumulative_updates_per_second": round(after / summary["elapsed_seconds"], 6),
            "peak_job_commit_bytes": peak,
            "ceiling_bytes": ceiling,
            "ceiling_headroom_bytes": margin,
            "ceiling_headroom_percent": round(100.0 * margin / ceiling, 2),
            "device": "cpu" if summary.get("cpu_only") else "unknown",
            "device_memory_bytes": 0,
            "journal_lines": len(lines),
            "journal_bytes": len(journal),
            "journal_sha256": sha256(journal),
            "resume": summary.get("resume"),
            "resume_journal_prefix_bytes": (len(journals[0][1]) if summary.get("resume") else None),
            "peak_commit_delta_vs_segment1_bytes": None,
            "latest_boundary": boundary,
            "latest_boundary_manifest_sha256_verified": sha256(manifest_path.read_bytes())
            == boundary["manifest_sha256"]
            == summary["latest_training_snapshot"]["manifest_sha256"],
            "catalogue_sha256": summary["catalogue_sha256"],
        }

        # per-segment invariants, each re-derived from the artifact bytes
        compare(f"{policy}.seg{index}.journal_lines==updates_after", len(lines), after)
        compare(f"{policy}.seg{index}.journal_total_images", journal_images,
                summary["images_seen"])
        compare(f"{policy}.seg{index}.images_this_segment_after_prefix",
                summary["accumulation"]["images_this_segment"], journal_images - prior_images)
        compare(f"{policy}.seg{index}.images_per_update_accounting",
                summary["accumulation"]["images_this_segment"],
                (after - before) * images_per_update)
        compare(f"{policy}.seg{index}.images_seen", summary["images_seen"],
                (summaries[index - 2][1]["images_seen"] if index > 1 else 0)
                + summary["accumulation"]["images_this_segment"])
        compare(f"{policy}.seg{index}.manifest_step", manifest["step"], after)
        compare(f"{policy}.seg{index}.manifest_ledger_sha",
                manifest["ledgers"]["groups.jsonl"]["sha256"], row["journal_sha256"])
        compare(f"{policy}.seg{index}.manifest_ledger_bytes",
                manifest["ledgers"]["groups.jsonl"]["bytes"], len(journal))
        compare(f"{policy}.seg{index}.summary_journal_sha",
                summary["accumulation"]["journal_sha256"], row["journal_sha256"])
        compare(f"{policy}.seg{index}.summary_stream_ledger",
                summary.get("accumulation_journal_sha256"), row["journal_sha256"])
        compare(f"{policy}.seg{index}.recorded_rate_is_cumulative",
                round(after / summary["elapsed_seconds"], 6),
                round(summary["optimizer_updates_per_second"], 6))
        compare(f"{policy}.seg{index}.summary_ceiling", summary["job_ceiling_bytes"], ceiling)
        check(f"{policy}.seg{index}.peak_under_ceiling", peak < ceiling,
              f"peak {peak} vs ceiling {ceiling}")
        compare(f"{policy}.seg{index}.headroom", margin, ceiling - peak)
        compare(f"{policy}.seg{index}.exit_code_convention", receipt["exit_code"],
                0 if summary["status"] == "complete" else 2)
        check(f"{policy}.seg{index}.cpu_only", summary.get("cpu_only") is True and receipt["status"] in
              ("passed", "failed"), "cpu-only run expected")
        check(f"{policy}.seg{index}.boundary_manifest_sha",
              row["latest_boundary_manifest_sha256_verified"], "manifest digest mismatch")
        rows.append(row)

    first, last = rows[0], rows[-1]
    prefix = journals[0][1]
    final = journals[-1][1]
    prefix_sha = sha256(prefix)
    final_sha = sha256(final)
    resumed_prefix_sha = sha256(final[: len(prefix)])
    boundary1 = first["latest_boundary"]
    resume_record = last["resume"] or {}
    contract = manifests[0][1]["contract"]
    target_updates, target_images = contract["target_updates"], contract["target_images"]
    resume = {
        "journal_prefix_is_exact_prefix": final.startswith(prefix),
        "journal_prefix_bytes": len(prefix),
        "journal_prefix_sha256": prefix_sha,
        "final_journal_bytes": len(final),
        "final_journal_sha256": final_sha,
        "resumed_journal_prefix_sha256": resumed_prefix_sha,
        "resume_record_prefix_sha256": resume_record.get("journal_prefix_sha256"),
        "segment1_boundary_manifest_sha256": sha256(manifests[0][0].read_bytes()),
        "resume_pointer_matches_segment1_boundary": resume_record.get("pointer") == boundary1,
        "resumed_from_update_matches": last["optimizer_updates_before"]
        == first["optimizer_updates_after"]
        == resume_record.get("step"),
        "resumed_images_match": resume_record.get("images_seen") == first["images_seen"],
        "catalogues_match": first["catalogue_sha256"]
        == last["catalogue_sha256"]
        == load(summaries[0][0])["accumulation"]["contract"]["source_catalogue_sha256"],
        "target_updates": target_updates,
        "target_images": target_images,
        "final_updates": last["optimizer_updates_after"],
        "final_images_seen": last["images_seen"],
        "completed_budget": last["optimizer_updates_after"] == target_updates
        and last["images_seen"] == target_images
        and last["status"] == "complete",
        "final_status": last["status"],
    }
    check(f"{policy}.resume.prefix_bytes", resume["journal_prefix_is_exact_prefix"],
          f"segment1 journal {len(prefix)} B is not a prefix of the resumed journal")
    compare(f"{policy}.resume.prefix_sha_recomputed", resumed_prefix_sha, prefix_sha)
    compare(f"{policy}.resume.record_prefix_sha", resume_record.get("journal_prefix_sha256"), prefix_sha)
    compare(f"{policy}.resume.pointer", resume_record.get("pointer"), boundary1)
    compare(f"{policy}.resume.cursor", last["optimizer_updates_before"], first["optimizer_updates_after"])
    compare(f"{policy}.resume.cursor_recorded", resume_record.get("step"),
            first["optimizer_updates_after"])
    check(f"{policy}.resume.final_journal_sha_matches_summary", final_sha
          == load(summaries[-1][0])["accumulation"]["journal_sha256"] ==
          manifests[-1][1]["ledgers"]["groups.jsonl"]["sha256"], "final journal digest mismatch")
    check(f"{policy}.resume.completed_budget", resume["completed_budget"],
          f"{last['optimizer_updates_after']}/{target_updates} updates, {last['images_seen']} images")
    for row in rows:
        row["peak_commit_delta_vs_segment1_bytes"] = (
            None if row["segment"] == 1 else
            row["peak_job_commit_bytes"] - first["peak_job_commit_bytes"])

    reload_run = f"reload-{policy}"
    reload_path, reload_receipt_path = HERE / f"{reload_run}.json", HERE / f"{reload_run}-job.json"
    reload, reload_receipt = load(reload_path), load(reload_receipt_path)
    inputs.extend([reload_path, reload_receipt_path])
    reload_block = {
        "source_run": reload["source_run"],
        "seconds": reload["seconds"],
        "images_seen": reload["images_seen"],
        "optimizer_updates": reload["optimizer_updates"],
        "journal_bytes": reload["journal_bytes"],
        "journal_bytes_recomputed": len(prefix),
        "journal_sha256": reload["journal_sha256"],
        "source_pointer": reload["source_pointer"],
        "pointer_matches_segment1_boundary": reload["source_pointer"] == boundary1,
        "ceiling_bytes": reload["ceiling_bytes"],
        "receipt_ceiling_bytes": reload_receipt["memory_bytes"],
        "receipt_peak_job_commit_bytes": reload_receipt["peak_job_commit_bytes"],
        "receipt_headroom_bytes": reload_receipt["memory_bytes"]
        - reload_receipt["peak_job_commit_bytes"],
        "receipt_status": reload_receipt["status"],
        "receipt_exit_code": reload_receipt["exit_code"],
        "device": reload["device"],
        "device_memory_bytes": reload["device_memory_bytes"],
    }
    compare(f"{policy}.reload.journal_bytes", len(prefix), reload["journal_bytes"])
    compare(f"{policy}.reload.journal_sha", reload["journal_sha256"], prefix_sha)
    compare(f"{policy}.reload.updates", reload["optimizer_updates"], first["optimizer_updates_after"])
    compare(f"{policy}.reload.images", reload["images_seen"], first["images_seen"])
    compare(f"{policy}.reload.pointer", reload["source_pointer"], boundary1)
    compare(f"{policy}.reload.ceiling", reload["ceiling_bytes"], reload_receipt["memory_bytes"])
    check(f"{policy}.reload.positive", reload["seconds"] > 0, f"{reload['seconds']} s")
    check(f"{policy}.reload.receipt_passed",
          reload_receipt["status"] == "passed" and reload_receipt["exit_code"] == 0,
          f"{reload_receipt['status']}/{reload_receipt['exit_code']}")
    check(f"{policy}.reload.peak_under_ceiling",
          reload_receipt["peak_job_commit_bytes"] < reload_receipt["memory_bytes"],
          "reload peak at or above ceiling")

    # admission: the profile receipt for this exact policy, at the segment ceiling
    profile_run = f"{policy}-profile"
    profile_path, profile_receipt_path = HERE / f"{profile_run}-summary.json", HERE / f"{profile_run}-job.json"
    profile, profile_receipt = load(profile_path), load(profile_receipt_path)
    inputs.extend([profile_path, profile_receipt_path])
    profile_updates = profile["optimizer_updates"]
    admission = {
        "result": "accepted",
        "profile": Path(profile["arguments"]["output"]).name,
        "profile_work_items": profile_receipt["work_items"],
        "required_profile_work_items": 2 * profile["images_per_update"],
        "profile_images_per_update": profile["images_per_update"],
        "profile_steps": profile["steps"],
        "profile_updates": profile_updates,
        "profile_elapsed_seconds": profile["elapsed_seconds"],
        "profile_receipt_elapsed_seconds": profile_receipt["elapsed_seconds"],
        "measured_peak_job_commit_bytes": profile_receipt["peak_job_commit_bytes"],
        "ceiling_bytes": profile_receipt["memory_bytes"],
        "headroom_bytes": profile_receipt["memory_bytes"] - profile_receipt["peak_job_commit_bytes"],
        "headroom_percent": round(
            100.0 * (profile_receipt["memory_bytes"] - profile_receipt["peak_job_commit_bytes"])
            / profile_receipt["memory_bytes"], 2),
        "measured_optimizer_updates_per_second": round(profile["optimizer_updates_per_second"], 6),
        "profile_updates_per_second_from_counts": round(
            profile_updates / profile["elapsed_seconds"], 6),
        "profile_rate_is_startup_dominated": profile_updates / profile["elapsed_seconds"]
        < 0.2 * max(row["segment_updates_per_second"] for row in rows),
    }
    compare(f"{policy}.admission.segment_names_this_profile",
            Path(load(summaries[0][0])["arguments"]["admission"] or "").name, admission["profile"])
    compare(f"{policy}.admission.work_items", profile_receipt["work_items"],
            2 * profile["images_per_update"])
    compare(f"{policy}.admission.steps", profile["steps"], 2)
    compare(f"{policy}.admission.updates", profile_updates, 2)
    compare(f"{policy}.admission.rate", round(profile_updates / profile["elapsed_seconds"], 6),
            round(profile["optimizer_updates_per_second"], 6))
    compare(f"{policy}.admission.ceiling", profile_receipt["memory_bytes"], rows[0]["ceiling_bytes"])
    compare(f"{policy}.admission.policy", profile["accumulation_policy"], policy)
    check(f"{policy}.admission.profile_complete", profile["status"] == "complete"
          and profile_receipt["status"] == "passed" and profile_receipt["exit_code"] == 0,
          "profile did not pass")
    check(f"{policy}.admission.profile_peak_under_ceiling",
          profile_receipt["peak_job_commit_bytes"] < profile_receipt["memory_bytes"],
          "profile peak at or above ceiling")
    check(f"{policy}.admission.profile_square", profile.get("profile_square") is True,
          "admission profile is not a full-square profile")
    check(f"{policy}.admission.catalogue", profile["catalogue_sha256"] == rows[0]["catalogue_sha256"],
          "profile and segment catalogues differ")
    check(f"{policy}.admission.positive_rate", profile["optimizer_updates_per_second"] > 0,
          "profile rate not positive")

    return ({"segments": rows, "resume": resume, "reload_measurement": reload_block},
            admission, inputs)


def doc_numbers(derived):
    """Every numeric cell of the published segment table, as the doc prints it."""
    text = DOC.read_text(encoding="utf-8")
    table = {}
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 8 and cells[0] in DOC_ARM_LABEL and cells[1].isdigit():
            table[(DOC_ARM_LABEL[cells[0]], int(cells[1]))] = cells
    problems = []
    for policy, data in derived.items():
        for row in data["segments"]:
            cells = table.get((policy, row["segment"]))
            if cells is None:
                problems.append(f"table row missing for {policy} segment {row['segment']}")
                continue
            expected = [
                str(row["updates_this_segment"]),
                str(row["images_this_segment"]),
                f"{row['wall_seconds']:.1f} s",
                f"{row['segment_updates_per_second']:.3f}",
                f"{row['peak_job_commit_bytes']:,} B",
                "-" if row["peak_commit_delta_vs_segment1_bytes"] is None
                else f"+{row['peak_commit_delta_vs_segment1_bytes']:,} B",
            ]
            for column, want in zip(cells[2:], expected):
                if column != want:
                    problems.append(
                        f"table {policy} seg{row['segment']}: printed {column!r}, derived {want!r}")
    return text, table, problems


def profile_doc_numbers():
    """The earlier two-update profile table, which cites accumulation-profile-r1."""
    text = DOC.read_text(encoding="utf-8")
    problems, derived = [], {}
    for label, name in (("1 (online interleaved)", "online-r2"), ("8 (balanced eight)", "balanced-r1")):
        summary = load(PROFILE_DOC_SCOPE / f"{name}-summary.json")
        receipt = load(PROFILE_DOC_SCOPE / f"{name}-job.json")
        warm = summary["warm_optimizer_backward"]
        row = {
            "wall_seconds": summary["elapsed_seconds"],
            "updates_per_second": summary["optimizer_updates_per_second"],
            "peak_job_commit_bytes": receipt["peak_job_commit_bytes"],
            "reload_seconds": warm["reload_seconds"],
            "reload_commit_delta_bytes": warm["reload_commit_delta_bytes"],
        }
        derived[label] = row
        cells = None
        for line in text.splitlines():
            candidate = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(candidate) == 6 and candidate[0] == label:
                cells = candidate
                break
        if cells is None:
            problems.append(f"profile table row missing for {label}")
            continue
        expected = [f"{row['wall_seconds']:.2f} s", f"{row['updates_per_second']:.3f}",
                    f"{row['peak_job_commit_bytes']:,} B", f"{row['reload_seconds']:.3f} s",
                    f"{row['reload_commit_delta_bytes']} B"]
        for column, want in zip(cells[1:], expected):
            if column != want:
                problems.append(f"profile table {label}: printed {column!r}, derived {want!r}")
    return derived, problems


def prose_checks(text, derived):
    problems = []
    balanced, online = derived["balanced-eight-v1"], derived["online-interleaved-v1"]
    reload_b = balanced["reload_measurement"]["seconds"]
    reload_o = online["reload_measurement"]["seconds"]
    expectations = {
        "reload_eight": f"{reload_b:.3f} s (eight-image arm)",
        "reload_online": f"{reload_o:.3f} s (online arm)",
        "updates_pair": f"({balanced['segments'][0]['optimizer_updates_after']} and "
                        f"{online['segments'][0]['optimizer_updates_after']} updates)",
        "final_budget": f"{balanced['resume']['final_updates']}/{balanced['resume']['target_updates']} and "
                         f"{online['resume']['final_updates']:,}/{online['resume']['target_updates']:,}",
    }
    margins = sorted((row["ceiling_headroom_bytes"], row["ceiling_headroom_percent"], row["run"])
                     for data in derived.values() for row in data["segments"])
    expectations["tightest_margin"] = f"{margins[0][0]:,} B ({margins[0][1]:.2f} %)"
    expectations["widest_margin"] = f"{margins[-1][0]:,} B ({margins[-1][1]:.2f} %)"
    for name, want in expectations.items():
        if want not in text:
            problems.append(f"prose {name}: {want!r} not printed in the document")
    return expectations, problems


def main(write):
    scope = load(SCOPE)
    derived, admissions, inputs = {}, {}, []
    for policy, runs in arms().items():
        data, admission, used = measure(policy, runs)
        derived[policy], admissions[policy] = data, admission
        inputs.extend(used)

    # compare every published derived field against the committed scope
    for policy, data in derived.items():
        published = scope["measurements"][policy]
        for key in ("journal_prefix_is_exact_prefix", "journal_prefix_bytes", "journal_prefix_sha256",
                    "final_journal_bytes", "final_journal_sha256", "resumed_journal_prefix_sha256",
                    "resume_record_prefix_sha256", "segment1_boundary_manifest_sha256",
                    "resume_pointer_matches_segment1_boundary", "resumed_from_update_matches",
                    "resumed_images_match", "catalogues_match", "target_updates", "target_images",
                    "final_updates", "final_images_seen", "completed_budget", "final_status"):
            if key in published.get("resume", {}):
                compare(f"scope.{policy}.resume.{key}", data["resume"][key],
                        published["resume"][key])
        for row, target in zip(data["segments"], published["segments"]):
            for key, value in row.items():
                if key == "latest_boundary":
                    compare(f"scope.{policy}.seg{row['segment']}.latest_boundary", value,
                            target.get(key))
                elif key in target:
                    compare(f"scope.{policy}.seg{row['segment']}.{key}", value, target[key])
            if "resume_delta_bytes" in target and row["segment"] > 1:
                compare(f"scope.{policy}.seg{row['segment']}.resume_delta_bytes",
                        row["peak_commit_delta_vs_segment1_bytes"], target["resume_delta_bytes"])
        published_reload = published.get("reload_measurement") or {}
        for key, value in data["reload_measurement"].items():
            if key in published_reload:
                compare(f"scope.{policy}.reload.{key}", value, published_reload[key])
        for key, value in admissions[policy].items():
            if key in scope["admissions"].get(policy, {}):
                compare(f"scope.{policy}.admission.{key}", value, scope["admissions"][policy][key])

    text, table, table_problems = doc_numbers(derived)
    for problem in table_problems:
        mismatches.append(problem)
        checks[f"doc.table.{problem}"] = False
    if not table_problems:
        checks["doc.table"] = True
    profile_table, profile_problems = profile_doc_numbers()
    for problem in profile_problems:
        mismatches.append(problem)
        checks[f"doc.profile_table.{problem}"] = False
    if not profile_problems:
        checks["doc.profile_table"] = True
    prose, prose_problems = prose_checks(text, derived)
    for problem in prose_problems:
        mismatches.append(problem)
        checks[f"doc.prose.{problem}"] = False
    if not prose_problems:
        checks["doc.prose"] = True

    payload = {
        "scope": "docs/evidence/accumulation-arm-r1",
        "instrument": "audit/audit_arm_evidence.py",
        "method": ("re-derived from the committed summaries, supervisor receipts, journals, "
                   "boundary manifests, profile artifacts and reload receipts with plain "
                   "arithmetic; the trainer and the evidence builder are not imported"),
        "inputs_sha256": {path.relative_to(HERE).as_posix(): sha256(path.read_bytes())
                          for path in sorted(inputs)},
        "derived": derived,
        "admissions": admissions,
        "doc_expectations": prose,
        "profile_table": profile_table,
        "checks": checks,
        "checks_total": len(checks),
        "checks_failed": sorted(name for name, ok in checks.items() if not ok),
        "mismatches": mismatches,
        "result": "reproduced" if not mismatches else "mismatch",
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if write:
        scope["ceiling_guard"] = {
            "limit": "Windows Job object commit limit enforced by scripts/run_bounded.py",
            "ceiling_bytes": derived[next(iter(derived))]["segments"][0]["ceiling_bytes"],
            "required_margin": ("the job is killed when aggregate commit reaches the limit, so a "
                                "passing run needs peak_job_commit_bytes strictly below "
                                "memory_bytes; the supervisor applies no additional fractional "
                                "margin"),
            "tightest_observed_headroom_bytes": min(
                row["ceiling_headroom_bytes"] for data in derived.values()
                for row in data["segments"]),
            "tightest_observed_headroom_percent": min(
                row["ceiling_headroom_percent"] for data in derived.values()
                for row in data["segments"]),
        }
        scope["wall_time_definition"] = (
            "wall_seconds is the trainer's in-child elapsed_seconds; job_wall_seconds is the "
            "supervisor receipt elapsed_seconds, which also covers process start and teardown. "
            "Published wall times are wall_seconds.")
        scope["resume_delta_definition"] = (
            "peak_commit_delta_vs_segment1_bytes is the difference between this segment's "
            "supervised peak commit and segment 1's; it is a between-run peak difference, not "
            "the memory cost of restoring the boundary, which is measured separately in "
            "reload_measurement.")
        scope["admissions"] = admissions
        scope["measurements"] = derived
        scope["audit"] = {
            "instrument": "audit/audit_arm_evidence.py",
            "instrument_sha256": sha256(Path(__file__).read_bytes()),
            "result": "audit/audit.json",
            "result_sha256": sha256(AUDIT.read_bytes()),
            "checks": len(checks),
        }
        scope["artifact_sha256"] = {
            path.relative_to(HERE).as_posix(): sha256(path.read_bytes())
            for path in sorted(HERE.rglob("*"))
            if path.is_file() and path.name != "scope.json"
        }
        SCOPE.write_text(json.dumps(scope, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not write:
        # checked after the byte-identical rewrite, so a seal and a plain run agree on the payload
        bad = [name for name, digest in scope.get("artifact_sha256", {}).items()
               if sha256((HERE / name).read_bytes()) != digest]
        if bad:
            mismatches.append(f"scope artifact hash mismatch for {bad[:3]}")
        audit_block = scope.get("audit", {})
        if audit_block.get("instrument_sha256") != sha256(Path(__file__).read_bytes()):
            mismatches.append("scope.audit.instrument_sha256 does not describe this instrument")
        if audit_block.get("result_sha256") != sha256(AUDIT.read_bytes()):
            mismatches.append("scope.audit.result_sha256 does not describe audit/audit.json")

    print(json.dumps({"result": "reproduced" if not mismatches else "mismatch",
                      "checks": len(checks),
                      "failed": sorted(name for name, ok in checks.items() if not ok),
                      "mismatches": mismatches[:20]}, indent=2))
    return 0 if not mismatches else 1


if __name__ == "__main__":
    sys.exit(main("--seal" in sys.argv[1:]))
