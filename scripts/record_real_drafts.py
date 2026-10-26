"""Curate aggregate unlabeled-corpus evidence, keeping screenshots private."""
import json
from pathlib import Path
import shutil
import subprocess

from corpus_tools import Limits, audit, canonical_digest, digest_file
from probe_model import require_bounded_job

ROOT = Path(__file__).resolve().parents[1]


def main():
    require_bounded_job()
    manifest = ROOT/"data/real-pilot-drafts/manifest.jsonl"
    profile_root = ROOT/"runs/real-drafts-overview"
    profile = json.loads((profile_root/"profile.json").read_text())
    jobs = {name: json.loads((ROOT/f"runs/real-drafts-{name}.json").read_text())
            for name in ("import", "audit", "profile")}
    if any(job["status"] != "passed" for job in jobs.values()):
        raise ValueError("all corpus jobs must have completed")
    result = audit(manifest, Limits(records=200, asset_bytes=128*1024**2))
    if result.errors or result.eligible or len(result.entries) != 166 or profile["images"] != 166:
        raise ValueError("expected the actual 166-image unadmitted draft corpus")
    for row in result.entries:
        if row["review"]["status"] != "draft" or row["annotation"] is not None or any(row["permission"][key] for key in ("training", "evaluation")):
            raise ValueError("draft corpus has acquired labels or permission")
    if profile["manifest_sha256"] != digest_file(manifest) or profile["record_identity_sha256"] != canonical_digest(result.entries):
        raise ValueError("profiled corpus changed")
    if profile["script_sha256"] != digest_file(ROOT/"scripts/profile_corpus.py"):
        raise ValueError("profile source changed")
    for page in profile["contact_pages"]:
        if digest_file(profile_root/page["path"]) != page["sha256"]:
            raise ValueError("contact page changed")
    evidence = ROOT/"docs/evidence/real-drafts"
    evidence.mkdir(exist_ok=False)
    for name in jobs:
        for suffix in (".json", ".log", ".stderr.log"):
            source = ROOT/f"runs/real-drafts-{name}{suffix}"
            shutil.copyfile(source, evidence/source.name)
    scope = {"status": "passed", "images": 166, "drafts": 166, "eligible_for_training_or_scoring": 0,
             "encoded_asset_bytes": result.bytes_admitted, "exact_duplicate_warnings": len(result.warnings),
             "manifest_sha256": profile["manifest_sha256"], "record_identity_sha256": profile["record_identity_sha256"],
             "contact_pages": profile["contact_pages"], "dimensions": profile["dimensions"],
             "profile_script_sha256": profile["script_sha256"], "curator_sha256": digest_file(Path(__file__)),
             "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
             "local_manifest": "data/real-pilot-drafts/manifest.jsonl", "local_overview": "runs/real-drafts-overview/",
             "user_intent": "Use the existing raw_screenshots folder; correct board answers are uncertain.",
             "purpose": "Unlabeled layout/appearance/crop stress-set discovery and later reviewed annotation.",
             "not_claimed": "Board truth, human labels, training/evaluation permission, model performance or independent sessions.",
             "privacy": "Images, thumbnail pages and per-frame names remain in ignored local data/runs; only aggregate receipts are curated."}
    (evidence/"scope.json").write_text(json.dumps(scope, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "drafts": 166, "eligible": 0, "contact_pages": len(profile["contact_pages"])}))


if __name__ == "__main__":
    main()
