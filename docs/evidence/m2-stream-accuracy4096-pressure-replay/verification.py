"""Point-in-time trajectory check after real ancestor pressure interruption."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT/"scripts"))
from probe_model import require_bounded_job
from source_snapshot import verify_source

require_bounded_job()
before = ROOT/"runs/m2-stream-accuracy4096-s1"
after = ROOT/"runs/m2-stream-accuracy4096-s2"
pointer = json.loads((before/"latest-training.json").read_text())
manifest = before/"training-snapshots"/pointer["generation"]/"manifest.json"
assert hashlib.sha256(manifest.read_bytes()).hexdigest() == pointer["manifest_sha256"]
snapshot = json.loads(manifest.read_text())
old_code = verify_source(before/"code", snapshot["contract"]["catalogue_sha256"])
assert verify_source(after/"code")["catalogue_sha256"] == old_code["catalogue_sha256"]
assert snapshot["step"] == 192 and snapshot["contract"]["steps"] == 4096

def rows(path, limit=None):
    assert path.stat().st_size <= 16*1024**2
    lines = path.read_text().splitlines()
    return [json.loads(line) for line in (lines if limit is None else lines[:limit])]

old_curve = rows(before/"curve.jsonl")
count = len(old_curve)
assert count == 229 and json.loads((after/"progress.json").read_text())["step"] >= count
new_curve = rows(after/"curve.jsonl", count)
without_time = lambda values: [{key: value for key, value in row.items() if key != "elapsed_seconds"} for row in values]
assert without_time(old_curve) == without_time(new_curve)
assert rows(before/"training.jsonl") == rows(after/"training.jsonl", count)
for name in ("curve.jsonl", "training.jsonl", "evaluations.jsonl"):
    cursor = snapshot["ledgers"][name]
    for run in (before, after):
        with (run/name).open("rb") as stream:
            prefix = stream.read(cursor["bytes"])
        assert hashlib.sha256(prefix).hexdigest() == cursor["sha256"]
job = json.loads((after.parent/(after.name+"-job.json")).read_text())
resume_index = job["command"].index("--resume-from")+1
assert Path(job["command"][resume_index]).resolve() == before
result = {"status": "passed", "banked_step": 192, "exact_recorded_prefix": count,
          "recomputed_after_bank": count-192, "training_data_records_equal": count,
          "loss_terms_gradient_norms_equal": count, "committed_ledger_prefixes_equal": 3,
          "source_catalogue_sha256": old_code["catalogue_sha256"], "target_steps": 4096,
          "target_complete": False, "old_final_model_tensor_comparison_available": False,
          "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "scope": "Actual pressure-bank192 restore recomputes37 later updates with exact recorded loss/terms/gradient norms/data. Elapsed time excluded. No old229 tensor artifact survived; no final-model tensor equality, completed target or quality claim."}
with (after/"pressure-replay229.json").open("x", encoding="utf-8") as output:
    json.dump(result, output, indent=2)
print(json.dumps(result), flush=True)
