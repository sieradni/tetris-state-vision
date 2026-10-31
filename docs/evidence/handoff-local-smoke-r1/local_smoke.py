"""Fixed three-model local smoke of the delivery verifier; no training."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from probe_model import require_bounded_job

require_bounded_job()
for name in ("m2-stream-allclass1024-s3", "m2-stream-interleave1024-s3", "m2-stream-accuracy4096-s10"):
    subprocess.run([sys.executable, str(ROOT / "scripts/verify_handoff_checkout.py"), "--smoke-model", name], cwd=ROOT, check=True)
