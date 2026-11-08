"""Time an actual restore of a committed accumulation boundary (bounded job)."""
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from probe_model import require_bounded_job

ceiling = require_bounded_job()
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from accumulation_checkpoint import restore_boundary
from tetris_vision.model import ModelConfig, TetrisStateNet

source, destination, output = (Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(),
                               Path(sys.argv[3]).resolve())
summary = json.loads((source/"summary.json").read_text())
contract = summary["accumulation"]["contract"]
model = TetrisStateNet(ModelConfig.pilot())
optimizer = torch.optim.AdamW(model.parameters(), lr=contract["learning_rate"], weight_decay=contract["weight_decay"],
                              betas=tuple(contract["betas"]), eps=contract["epsilon"])
destination.mkdir(parents=True, exist_ok=False)
started = time.monotonic()
state = restore_boundary(source, destination, model, optimizer, contract)
seconds = time.monotonic()-started
record = {"source_run": str(source), "seconds": seconds, "images_seen": state["images_seen"],
          "optimizer_updates": state["optimizer_updates"],
          "journal_bytes": (destination/"groups.jsonl").stat().st_size,
          "journal_sha256": hashlib.sha256((destination/"groups.jsonl").read_bytes()).hexdigest(),
          "source_pointer": state["source_pointer"], "ceiling_bytes": ceiling,
          "device": "cpu", "device_memory_bytes": 0}
output.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
print(json.dumps({key: record[key] for key in ("seconds", "images_seen", "optimizer_updates", "journal_sha256")}))
