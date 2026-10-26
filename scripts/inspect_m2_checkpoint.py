"""One guarded full-image M2 inspection call; never an accepted solver request."""
import argparse
import json
import os
from pathlib import Path

from probe_model import require_bounded_job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_bounded_job()
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    from PIL import Image
    from tetris_vision.runtime import ImageParser
    torch.set_num_threads(1)
    if args.output.exists():
        raise FileExistsError("preserve existing inspection output; choose a new path")
    model = ImageParser.from_checkpoint(args.checkpoint)
    with Image.open(args.image) as image:
        state = model.parse_image(image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(state.to_json())
    print(json.dumps({"status": "passed", "decision": state.decision, "model_id": state.model_id,
                      "output": str(args.output), "note": "M2 inspection only; queue/hold are unresolved."}))


if __name__ == "__main__":
    main()
