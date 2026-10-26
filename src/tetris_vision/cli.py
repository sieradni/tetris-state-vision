"""Single finite entry point for state validation, synthetic data and evaluation."""
import argparse
import hashlib
import json
from pathlib import Path

from .state import VisionState, check


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tetris-vision")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate a rich state JSON")
    validate.add_argument("state", type=Path)
    demo = commands.add_parser("generate", help="generate 1..64 five-image counterfactual groups")
    demo.add_argument("--output", type=Path, required=True)
    demo.add_argument("--count", type=int, default=3)
    demo.add_argument("--seed", type=int, default=20261025)
    score = commands.add_parser("evaluate", help="score predictions against a manifest; missing predictions fail")
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "validate":
        state = VisionState.from_json(args.state.read_text(encoding="utf-8"))
        print(json.dumps({"valid": True, "schema_version": state.schema_version, "decision": state.decision}))
    elif args.command == "generate":
        from .dataset import generate
        print(json.dumps(generate(args.output, count=args.count, seed=args.seed)))
    else:
        from .evaluate import Example, evaluate
        examples = []
        root = args.manifest.resolve().parent
        for line in args.manifest.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            state_path = (root / record["state"]).resolve()
            check(state_path.is_relative_to(root), "state path escapes manifest root")
            if "state_sha256" in record:
                check(hashlib.sha256(state_path.read_bytes()).hexdigest() == record["state_sha256"],
                      "state checksum mismatch")
            target = VisionState.from_json(state_path.read_text(encoding="utf-8"))
            examples.append(Example(record["sample_id"], record["group_id"], record["family"], target))
        predictions = {}
        for line in args.predictions.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            check(record["sample_id"] not in predictions, "duplicate prediction id")
            predictions[record["sample_id"]] = VisionState.from_dict(record["state"])
        print(json.dumps(evaluate(examples, predictions), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
