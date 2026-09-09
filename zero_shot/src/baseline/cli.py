import argparse
import json

from .config import load_config


def main():
    parser = argparse.ArgumentParser(description="Zero-shot Qwen Spider test baselines")
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--devices", nargs="+", required=True,
                          help="One base-model replica per device, e.g. cuda:0 cuda:1")
    evaluate.add_argument("--limit", type=int, help="Deterministic test-only smoke subset")
    rescore = commands.add_parser(
        "rescore", help="Re-score an existing run without loading the model")
    rescore.add_argument("--config", required=True)
    rescore.add_argument("--output", required=True,
                         help="Existing output directory containing generation_shards")
    rescore.add_argument("--test-suite-database-dir", required=True,
                         help="Generated test-suite database root for this split")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "evaluate":
        from .evaluator import evaluate as run
        result = run(cfg, args.output, args.devices, args.limit)
    else:
        from .evaluator import rescore as run
        result = run(cfg, args.output, args.test_suite_database_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
