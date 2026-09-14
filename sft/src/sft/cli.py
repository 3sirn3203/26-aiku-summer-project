import argparse
import json

from .config import load_config


def main():
    parser = argparse.ArgumentParser(description="Config-driven answer-only Text-to-SQL SFT")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check-config", "train"):
        command = sub.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument(
            "--output-dir",
            help="Override training.output_dir from the JSON config",
        )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.output_dir:
        cfg.training.output_dir = args.output_dir
    if args.command == "check-config":
        result = {"valid": True}
    else:
        from .train import train
        result = train(cfg)
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
