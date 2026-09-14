import argparse
import json
from pathlib import Path

from .config import load_config
from .data import prepare, prepare_official


def main():
    parser = argparse.ArgumentParser(description="Free-policy Text-to-SQL GRPO training")
    commands = parser.add_subparsers(dest="command", required=True)
    split = commands.add_parser("prepare", help="Validate official split resources, or explicitly prepare a legacy internal split")
    split.add_argument("--config", help="Official split config")
    split.add_argument("--train-json", help="Legacy internal split input")
    split.add_argument("--output", required=True)
    split.add_argument("--validation-fraction", type=float, default=0.1)
    split.add_argument("--seed", type=int, default=42)
    train = commands.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--model", help="Override model.name_or_path")
    train.add_argument("--output", help="Override training output directory")
    train.add_argument("--resume", help="Trusted complete checkpoint directory")
    args = parser.parse_args()
    if args.command == "prepare":
        if args.config:
            if args.train_json:
                parser.error("Use either --config or legacy --train-json")
            cfg = load_config(args.config)
            if cfg.data.split_mode != "official":
                parser.error("prepare --config requires data.split_mode=official")
            result = prepare_official(cfg, args.output)
        else:
            if not args.train_json:
                parser.error("prepare requires --config or legacy --train-json")
            result = prepare(args.train_json, args.output, args.validation_fraction, args.seed)
    else:
        cfg = load_config(args.config)
        if args.model:
            cfg.model.name_or_path = args.model
        if args.output:
            cfg.train.output_dir = args.output
        cfg.validate()
        from .train import train
        if args.resume:
            original = load_config(Path(args.resume) / "run_config.json")
                # Preserve objective, reference policy and sampling policy across resume.
            from dataclasses import asdict
            old, new = asdict(original), asdict(cfg)
            for key in ("output_dir", "max_steps", "max_groups", "save_steps"):
                old["train"].pop(key)
                new["train"].pop(key)
                # Validation parallelism does not change the training objective or sampling policy.
            old["runtime"].pop("validation_devices", None)
            new["runtime"].pop("validation_devices", None)
            old.pop("tracking")
            new.pop("tracking")
            if old != new:
                raise ValueError("Resume config changed; only output_dir/max_steps/max_groups/save_steps/tracking may differ")
        result = train(cfg, args.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
