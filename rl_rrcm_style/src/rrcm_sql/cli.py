import argparse
import json
from pathlib import Path

from .config import load_config
from .data import prepare, prepare_official


def main():
    parser = argparse.ArgumentParser(description="Standalone adaptive Text-to-SQL SFT / GRPO / evaluation")
    commands = parser.add_subparsers(dest="command", required=True)
    split = commands.add_parser("prepare", help="Validate official split resources, or explicitly prepare a legacy internal split")
    split.add_argument("--config", help="Official split config")
    split.add_argument("--train-json", help="Legacy internal split input")
    split.add_argument("--output", required=True)
    split.add_argument("--validation-fraction", type=float, default=0.1)
    split.add_argument("--seed", type=int, default=42)
    for command in ("train", "sft", "evaluate"):
        sub = commands.add_parser(command)
        sub.add_argument("--config", required=True)
        sub.add_argument("--model", help="Override model.name_or_path (HF ID / local model / local PEFT adapter)")
        sub.add_argument("--output", help="Override training output dir; required for evaluation")
        if command == "train":
            sub.add_argument("--resume", help="Trusted complete checkpoint directory")
        if command == "evaluate":
            sub.add_argument("--split", choices=("dev", "test"), default="dev")
            sub.add_argument("--checkpoint")
            sub.add_argument("--data-json", help="Optional custom evaluation data; resource paths follow --split")
            sub.add_argument("--limit", type=int)
            sub.add_argument("--devices", nargs="+", help="One model replica per device, e.g. cuda:0 cuda:1")
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
        if args.output and args.command != "evaluate":
            cfg.train.output_dir = args.output
        cfg.validate()
        if args.command == "train":
            from .train import train
            if args.resume:
                original = load_config(Path(args.resume) / "run_config.json")
                # Preserve objective, reference policy and sampling policy across resume.
                from dataclasses import asdict
                old, new = asdict(original), asdict(cfg)
                for key in ("output_dir", "max_steps", "max_groups", "save_steps"):
                    old["train"].pop(key)
                    new["train"].pop(key)
                old.pop("tracking")
                new.pop("tracking")
                if old != new:
                    raise ValueError("Resume config changed; only output_dir/max_steps/max_groups/save_steps/tracking may differ")
            result = train(cfg, args.resume)
        elif args.command == "sft":
            from .train import sft
            result = sft(cfg)
        else:
            from .evaluate import evaluate
            if not args.output:
                parser.error("evaluate requires --output")
            result = evaluate(cfg, args.output, args.checkpoint, args.data_json, args.limit,
                              args.split, args.devices)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
