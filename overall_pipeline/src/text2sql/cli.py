from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

from text2sql.config import ConfigError, load_config
from text2sql.single_turn.smoke_runner import run_smoke
from text2sql.core.spider import SpiderDataError, SpiderDataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spider-smoke",
        description=(
            "Spider 1.0 single-turn baseline and planner-coder-verifier "
            "Text-to-SQL evaluation pipelines"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-data", help="validate Spider files and DBs")
    validate.add_argument("--config", type=Path, required=True)

    validate_official = subparsers.add_parser(
        "validate-official",
        help="validate the pinned Spider evaluator and generated test-suite DBs",
    )
    validate_official.add_argument("--config", type=Path, required=True)
    validate_official.add_argument(
        "--all-examples",
        action="store_true",
        help="validate generated test-suite DBs for every configured split example",
    )

    doctor = subparsers.add_parser(
        "doctor", help="inspect server torch/CUDA compatibility without loading a model"
    )
    doctor.add_argument("--config", type=Path, required=True)

    smoke = subparsers.add_parser("smoke", help="run the fixed end-to-end smoke suite")
    smoke.add_argument("--config", type=Path, required=True)
    smoke.add_argument("--backend", choices=("mock", "hf"), default="mock")
    smoke.add_argument("--device", help="override the configured model device")
    smoke.add_argument("--output-dir", type=Path, help="override output root")
    smoke.add_argument("--run-name", help="stable run directory name")
    smoke.add_argument(
        "--allow-model-download",
        action="store_true",
        help="allow Hugging Face to download uncached model files (hf backend only)",
    )

    evaluate = subparsers.add_parser(
        "evaluate",
        help="run zero-shot inference and evaluation on the full configured split",
    )
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument(
        "--backend", choices=("mock", "hf", "peft"), default="hf"
    )
    evaluate.add_argument(
        "--adapter-dir",
        type=Path,
        help="local PEFT adapter directory; required with --backend peft",
    )
    evaluate.add_argument(
        "--selection",
        choices=("all", "smoke"),
        default="all",
        help="evaluate the full split or exercise the distributed path on fixed smoke samples",
    )
    evaluate.add_argument(
        "--gpus",
        required=True,
        help=(
            "comma-separated physical GPU indices; each worker sees its GPU as cuda:0"
        ),
    )
    evaluate.add_argument("--output-dir", type=Path, help="override evaluation output root")
    naming = evaluate.add_mutually_exclusive_group()
    naming.add_argument("--run-name", help="stable new run directory name")
    naming.add_argument(
        "--resume-run",
        help="resume an existing run name under the evaluation output root",
    )
    evaluate.add_argument(
        "--allow-model-download",
        action="store_true",
        help="allow base-model downloads in every GPU worker (hf/peft only)",
    )
    evaluate.add_argument(
        "--no-progress",
        action="store_true",
        help="disable parent-side progress output on stderr",
    )
    evaluate.add_argument(
        "--progress-interval-seconds",
        type=float,
        default=5.0,
        help="periodic non-TTY progress log interval (default: 5 seconds)",
    )

    agent_doctor = subparsers.add_parser(
        "agent-doctor",
        help="inspect every configured planner/coder/verifier GPU without loading models",
    )
    agent_doctor.add_argument("--config", type=Path, required=True)
    agent_doctor.add_argument(
        "--planner-gpus", help="override planner physical GPU pool (comma-separated)"
    )
    agent_doctor.add_argument(
        "--coder-gpus", help="override coder physical GPU pool (comma-separated)"
    )
    agent_doctor.add_argument(
        "--verifier-gpus", help="override verifier physical GPU pool (comma-separated)"
    )

    agent_evaluate = subparsers.add_parser(
        "agent-evaluate",
        help="run the planner-coder-verifier workflow on smoke samples or full dev",
    )
    agent_evaluate.add_argument("--config", type=Path, required=True)
    agent_evaluate.add_argument("--backend", choices=("mock", "hf"), default="hf")
    agent_evaluate.add_argument(
        "--selection", choices=("smoke", "all"), default="all"
    )
    agent_evaluate.add_argument(
        "--planner-gpus", help="override planner physical GPU pool (comma-separated)"
    )
    agent_evaluate.add_argument(
        "--coder-gpus", help="override coder physical GPU pool (comma-separated)"
    )
    agent_evaluate.add_argument(
        "--verifier-gpus", help="override verifier physical GPU pool (comma-separated)"
    )
    agent_evaluate.add_argument(
        "--output-dir", type=Path, help="override agent-evaluation output root"
    )
    agent_naming = agent_evaluate.add_mutually_exclusive_group()
    agent_naming.add_argument("--run-name", help="stable new run directory name")
    agent_naming.add_argument(
        "--resume-run", help="resume a checkpointed agent-evaluation run"
    )
    agent_evaluate.add_argument(
        "--allow-model-download",
        action="store_true",
        help="allow Hugging Face downloads in role workers (hf only)",
    )
    agent_evaluate.add_argument(
        "--no-progress", action="store_true", help="disable progress output on stderr"
    )
    agent_evaluate.add_argument(
        "--progress-interval-seconds",
        type=float,
        default=5.0,
        help="periodic non-TTY progress log interval (default: 5 seconds)",
    )

    agent_doctor_worker = subparsers.add_parser(
        "_agent-doctor-worker", help="internal isolated GPU doctor worker"
    )
    agent_doctor_worker.add_argument("--config", type=Path, required=True)
    agent_doctor_worker.add_argument(
        "--role", choices=("planner", "coder", "verifier"), required=True
    )
    agent_doctor_worker.add_argument("--physical-gpu", type=int, required=True)

    worker = subparsers.add_parser(
        "_generation-worker", help="internal worker used by the evaluate command"
    )
    worker.add_argument("--config", type=Path, required=True)
    worker.add_argument("--assignment", type=Path, required=True)
    worker.add_argument("--shard", type=Path, required=True)
    worker.add_argument("--status", type=Path, required=True)
    worker.add_argument(
        "--backend", choices=("mock", "hf", "peft", "two_turn"), required=True
    )
    worker.add_argument("--allow-model-download", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command in {
            "agent-doctor",
            "agent-evaluate",
            "_agent-doctor-worker",
        }:
            from text2sql.multi_turn_agent.config import (
                AgentGpuPoolsConfig,
                load_agent_config,
            )
            from text2sql.single_turn.distributed import parse_gpu_ids

            agent_config = load_agent_config(args.config)

            def selected_pool(value, configured):
                return parse_gpu_ids(value) if value is not None else tuple(configured)

            if args.command == "_agent-doctor-worker":
                from text2sql.multi_turn_agent.doctor import inspect_role_worker

                report = inspect_role_worker(
                    agent_config.source_path, args.role, args.physical_gpu
                )
                print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
                return 0 if report["ok"] else 1

            planner_gpus = selected_pool(
                args.planner_gpus, agent_config.gpu_pools.planner
            )
            coder_gpus = selected_pool(args.coder_gpus, agent_config.gpu_pools.coder)
            verifier_gpus = selected_pool(
                args.verifier_gpus, agent_config.gpu_pools.verifier
            )
            combined = planner_gpus + coder_gpus + verifier_gpus
            if len(set(combined)) != len(combined):
                raise ConfigError("agent role GPU pools must not overlap")
            pools = AgentGpuPoolsConfig(
                planner=planner_gpus,
                coder=coder_gpus,
                verifier=verifier_gpus,
            )
            agent_config = replace(agent_config, gpu_pools=pools)

            if args.command == "agent-doctor":
                from text2sql.multi_turn_agent.doctor import run_agent_doctor

                report = run_agent_doctor(
                    agent_config,
                    {
                        "planner": planner_gpus,
                        "coder": coder_gpus,
                        "verifier": verifier_gpus,
                    },
                )
                print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
                return 0 if report["ok"] else 1

            if args.allow_model_download and args.backend != "hf":
                raise ConfigError(
                    "--allow-model-download is valid only with --backend hf"
                )
            if (
                not math.isfinite(args.progress_interval_seconds)
                or args.progress_interval_seconds <= 0
            ):
                raise ConfigError(
                    "--progress-interval-seconds must be positive and finite"
                )
            if args.output_dir is not None:
                agent_config = replace(
                    agent_config,
                    output=replace(
                        agent_config.output, directory=args.output_dir.resolve()
                    ),
                )
            from text2sql.core.progress import ProgressReporter
            from text2sql.multi_turn_agent.runner import run_agent_evaluation

            progress = ProgressReporter(
                enabled=not args.no_progress,
                interval_seconds=args.progress_interval_seconds,
                stream=sys.stderr,
            )
            try:
                result = run_agent_evaluation(
                    agent_config,
                    backend_name=args.backend,
                    allow_model_download=args.allow_model_download,
                    run_name=args.run_name,
                    resume_run=args.resume_run,
                    selection=args.selection,
                    progress=progress,
                    invocation={
                        "interface": "cli",
                        "command": "agent-evaluate",
                        "config": str(agent_config.source_path),
                        "backend": args.backend,
                        "selection": args.selection,
                        "gpu_pools": {
                            "planner": list(planner_gpus),
                            "coder": list(coder_gpus),
                            "verifier": list(verifier_gpus),
                        },
                        "worker_logical_device": "cuda:0",
                        "output_directory": str(agent_config.output.directory),
                        "run_name": args.run_name,
                        "resume_run": args.resume_run,
                        "allow_model_download": args.allow_model_download,
                        "progress_enabled": not args.no_progress,
                        "progress_interval_seconds": args.progress_interval_seconds,
                    },
                )
            finally:
                progress.close()
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 0 if result["summary"]["pipeline_pass"] else 1

        config = load_config(args.config)
        if args.command == "validate-data":
            report = SpiderDataset(config.spider).validate()
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
            return 0 if report.ok else 1
        if args.command == "validate-official":
            from text2sql.core.official_eval import validate_official_environment

            dataset = SpiderDataset(config.spider)
            db_ids = (
                [example.db_id for example in dataset.examples]
                if args.all_examples
                else [sample.db_id for sample in config.smoke.samples]
            )
            report = validate_official_environment(
                evaluator_root=config.official_evaluation.evaluator_root,
                database_root=config.official_evaluation.test_suite_database_root,
                tables_path=dataset.tables_path,
                expected_commit=config.official_evaluation.upstream_commit,
                nltk_data_dir=config.official_evaluation.nltk_data_dir,
                db_ids=db_ids,
            )
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
            return 0 if report["ok"] else 1
        if args.command == "doctor":
            from text2sql.core.doctor import server_doctor
            from text2sql.core.model_source import inspect_local_checkpoint

            report = server_doctor(config.model.device)
            report["configured_model"] = config.model.model_id
            report["configured_model_source"] = config.model.source
            report["configured_revision"] = config.model.revision
            if config.model.source == "local":
                checkpoint = inspect_local_checkpoint(Path(config.model.model_id))
                report["local_checkpoint"] = checkpoint.to_dict()
                if not checkpoint.ok:
                    report["errors"].extend(checkpoint.errors)
                    report["ok"] = False
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
            return 0 if report["ok"] else 1
        if args.command == "smoke":
            if args.allow_model_download and args.backend != "hf":
                raise ConfigError("--allow-model-download is valid only with --backend hf")
            if args.device:
                if re.fullmatch(r"cuda:[0-9]+", args.device) is None:
                    raise ConfigError("--device must use the explicit cuda:<index> form")
                config = replace(config, model=replace(config.model, device=args.device))
            if args.output_dir:
                config = replace(
                    config,
                    output=replace(config.output, directory=args.output_dir.resolve()),
                )
            result = run_smoke(
                config,
                backend_name=args.backend,
                allow_model_download=args.allow_model_download,
                run_name=args.run_name,
                invocation={
                    "interface": "cli",
                    "command": "smoke",
                    "config": str(config.source_path),
                    "backend": args.backend,
                    "device": config.model.device,
                    "output_directory": str(config.output.directory),
                    "run_name": args.run_name,
                    "allow_model_download": args.allow_model_download,
                },
            )
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 0 if result["summary"]["pipeline_pass"] else 1
        if args.command == "evaluate":
            from text2sql.single_turn.distributed import parse_gpu_ids
            from text2sql.single_turn.evaluation_runner import run_full_evaluation

            if args.allow_model_download and args.backend not in {"hf", "peft"}:
                raise ConfigError(
                    "--allow-model-download is valid only with --backend hf or peft"
                )
            if args.backend == "peft" and args.adapter_dir is None:
                raise ConfigError("--adapter-dir is required with --backend peft")
            if args.backend != "peft" and args.adapter_dir is not None:
                raise ConfigError("--adapter-dir is valid only with --backend peft")
            if (
                not math.isfinite(args.progress_interval_seconds)
                or args.progress_interval_seconds <= 0
            ):
                raise ConfigError("--progress-interval-seconds must be positive and finite")
            gpu_ids = parse_gpu_ids(args.gpus)
            output_directory = (
                args.output_dir.resolve()
                if args.output_dir is not None
                else config.output.directory.resolve()
            )
            config = replace(
                config,
                model=replace(config.model, device="cuda:0"),
                output=replace(config.output, directory=output_directory),
            )
            from text2sql.core.progress import ProgressReporter

            progress = ProgressReporter(
                enabled=not args.no_progress,
                interval_seconds=args.progress_interval_seconds,
                stream=sys.stderr,
            )
            try:
                result = run_full_evaluation(
                    config,
                    backend_name=args.backend,
                    gpu_ids=gpu_ids,
                    allow_model_download=args.allow_model_download,
                    run_name=args.run_name,
                    resume_run=args.resume_run,
                    selection=args.selection,
                    progress=progress,
                    adapter_dir=(
                        args.adapter_dir.resolve()
                        if args.adapter_dir is not None
                        else None
                    ),
                    invocation={
                        "interface": "cli",
                        "command": "evaluate",
                        "config": str(config.source_path),
                        "backend": args.backend,
                        "selection": args.selection,
                        "physical_gpu_ids": list(gpu_ids),
                        "worker_logical_device": "cuda:0",
                        "output_directory": str(output_directory),
                        "run_name": args.run_name,
                        "resume_run": args.resume_run,
                        "allow_model_download": args.allow_model_download,
                        "adapter_directory": (
                            str(args.adapter_dir.resolve())
                            if args.adapter_dir is not None
                            else None
                        ),
                        "progress_enabled": not args.no_progress,
                        "progress_interval_seconds": args.progress_interval_seconds,
                    },
                )
            finally:
                progress.close()
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 0 if result["summary"]["pipeline_pass"] else 1
        if args.command == "_generation-worker":
            from text2sql.single_turn.distributed import run_generation_worker

            if args.allow_model_download and args.backend not in {
                "hf",
                "peft",
                "two_turn",
            }:
                raise ConfigError(
                    "--allow-model-download is invalid for the selected backend"
                )
            run_generation_worker(
                config_path=args.config,
                assignment_path=args.assignment,
                shard_path=args.shard,
                status_path=args.status,
                backend_name=args.backend,
                allow_model_download=args.allow_model_download,
            )
            return 0
        raise AssertionError("unhandled command")
    except (ConfigError, SpiderDataError, RuntimeError, ValueError, OSError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
