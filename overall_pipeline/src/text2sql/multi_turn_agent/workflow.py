from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from text2sql.core.models import ExecutionResult, GenerationRequest, GenerationResult, ParseResult
from text2sql.core.sql_output import extract_sql
from text2sql.multi_turn_agent.contracts import (
    MAX_ITERATIONS,
    ContractError,
    parse_planner_output,
    parse_verifier_output,
)
from text2sql.multi_turn_agent.observation import bound_execution_observation
from text2sql.multi_turn_agent.prompts import (
    PLANNER_FORMAT_RETRY_PROMPT,
    VERIFIER_FORMAT_RETRY_PROMPT,
    build_coder_messages,
    build_planner_messages,
    build_verifier_messages,
)


RoleGenerator = Callable[[GenerationRequest], GenerationResult]
CandidateExecutor = Callable[[str], ExecutionResult]
StageCallback = Callable[[str, Mapping[str, Any]], None]


@dataclass(frozen=True)
class AgentEpisodeRequest:
    example_id: str
    question: str
    serialized_schema: str


@dataclass(frozen=True)
class GenerationTrace:
    status: str
    elapsed_seconds: float
    raw_output: str
    error_type: Optional[str]
    error_message: Optional[str]

    @classmethod
    def from_result(cls, result: GenerationResult) -> "GenerationTrace":
        return cls(
            status=result.status,
            elapsed_seconds=result.elapsed_seconds,
            raw_output=result.raw_output,
            error_type=result.error_type,
            error_message=result.error_message,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GenerationTrace":
        return cls(
            status=str(payload["status"]),
            elapsed_seconds=float(payload["elapsed_seconds"]),
            raw_output=str(payload["raw_output"]),
            error_type=str(payload["error_type"]) if payload.get("error_type") is not None else None,
            error_message=(
                str(payload["error_message"])
                if payload.get("error_message") is not None
                else None
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RoleTrace:
    prompt_sha256: str
    generation: GenerationTrace
    parsed_output: Optional[Mapping[str, Any]] = None
    contract_error: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RoleTrace":
        parsed = payload.get("parsed_output")
        return cls(
            prompt_sha256=str(payload["prompt_sha256"]),
            generation=GenerationTrace.from_dict(payload["generation"]),
            parsed_output=dict(parsed) if isinstance(parsed, Mapping) else None,
            contract_error=(
                str(payload["contract_error"])
                if payload.get("contract_error") is not None
                else None
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_sha256": self.prompt_sha256,
            "generation": self.generation.to_dict(),
            "parsed_output": dict(self.parsed_output) if self.parsed_output is not None else None,
            "contract_error": self.contract_error,
        }


@dataclass
class AgentIterationRecord:
    iteration: int
    planner: RoleTrace
    coder: Optional[RoleTrace] = None
    sql_parsing: Optional[Mapping[str, Any]] = None
    execution_observation: Optional[Mapping[str, Any]] = None
    verifier: Optional[RoleTrace] = None
    verifier_initial_attempt: Optional[RoleTrace] = None
    planner_initial_attempt: Optional[RoleTrace] = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AgentIterationRecord":
        coder = payload.get("coder")
        verifier = payload.get("verifier")
        initial_attempt = payload.get("verifier_initial_attempt")
        planner_initial_attempt = payload.get("planner_initial_attempt")
        sql_parsing = payload.get("sql_parsing")
        observation = payload.get("execution_observation")
        return cls(
            iteration=int(payload["iteration"]),
            planner=RoleTrace.from_dict(payload["planner"]),
            coder=RoleTrace.from_dict(coder) if isinstance(coder, Mapping) else None,
            sql_parsing=dict(sql_parsing) if isinstance(sql_parsing, Mapping) else None,
            execution_observation=(dict(observation) if isinstance(observation, Mapping) else None),
            verifier=RoleTrace.from_dict(verifier) if isinstance(verifier, Mapping) else None,
            verifier_initial_attempt=(
                RoleTrace.from_dict(initial_attempt)
                if isinstance(initial_attempt, Mapping) else None
            ),
            planner_initial_attempt=(
                RoleTrace.from_dict(planner_initial_attempt)
                if isinstance(planner_initial_attempt, Mapping) else None
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "planner": self.planner.to_dict(),
            "coder": self.coder.to_dict() if self.coder is not None else None,
            "sql_parsing": dict(self.sql_parsing) if self.sql_parsing is not None else None,
            "execution_observation": (
                dict(self.execution_observation)
                if self.execution_observation is not None
                else None
            ),
            "verifier": self.verifier.to_dict() if self.verifier is not None else None,
            "verifier_initial_attempt": (
                self.verifier_initial_attempt.to_dict()
                if self.verifier_initial_attempt is not None else None
            ),
            "planner_initial_attempt": (
                self.planner_initial_attempt.to_dict()
                if self.planner_initial_attempt is not None else None
            ),
        }

    def to_planner_history(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "planner": self.planner.parsed_output,
            "coder": {
                "generation_status": self.coder.generation.status if self.coder else None,
                "raw_output": self.coder.generation.raw_output if self.coder else None,
                "sql_parsing": self.sql_parsing,
            },
            "execution_observation": self.execution_observation,
            "verifier": self.verifier.parsed_output if self.verifier else None,
        }


@dataclass
class EpisodeResult:
    example_id: str
    status: str
    termination_reason: str
    final_iteration: Optional[int]
    final_raw_output: Optional[str]
    final_sql: Optional[str]
    final_sql_parsing: Optional[Mapping[str, Any]]
    iterations: Sequence[AgentIterationRecord] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EpisodeResult":
        parsing = payload.get("final_sql_parsing")
        return cls(
            example_id=str(payload["example_id"]),
            status=str(payload["status"]),
            termination_reason=str(payload["termination_reason"]),
            final_iteration=(
                int(payload["final_iteration"])
                if payload.get("final_iteration") is not None
                else None
            ),
            final_raw_output=(
                str(payload["final_raw_output"])
                if payload.get("final_raw_output") is not None
                else None
            ),
            final_sql=str(payload["final_sql"]) if payload.get("final_sql") is not None else None,
            final_sql_parsing=dict(parsing) if isinstance(parsing, Mapping) else None,
            iterations=tuple(
                AgentIterationRecord.from_dict(item) for item in payload.get("iterations", [])
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "status": self.status,
            "termination_reason": self.termination_reason,
            "final_iteration": self.final_iteration,
            "final_raw_output": self.final_raw_output,
            "final_sql": self.final_sql,
            "final_sql_parsing": (
                dict(self.final_sql_parsing) if self.final_sql_parsing is not None else None
            ),
            "iterations": [record.to_dict() for record in self.iterations],
        }


@dataclass
class EpisodeCheckpoint:
    example_id: str
    next_iteration: int
    next_stage: str
    completed_iterations: Sequence[AgentIterationRecord]
    current_iteration: Optional[AgentIterationRecord]
    last_iteration: Optional[int]
    last_raw_output: Optional[str]
    last_sql_parsing: Optional[Mapping[str, Any]]
    result: Optional[EpisodeResult] = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EpisodeCheckpoint":
        if int(payload.get("schema_version", -1)) != 1:
            raise ValueError("Unsupported episode checkpoint schema")
        current = payload.get("current_iteration")
        parsing = payload.get("last_sql_parsing")
        result = payload.get("result")
        checkpoint = cls(
            example_id=str(payload["example_id"]),
            next_iteration=int(payload["next_iteration"]),
            next_stage=str(payload["next_stage"]),
            completed_iterations=tuple(
                AgentIterationRecord.from_dict(item)
                for item in payload.get("completed_iterations", [])
            ),
            current_iteration=(
                AgentIterationRecord.from_dict(current) if isinstance(current, Mapping) else None
            ),
            last_iteration=(
                int(payload["last_iteration"])
                if payload.get("last_iteration") is not None
                else None
            ),
            last_raw_output=(
                str(payload["last_raw_output"])
                if payload.get("last_raw_output") is not None
                else None
            ),
            last_sql_parsing=dict(parsing) if isinstance(parsing, Mapping) else None,
            result=EpisodeResult.from_dict(result) if isinstance(result, Mapping) else None,
        )
        checkpoint.validate()
        return checkpoint

    def validate(self) -> None:
        if self.next_stage not in {
            "planner",
            "coder",
            "execution",
            "verifier",
            "complete",
        }:
            raise ValueError("Invalid checkpoint next_stage: %s" % self.next_stage)
        if self.next_iteration < 1 or self.next_iteration > MAX_ITERATIONS + 1:
            raise ValueError("Checkpoint iteration is outside the episode boundary")
        if self.next_stage in {"coder", "execution", "verifier"}:
            if self.current_iteration is None:
                raise ValueError("Checkpoint is missing its current iteration")
            if self.current_iteration.iteration != self.next_iteration:
                raise ValueError("Checkpoint current iteration does not match next_iteration")
        if self.next_stage in {"execution", "verifier"}:
            if (
                self.current_iteration is None
                or self.current_iteration.coder is None
                or self.current_iteration.sql_parsing is None
            ):
                raise ValueError(
                    "Execution/verifier checkpoint is missing coder output"
                )
        if self.next_stage == "verifier" and (
            self.current_iteration is None
            or self.current_iteration.execution_observation is None
        ):
            raise ValueError("Verifier checkpoint is missing execution observation")
        if self.next_stage == "complete" and self.result is None:
            raise ValueError("Complete checkpoint is missing its result")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "example_id": self.example_id,
            "next_iteration": self.next_iteration,
            "next_stage": self.next_stage,
            "completed_iterations": [r.to_dict() for r in self.completed_iterations],
            "current_iteration": (
                self.current_iteration.to_dict() if self.current_iteration is not None else None
            ),
            "last_iteration": self.last_iteration,
            "last_raw_output": self.last_raw_output,
            "last_sql_parsing": (
                dict(self.last_sql_parsing) if self.last_sql_parsing is not None else None
            ),
            "result": self.result.to_dict() if self.result is not None else None,
        }


def _prompt_sha256(messages: Sequence[Mapping[str, str]]) -> str:
    encoded = json.dumps(
        list(messages), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _request_id(example_id: str, role: str, iteration: int) -> str:
    return "%s:%s:iteration-%d" % (example_id, role, iteration)


def _generate(
    generate: RoleGenerator,
    example_id: str,
    role: str,
    iteration: int,
    messages: Sequence[Mapping[str, str]],
) -> GenerationResult:
    result = generate(
        GenerationRequest(
            example_id=_request_id(example_id, role, iteration), messages=messages
        )
    )
    if not isinstance(result, GenerationResult):
        raise TypeError("%s generator must return GenerationResult" % role)
    return result


def _role_trace(
    messages: Sequence[Mapping[str, str]],
    generation: GenerationResult,
    parsed_output: Optional[Mapping[str, Any]] = None,
    contract_error: Optional[str] = None,
) -> RoleTrace:
    return RoleTrace(
        prompt_sha256=_prompt_sha256(messages),
        generation=GenerationTrace.from_result(generation),
        parsed_output=parsed_output,
        contract_error=contract_error,
    )


def _parse_result(payload: Optional[Mapping[str, Any]]) -> Optional[ParseResult]:
    return ParseResult(**dict(payload)) if payload is not None else None


def _not_executed_observation(parse_result: ParseResult) -> Dict[str, Any]:
    return bound_execution_observation(
        ExecutionResult(
            status="not_executed",
            error_type=parse_result.error_type,
            error_message=parse_result.error_message,
        )
    )


def _result(
    request: AgentEpisodeRequest,
    records: Sequence[AgentIterationRecord],
    termination_reason: str,
    last_iteration: Optional[int],
    last_raw_output: Optional[str],
    last_parse: Optional[ParseResult],
) -> EpisodeResult:
    return EpisodeResult(
        example_id=request.example_id,
        status="completed" if last_iteration is not None else "no_candidate",
        termination_reason=termination_reason,
        final_iteration=last_iteration,
        final_raw_output=last_raw_output,
        final_sql=(
            last_parse.sql
            if last_parse is not None and last_parse.status == "success"
            else None
        ),
        final_sql_parsing=last_parse.to_dict() if last_parse is not None else None,
        iterations=tuple(records),
    )


def _checkpoint(
    request: AgentEpisodeRequest,
    records: Sequence[AgentIterationRecord],
    current: Optional[AgentIterationRecord],
    next_iteration: int,
    next_stage: str,
    last_iteration: Optional[int],
    last_raw_output: Optional[str],
    last_parse: Optional[ParseResult],
    result: Optional[EpisodeResult] = None,
) -> EpisodeCheckpoint:
    return EpisodeCheckpoint(
        example_id=request.example_id,
        next_iteration=next_iteration,
        next_stage=next_stage,
        completed_iterations=tuple(records),
        current_iteration=current,
        last_iteration=last_iteration,
        last_raw_output=last_raw_output,
        last_sql_parsing=last_parse.to_dict() if last_parse is not None else None,
        result=result,
    )


def _notify(callback: Optional[StageCallback], event: str, state: EpisodeCheckpoint) -> None:
    if callback is not None:
        callback(event, state.to_dict())


def _finish_after_stage(
    request: AgentEpisodeRequest,
    records: Sequence[AgentIterationRecord],
    termination_reason: str,
    last_iteration: Optional[int],
    last_raw_output: Optional[str],
    last_parse: Optional[ParseResult],
    callback: Optional[StageCallback],
    event: str,
) -> EpisodeResult:
    result = _result(
        request, records, termination_reason, last_iteration, last_raw_output, last_parse
    )
    state = _checkpoint(
        request,
        records,
        None,
        min(MAX_ITERATIONS + 1, (last_iteration or 0) + 1),
        "complete",
        last_iteration,
        last_raw_output,
        last_parse,
        result=result,
    )
    _notify(callback, event, state)
    return result


def run_episode(
    request: AgentEpisodeRequest,
    planner_generate: RoleGenerator,
    coder_generate: RoleGenerator,
    verifier_generate: RoleGenerator,
    execute_candidate: CandidateExecutor,
    stage_callback: Optional[StageCallback] = None,
    resume_state: Optional[Mapping[str, Any]] = None,
) -> EpisodeResult:
    """Run one deterministic planner-coder-executor-verifier episode.

    The callback receives a complete serializable checkpoint after planner,
    coder/parse, SQL execution, and verifier. Passing that mapping back as
    ``resume_state`` skips every completed LLM call and tool stage. Exceptions
    raised by injected callables propagate so the outer process coordinator can
    retry them once. Planner and verifier contract failures receive one bounded
    format-repair generation.
    """

    if resume_state is not None:
        saved = EpisodeCheckpoint.from_dict(resume_state)
        if saved.example_id != request.example_id:
            raise ValueError("Checkpoint example_id does not match the request")
        if saved.next_stage == "complete":
            assert saved.result is not None
            return saved.result
        records = list(saved.completed_iterations)
        current = saved.current_iteration
        iteration = saved.next_iteration
        stage = saved.next_stage
        last_iteration = saved.last_iteration
        last_raw_output = saved.last_raw_output
        last_parse = _parse_result(saved.last_sql_parsing)
    else:
        records: List[AgentIterationRecord] = []
        current: Optional[AgentIterationRecord] = None
        iteration = 1
        stage = "planner"
        last_iteration: Optional[int] = None
        last_raw_output: Optional[str] = None
        last_parse: Optional[ParseResult] = None

    while iteration <= MAX_ITERATIONS:
        if stage == "planner":
            planner_messages = build_planner_messages(
                request.question,
                request.serialized_schema,
                iteration,
                [record.to_planner_history() for record in records],
            )
            planner_initial_attempt = None
            for attempt in range(2):
                planner_generation = _generate(
                    planner_generate,
                    request.example_id,
                    "planner" if attempt == 0 else "planner-format-retry",
                    iteration,
                    planner_messages,
                )
                if planner_generation.status != "success":
                    current = AgentIterationRecord(
                        iteration=iteration,
                        planner=_role_trace(planner_messages, planner_generation),
                        planner_initial_attempt=planner_initial_attempt,
                    )
                    records.append(current)
                    return _finish_after_stage(
                        request, records, "planner_generation_error", last_iteration,
                        last_raw_output, last_parse, stage_callback, "planner_completed"
                    )
                try:
                    planner_output = parse_planner_output(
                        planner_generation.raw_output, request.serialized_schema
                    )
                    break
                except ContractError as exc:
                    trace = _role_trace(
                        planner_messages, planner_generation, contract_error=str(exc)
                    )
                    if attempt == 0:
                        planner_initial_attempt = trace
                        planner_messages = planner_messages + [
                            {"role": "assistant", "content": planner_generation.raw_output},
                            {"role": "user", "content": PLANNER_FORMAT_RETRY_PROMPT + str(exc)},
                        ]
                        continue
                    current = AgentIterationRecord(
                        iteration=iteration,
                        planner=trace,
                        planner_initial_attempt=planner_initial_attempt,
                    )
                    records.append(current)
                    return _finish_after_stage(
                        request, records, "planner_output_error", last_iteration,
                        last_raw_output, last_parse, stage_callback, "planner_completed"
                    )
            current = AgentIterationRecord(
                iteration=iteration,
                planner=_role_trace(
                    planner_messages,
                    planner_generation,
                    parsed_output=planner_output.to_dict(),
                ),
                planner_initial_attempt=planner_initial_attempt,
            )
            stage = "coder"
            _notify(
                stage_callback,
                "planner_completed",
                _checkpoint(
                    request, records, current, iteration, stage, last_iteration,
                    last_raw_output, last_parse
                ),
            )

        if stage == "coder":
            if current is None or current.planner.parsed_output is None:
                raise ValueError("Coder stage has no valid planner output")
            planner_output = parse_planner_output(
                json.dumps(current.planner.parsed_output), request.serialized_schema
            )
            coder_messages = build_coder_messages(
                request.question,
                request.serialized_schema,
                iteration,
                planner_output,
            )
            coder_generation = _generate(
                coder_generate, request.example_id, "coder", iteration, coder_messages
            )
            current.coder = _role_trace(coder_messages, coder_generation)
            if coder_generation.status == "success":
                parse_result = extract_sql(coder_generation.raw_output)
            else:
                parse_result = ParseResult(
                    status="error",
                    error_type=coder_generation.error_type or "coder_generation_error",
                    error_message=coder_generation.error_message or "Coder generation failed",
                )
            current.sql_parsing = parse_result.to_dict()
            last_iteration = iteration
            last_raw_output = coder_generation.raw_output
            last_parse = parse_result
            stage = "execution"
            _notify(
                stage_callback,
                "coder_completed",
                _checkpoint(
                    request, records, current, iteration, stage, last_iteration,
                    last_raw_output, last_parse
                ),
            )

        if stage == "execution":
            if current is None or current.sql_parsing is None:
                raise ValueError("Execution stage has no parsed coder output")
            parse_result = _parse_result(current.sql_parsing)
            assert parse_result is not None
            if parse_result.status == "success" and parse_result.sql is not None:
                execution = execute_candidate(parse_result.sql)
                if not isinstance(execution, ExecutionResult):
                    raise TypeError("execute_candidate must return ExecutionResult")
                observation = bound_execution_observation(execution)
            else:
                observation = _not_executed_observation(parse_result)
            current.execution_observation = observation
            stage = "verifier"
            _notify(
                stage_callback,
                "execution_completed",
                _checkpoint(
                    request, records, current, iteration, stage, last_iteration,
                    last_raw_output, last_parse
                ),
            )

        if stage == "verifier":
            if (
                current is None
                or current.planner.parsed_output is None
                or current.coder is None
                or current.sql_parsing is None
                or current.execution_observation is None
            ):
                raise ValueError("Verifier stage checkpoint is incomplete")
            planner_output = parse_planner_output(
                json.dumps(current.planner.parsed_output), request.serialized_schema
            )
            verifier_messages = build_verifier_messages(
                request.question,
                request.serialized_schema,
                iteration,
                planner_output,
                current.coder.generation.status,
                current.coder.generation.raw_output,
                current.sql_parsing,
                current.execution_observation,
            )
            for attempt in range(2):
                verifier_generation = _generate(
                    verifier_generate,
                    request.example_id,
                    "verifier" if attempt == 0 else "verifier-format-retry",
                    iteration,
                    verifier_messages,
                )
                if verifier_generation.status != "success":
                    current.verifier = _role_trace(verifier_messages, verifier_generation)
                    records.append(current)
                    return _finish_after_stage(
                        request, records, "verifier_generation_error", last_iteration,
                        last_raw_output, last_parse, stage_callback, "verifier_completed"
                    )
                try:
                    verifier_output = parse_verifier_output(verifier_generation.raw_output)
                    break
                except ContractError as exc:
                    trace = _role_trace(
                        verifier_messages, verifier_generation, contract_error=str(exc)
                    )
                    if attempt == 0:
                        current.verifier_initial_attempt = trace
                        verifier_messages = verifier_messages + [
                            {"role": "assistant", "content": verifier_generation.raw_output},
                            {"role": "user", "content": VERIFIER_FORMAT_RETRY_PROMPT},
                        ]
                        continue
                    current.verifier = trace
                    records.append(current)
                    return _finish_after_stage(
                        request, records, "verifier_output_error", last_iteration,
                        last_raw_output, last_parse, stage_callback, "verifier_completed"
                    )
            current.verifier = _role_trace(
                verifier_messages,
                verifier_generation,
                parsed_output=verifier_output.to_dict(),
            )
            records.append(current)
            if verifier_output.decision == "stop":
                return _finish_after_stage(
                    request, records, "verifier_stop", last_iteration, last_raw_output,
                    last_parse, stage_callback, "verifier_completed"
                )
            if iteration == MAX_ITERATIONS:
                return _finish_after_stage(
                    request, records, "max_iterations_reached", last_iteration,
                    last_raw_output, last_parse, stage_callback, "verifier_completed"
                )
            iteration += 1
            current = None
            stage = "planner"
            _notify(
                stage_callback,
                "verifier_completed",
                _checkpoint(
                    request, records, current, iteration, stage, last_iteration,
                    last_raw_output, last_parse
                ),
            )

    raise AssertionError("unreachable episode state")
