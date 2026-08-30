from __future__ import annotations

import json
import unittest
from typing import List

from text2sql.core.models import ExecutionResult, GenerationRequest, GenerationResult
from text2sql.multi_turn_agent.contracts import (
    ContractError,
    parse_planner_output,
    parse_verifier_output,
)
from text2sql.multi_turn_agent.observation import bound_execution_observation
from text2sql.multi_turn_agent.prompts import (
    build_planner_messages,
    build_verifier_messages,
)
from text2sql.multi_turn_agent.workflow import AgentEpisodeRequest, run_episode


def planner_json(iteration: int, approach: str = "direct") -> str:
    return json.dumps(
        {
            "iteration": iteration,
            "approach": approach,
            "plan": ["Identify the required projection."],
            "coder_instruction": "Write the corresponding read-only query.",
        }
    )


def verifier_json(iteration: int, decision: str, feedback: str = "") -> str:
    return json.dumps(
        {
            "iteration": iteration,
            "decision": decision,
            "reason": "The candidate was assessed against the request.",
            "feedback": feedback,
        }
    )


class ScriptedGenerator:
    def __init__(self, outputs: List[str]):
        self.outputs = list(outputs)
        self.requests: List[GenerationRequest] = []

    def __call__(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        if not self.outputs:
            raise AssertionError("Unexpected generation call: %s" % request.example_id)
        return GenerationResult(status="success", raw_output=self.outputs.pop(0))


class AgentContractTests(unittest.TestCase):
    def test_plain_and_single_json_fence_are_accepted(self) -> None:
        planner = parse_planner_output(planner_json(1), 1)
        self.assertEqual(planner.approach, "direct")
        fenced = "```json\n%s\n```" % verifier_json(1, "stop")
        verifier = parse_verifier_output(fenced, 1)
        self.assertEqual(verifier.decision, "stop")

    def test_contract_rejects_prose_wrong_iteration_and_empty_feedback(self) -> None:
        with self.assertRaises(ContractError):
            parse_planner_output("Here is the plan: " + planner_json(1), 1)
        with self.assertRaises(ContractError):
            parse_planner_output(planner_json(1), 2)
        with self.assertRaises(ContractError):
            parse_verifier_output(verifier_json(1, "continue"), 1)
        with self.assertRaises(ContractError):
            parse_verifier_output(
                "```json\n%s\n```\n```json\n{}\n```" % verifier_json(1, "stop"),
                1,
            )

    def test_prompts_use_current_iteration_and_mark_inputs_untrusted(self) -> None:
        history = [
            {
                "iteration": 1,
                "coder": {"raw_output": "SELECT wrong"},
                "verifier": {"feedback": "Use the visits table."},
                "execution_observation": {"vm_steps_lower_bound": 2000},
            }
        ]
        planner_messages = build_planner_messages(
            "Count visits.", "Database: fixture", 2, history
        )
        planner_rendered = "\n".join(item["content"] for item in planner_messages)
        self.assertIn("Iteration 2 of 3", planner_rendered)
        self.assertIn('"iteration":2', planner_messages[0]["content"])
        self.assertNotIn('"iteration":1', planner_messages[0]["content"])
        self.assertIn("Use the visits table.", planner_rendered)
        self.assertIn("SELECT wrong", planner_rendered)
        self.assertIn("untrusted data", planner_messages[0]["content"])
        self.assertNotIn("vm_steps", planner_rendered)

        planner_output = parse_planner_output(planner_json(3), 3)
        verifier_messages = build_verifier_messages(
            "Count visits.",
            "Database: fixture",
            3,
            planner_output,
            "success",
            "SELECT count(*) FROM visits",
            {"status": "success", "sql": "SELECT count(*) FROM visits"},
            {
                "status": "success",
                "rows": [[3]],
                "truncated": False,
                "vm_steps_lower_bound": 2000,
            },
        )
        verifier_rendered = "\n".join(item["content"] for item in verifier_messages)
        self.assertIn("Iteration 3 of 3", verifier_rendered)
        self.assertIn('"iteration":3', verifier_messages[0]["content"])
        self.assertNotIn('"iteration":1', verifier_messages[0]["content"])
        self.assertIn("final allowed iteration", verifier_rendered)
        self.assertIn("untrusted data", verifier_messages[0]["content"])
        self.assertNotIn("vm_steps", verifier_rendered)

    def test_observation_has_row_and_byte_bounds(self) -> None:
        execution = ExecutionResult(
            status="success",
            columns=["value"],
            rows=[["한" * 5000]] + [[number] for number in range(9)],
            row_count=100,
            query_elapsed_ns=1234,
            vm_steps_lower_bound=2000,
            vm_steps_upper_bound_exclusive=3000,
            vm_step_progress_interval=1000,
            vm_step_measurement_complete=True,
        )
        observation = bound_execution_observation(execution)
        encoded = json.dumps(
            observation, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        self.assertLessEqual(len(encoded), 4096)
        self.assertLessEqual(len(observation["rows"]), 5)
        self.assertEqual(observation["row_count"], 100)
        self.assertTrue(observation["truncated"])
        self.assertEqual(observation["vm_steps_lower_bound"], 2000)
        self.assertEqual(observation["vm_steps_upper_bound_exclusive"], 3000)

        failed = bound_execution_observation(
            ExecutionResult(status="execution_error", error_message="bad")
        )
        self.assertIsNone(failed["row_count"])


class AgentWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.request = AgentEpisodeRequest(
            example_id="dev:0",
            question="How many singers are there?",
            serialized_schema='Database: fixture\nTable "singer"\n  - "id" NUMBER',
        )

    @staticmethod
    def success_executor(sql: str) -> ExecutionResult:
        return ExecutionResult(
            status="success",
            columns=["count(*)"],
            rows=[[1]],
            query_elapsed_ns=99,
        )

    def test_direct_still_calls_coder_and_verifier(self) -> None:
        planner = ScriptedGenerator([planner_json(1, "direct")])
        coder = ScriptedGenerator(["SELECT count(*) FROM singer"])
        verifier = ScriptedGenerator([verifier_json(1, "stop")])
        executed: List[str] = []

        def execute(sql: str) -> ExecutionResult:
            executed.append(sql)
            return self.success_executor(sql)

        result = run_episode(self.request, planner, coder, verifier, execute)
        self.assertEqual(result.termination_reason, "verifier_stop")
        self.assertEqual(result.final_sql, "SELECT count(*) FROM singer")
        self.assertEqual(len(planner.requests), 1)
        self.assertEqual(len(coder.requests), 1)
        self.assertEqual(len(verifier.requests), 1)
        self.assertEqual(executed, ["SELECT count(*) FROM singer"])

    def test_two_continues_feed_full_history_to_third_planner(self) -> None:
        planner = ScriptedGenerator(
            [planner_json(1, "iterative"), planner_json(2), planner_json(3)]
        )
        coder = ScriptedGenerator(["SELECT 1", "SELECT 2", "SELECT 3"])
        verifier = ScriptedGenerator(
            [
                verifier_json(1, "continue", "Check the selected value."),
                verifier_json(2, "continue", "Use the final value."),
                verifier_json(3, "stop"),
            ]
        )
        result = run_episode(
            self.request, planner, coder, verifier, self.success_executor
        )
        self.assertEqual(result.termination_reason, "verifier_stop")
        self.assertEqual(result.final_iteration, 3)
        self.assertEqual(result.final_sql, "SELECT 3")
        self.assertEqual(len(result.iterations), 3)
        third_prompt = planner.requests[2].messages[1]["content"]
        self.assertIn("SELECT 1", third_prompt)
        self.assertIn("SELECT 2", third_prompt)
        self.assertIn("Check the selected value.", third_prompt)
        self.assertIn("Use the final value.", third_prompt)
        self.assertIn("query_elapsed_ns", third_prompt)

    def test_third_continue_uses_latest_candidate(self) -> None:
        result = run_episode(
            self.request,
            ScriptedGenerator([planner_json(1), planner_json(2), planner_json(3)]),
            ScriptedGenerator(["SELECT 1", "SELECT 2", "SELECT 3"]),
            ScriptedGenerator(
                [
                    verifier_json(1, "continue", "retry one"),
                    verifier_json(2, "continue", "retry two"),
                    verifier_json(3, "continue", "would retry"),
                ]
            ),
            self.success_executor,
        )
        self.assertEqual(result.termination_reason, "max_iterations_reached")
        self.assertEqual(result.final_sql, "SELECT 3")

    def test_execution_error_does_not_override_verifier_stop(self) -> None:
        def fail_execution(_sql: str) -> ExecutionResult:
            return ExecutionResult(
                status="execution_error",
                error_type="execution_error",
                error_message="no such column",
                query_elapsed_ns=100,
            )

        verifier = ScriptedGenerator([verifier_json(1, "stop")])
        result = run_episode(
            self.request,
            ScriptedGenerator([planner_json(1)]),
            ScriptedGenerator(["SELECT missing FROM singer"]),
            verifier,
            fail_execution,
        )
        self.assertEqual(result.termination_reason, "verifier_stop")
        self.assertEqual(result.final_sql, "SELECT missing FROM singer")
        verifier_prompt = verifier.requests[0].messages[1]["content"]
        self.assertIn("execution_error", verifier_prompt)
        self.assertIn("no such column", verifier_prompt)

    def test_malformed_role_json_ends_without_repair(self) -> None:
        planner = ScriptedGenerator([planner_json(1), "not json"])
        coder = ScriptedGenerator(["SELECT 1"])
        verifier = ScriptedGenerator(
            [verifier_json(1, "continue", "Try another projection.")]
        )
        result = run_episode(
            self.request, planner, coder, verifier, self.success_executor
        )
        self.assertEqual(result.termination_reason, "planner_output_error")
        self.assertEqual(result.final_sql, "SELECT 1")
        self.assertEqual(len(planner.requests), 2)
        self.assertEqual(len(coder.requests), 1)
        self.assertEqual(len(verifier.requests), 1)

        malformed_verifier = ScriptedGenerator(["not json"])
        verifier_result = run_episode(
            self.request,
            ScriptedGenerator([planner_json(1)]),
            ScriptedGenerator(["SELECT 9"]),
            malformed_verifier,
            self.success_executor,
        )
        self.assertEqual(verifier_result.termination_reason, "verifier_output_error")
        self.assertEqual(verifier_result.final_sql, "SELECT 9")
        self.assertEqual(len(malformed_verifier.requests), 1)

    def test_execution_checkpoint_resume_skips_completed_calls_and_tool(self) -> None:
        saved = []

        def checkpoint(event, state) -> None:
            saved.append((event, state))
            if event == "execution_completed":
                raise RuntimeError("simulated coordinator interruption")

        planner = ScriptedGenerator([planner_json(1)])
        coder = ScriptedGenerator(["SELECT 7"])
        verifier = ScriptedGenerator([verifier_json(1, "stop")])
        executed: List[str] = []

        def execute(sql: str) -> ExecutionResult:
            executed.append(sql)
            return self.success_executor(sql)

        with self.assertRaisesRegex(RuntimeError, "simulated"):
            run_episode(
                self.request,
                planner,
                coder,
                verifier,
                execute,
                stage_callback=checkpoint,
            )
        self.assertEqual(saved[-1][0], "execution_completed")

        def unexpected(_request: GenerationRequest) -> GenerationResult:
            raise AssertionError("A completed generation was repeated")

        resumed = run_episode(
            self.request,
            unexpected,
            unexpected,
            verifier,
            lambda _sql: (_ for _ in ()).throw(
                AssertionError("A completed SQL execution was repeated")
            ),
            resume_state=saved[-1][1],
        )
        self.assertEqual(resumed.termination_reason, "verifier_stop")
        self.assertEqual(resumed.final_sql, "SELECT 7")
        self.assertEqual(len(verifier.requests), 1)
        self.assertEqual(executed, ["SELECT 7"])

    def test_coder_checkpoint_resume_does_not_repeat_coder_generation(self) -> None:
        saved = []

        def checkpoint(event, state) -> None:
            saved.append((event, state))
            if event == "coder_completed":
                raise RuntimeError("interrupted before tool execution")

        planner = ScriptedGenerator([planner_json(1)])
        coder = ScriptedGenerator(["SELECT 8"])
        verifier = ScriptedGenerator([verifier_json(1, "stop")])
        with self.assertRaisesRegex(RuntimeError, "before tool"):
            run_episode(
                self.request,
                planner,
                coder,
                verifier,
                self.success_executor,
                stage_callback=checkpoint,
            )
        self.assertEqual(saved[-1][0], "coder_completed")
        self.assertEqual(saved[-1][1]["next_stage"], "execution")

        def unexpected(_request: GenerationRequest) -> GenerationResult:
            raise AssertionError("A completed LLM generation was repeated")

        executed: List[str] = []

        def execute(sql: str) -> ExecutionResult:
            executed.append(sql)
            return self.success_executor(sql)

        resumed = run_episode(
            self.request,
            unexpected,
            unexpected,
            verifier,
            execute,
            resume_state=saved[-1][1],
        )
        self.assertEqual(resumed.final_sql, "SELECT 8")
        self.assertEqual(executed, ["SELECT 8"])
        self.assertEqual(len(verifier.requests), 1)


if __name__ == "__main__":
    unittest.main()
