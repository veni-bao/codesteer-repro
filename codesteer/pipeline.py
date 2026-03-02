"""
Main CodeSteer pipeline.
Implements the multi-turn guidance loop from the paper (Section 3).
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field

from .steer_llm import CodeSteerLLM
from .task_llm import TaskLLM
from .checkers import symbolic_check, selfanswer_check, execute_code, extract_code


@dataclass
class Turn:
    turn_num: int
    guidance_type: str        # CODE | TEXT | FINALIZE | SWITCH
    guidance: str
    task_response: str
    executed_output: str = ""
    symbolic_result: dict = field(default_factory=dict)
    selfanswer_result: dict = field(default_factory=dict)
    final_answer: str = ""


@dataclass
class EvalResult:
    task_name: str
    question: str
    correct_answer: str
    final_answer: str
    success: bool
    n_turns: int
    turns: list[Turn]
    method: str = "codesteer"


class CodeSteerPipeline:
    """
    Multi-turn CodeSteer pipeline.

    At each turn:
    1. CodeSteerLLM gives guidance (code or text)
    2. TaskLLM generates answer following guidance
    3. If guidance_type == CODE: execute code, run Symbolic Checker
    4. Self-answer Checker verifies the answer
    5. CodeSteerLLM decides to continue or finalize
    """

    def __init__(self, steer_llm: CodeSteerLLM, task_llm: TaskLLM, config: dict):
        self.steer = steer_llm
        self.task = task_llm
        self.max_turns = config.get("max_turns", 5)
        self.code_timeout = config.get("code_timeout", 30)
        self.temperature = config.get("temperature", 0.0)
        self.steer_temp = config.get("steer_temperature", 0.7)

    def run(self, question: str, correct_answer: str, task_name: str = "") -> EvalResult:
        """
        Run the full CodeSteer pipeline on a single question.
        """
        history_steer = []   # for CodeSteerLLM
        history_task = []    # for TaskLLM
        turns = []
        current_answer = ""
        current_mode = "TEXT"  # default

        for turn_num in range(1, self.max_turns + 1):
            # ── Step 1: Get guidance from CodeSteerLLM ──
            symbolic_str = "N/A"
            selfanswer_str = "N/A"

            if turns:
                last = turns[-1]
                symbolic_str = last.symbolic_result.get("summary", "N/A")
                selfanswer_str = f"{last.selfanswer_result.get('verdict', 'UNKNOWN')}: {last.selfanswer_result.get('details', '')[:80]}"

            steer_out = self.steer.guide(
                task=question,
                turn=turn_num,
                max_turns=self.max_turns,
                current_answer=current_answer,
                history=history_steer,
                symbolic_result=symbolic_str,
                selfanswer_result=selfanswer_str,
                temperature=self.steer_temp,
            )

            g_type = steer_out["type"]
            guidance = steer_out["guidance"]

            # Handle SWITCH: flip the current mode
            if g_type == "SWITCH":
                current_mode = "TEXT" if current_mode == "CODE" else "CODE"
                g_type = current_mode
            elif g_type in ("CODE", "TEXT"):
                current_mode = g_type

            # ── Step 2: Finalize if CodeSteerLLM says so ──
            if g_type == "FINALIZE" or turn_num == self.max_turns:
                final = self._extract_final_answer(current_answer)
                success = self._check_success(final, correct_answer)
                turns.append(Turn(
                    turn_num=turn_num,
                    guidance_type="FINALIZE",
                    guidance=guidance,
                    task_response=current_answer,
                    final_answer=final,
                ))
                return EvalResult(
                    task_name=task_name,
                    question=question,
                    correct_answer=correct_answer,
                    final_answer=final,
                    success=success,
                    n_turns=turn_num,
                    turns=turns,
                )

            # ── Step 3: TaskLLM generates answer ──
            task_response = self.task.solve(
                task=question,
                guidance=guidance,
                guidance_type=g_type,
                history=history_task,
                temperature=self.temperature,
            )

            # ── Step 4: Execute code if CODE mode ──
            executed_output = ""
            symbolic_result = {}
            if g_type == "CODE":
                code = extract_code(task_response)
                if code:
                    symbolic_result = symbolic_check(code)
                    exec_res = execute_code(code, timeout=self.code_timeout)
                    if exec_res["success"] and exec_res["stdout"]:
                        executed_output = exec_res["stdout"]
                        task_response = task_response + f"\n\n[Code output]: {executed_output}"
                    elif exec_res["error"]:
                        task_response = task_response + f"\n\n[Code error]: {exec_res['error']}"

            # ── Step 5: Self-answer Checker ──
            selfanswer_result = {}
            if turn_num >= 2:  # run from turn 2 onwards to save API calls
                selfanswer_result = selfanswer_check(
                    self.task, question, task_response, timeout=self.code_timeout
                )

            current_answer = task_response

            turn_record = Turn(
                turn_num=turn_num,
                guidance_type=g_type,
                guidance=guidance,
                task_response=task_response,
                executed_output=executed_output,
                symbolic_result=symbolic_result,
                selfanswer_result=selfanswer_result,
            )
            turns.append(turn_record)

            # Update histories
            history_steer.append({"turn": turn_num, "type": g_type, "guidance": guidance})
            history_task.append({
                "user": f"Task: {question}\nGuidance ({g_type}): {guidance}",
                "assistant": task_response,
            })

        # Should not reach here normally
        final = self._extract_final_answer(current_answer)
        return EvalResult(
            task_name=task_name,
            question=question,
            correct_answer=correct_answer,
            final_answer=final,
            success=self._check_success(final, correct_answer),
            n_turns=self.max_turns,
            turns=turns,
        )

    def _extract_final_answer(self, response: str) -> str:
        """Extract the final answer from a response."""
        # Look for explicit answer markers
        for pattern in [
            r"(?:final answer|answer):\s*(.+?)(?:\n|$)",
            r"\*\*(.+?)\*\*",
            r"=\s*(\d[\d\s,\.]+)",
        ]:
            m = re.search(pattern, response, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        # Fallback: last non-empty line
        lines = [l.strip() for l in response.split("\n") if l.strip()]
        return lines[-1] if lines else response[:100]

    def _check_success(self, predicted: str, correct: str) -> bool:
        """Simple string-based success check. Tasks override this."""
        pred = re.sub(r"\s+", "", predicted.lower())
        corr = re.sub(r"\s+", "", correct.lower())
        return pred == corr or corr in pred


# ─────────────────────────────────────────────
#  Baseline pipelines (for comparison)
# ─────────────────────────────────────────────

class BaselinePipeline:
    """
    Runs simple baselines without CodeSteer.
    method: "text_only" | "code_only" | "only_question"
    """

    def __init__(self, task_llm: TaskLLM, method: str, config: dict):
        self.task = task_llm
        self.method = method
        self.code_timeout = config.get("code_timeout", 30)

    def run(self, question: str, correct_answer: str, task_name: str = "") -> EvalResult:
        if self.method == "only_question":
            guidance = "Answer the question directly."
            g_type = "TEXT"
        elif self.method == "text_only":
            guidance = "Think step by step and reason through this problem in text."
            g_type = "TEXT"
        elif self.method == "code_only":
            guidance = "Write Python code to solve this problem algorithmically."
            g_type = "CODE"
        else:
            raise ValueError(f"Unknown method: {self.method}")

        response = self.task.solve(
            task=question,
            guidance=guidance,
            guidance_type=g_type,
        )

        executed_output = ""
        if g_type == "CODE":
            code = extract_code(response)
            if code:
                exec_res = execute_code(code, timeout=self.code_timeout)
                if exec_res["success"]:
                    executed_output = exec_res["stdout"]
                    response = response + f"\n\n[Code output]: {executed_output}"

        final = response.strip()
        # simple check
        success = correct_answer.lower() in final.lower() if correct_answer else False

        return EvalResult(
            task_name=task_name,
            question=question,
            correct_answer=correct_answer,
            final_answer=final,
            success=success,
            n_turns=1,
            turns=[Turn(
                turn_num=1,
                guidance_type=g_type,
                guidance=guidance,
                task_response=response,
                executed_output=executed_output,
            )],
            method=self.method,
        )
