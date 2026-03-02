"""
Symbolic Checker and Self-answer Checker.

Symbolic Checker: rule-based script that checks if generated code
uses real algorithmic computation (loops, search, recursion, etc.)
rather than hardcoded answers.

Self-answer Checker: asks TaskLLM to verify its own answer via code execution.
"""

from __future__ import annotations
import ast
import re
import subprocess
import sys
import tempfile
import os


# ─────────────────────────────────────────────
#  Symbolic Checker
# ─────────────────────────────────────────────

SYMBOLIC_KEYWORDS = [
    "for", "while", "range", "itertools", "permutations", "combinations",
    "product", "enumerate", "recursion", "dfs", "bfs", "backtrack",
    "queue", "stack", "heapq", "sorted", "numpy", "scipy", "sympy",
    "solve", "optimize", "minimize", "maximize", "search",
]

HARDCODE_PATTERNS = [
    r"answer\s*=\s*['\"]?\d+['\"]?",       # answer = 42
    r"return\s+['\"]?\d+['\"]?",            # return 42
    r"print\s*\(\s*['\"]?\d+['\"]?\s*\)",   # print(42)
]


def symbolic_check(code: str) -> dict:
    """
    Analyze code for algorithmic complexity.
    Returns: {"score": int (0-10), "summary": str, "is_symbolic": bool}
    """
    if not code or not code.strip():
        return {"score": 0, "summary": "No code provided.", "is_symbolic": False}

    score = 0
    notes = []

    # Check for symbolic/algorithmic keywords
    lower = code.lower()
    found_keywords = [kw for kw in SYMBOLIC_KEYWORDS if kw in lower]
    score += min(len(found_keywords) * 2, 6)
    if found_keywords:
        notes.append(f"Uses: {', '.join(found_keywords[:5])}")

    # Penalize hardcoded answers
    hardcoded = sum(1 for p in HARDCODE_PATTERNS if re.search(p, code))
    score -= hardcoded * 3
    if hardcoded:
        notes.append(f"⚠️ Possible hardcoded answer ({hardcoded} patterns)")

    # Bonus for nesting depth (real algorithms tend to be deeper)
    try:
        tree = ast.parse(code)
        max_depth = _ast_depth(tree)
        if max_depth >= 3:
            score += 2
            notes.append(f"AST depth: {max_depth}")
    except SyntaxError:
        notes.append("⚠️ Syntax error in code")
        score -= 2

    # Bonus for code length (minimal one-liners are often hardcoded)
    lines = [l for l in code.split("\n") if l.strip() and not l.strip().startswith("#")]
    if len(lines) >= 5:
        score += 1

    score = max(0, min(10, score))
    is_symbolic = score >= 4 and hardcoded == 0

    return {
        "score": score,
        "summary": f"Score {score}/10. " + " ".join(notes),
        "is_symbolic": is_symbolic,
    }


def _ast_depth(node, depth=0):
    """Recursively find the maximum depth of an AST node."""
    children = list(ast.iter_child_nodes(node))
    if not children:
        return depth
    return max(_ast_depth(child, depth + 1) for child in children)


# ─────────────────────────────────────────────
#  Code Executor
# ─────────────────────────────────────────────

def extract_code(text: str) -> str | None:
    """Extract Python code from a markdown code block."""
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # fallback: look for any code block
    match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def execute_code(code: str, timeout: int = 30) -> dict:
    """
    Execute Python code in a subprocess with timeout.
    Returns: {"stdout": str, "stderr": str, "success": bool, "error": str}
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "success": result.returncode == 0,
            "error": "",
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "", "success": False, "error": f"Timeout after {timeout}s"}
    except Exception as e:
        return {"stdout": "", "stderr": "", "success": False, "error": str(e)}
    finally:
        os.unlink(tmp_path)


# ─────────────────────────────────────────────
#  Self-answer Checker
# ─────────────────────────────────────────────

SELFANSWER_PROMPT = """The following answer was given to this task. Please write Python code to verify whether the answer is correct.

Task: {task}
Claimed answer: {answer}

Write Python code that:
1. Implements a verification function
2. Checks if the claimed answer satisfies all constraints of the task
3. Prints "CORRECT" or "INCORRECT: <reason>"

```python
# verification code here
```"""


def selfanswer_check(task_llm, task: str, answer: str, timeout: int = 30) -> dict:
    """
    Ask TaskLLM to write verification code, then execute it.
    Returns: {"verdict": "CORRECT"|"INCORRECT"|"UNKNOWN", "details": str}
    """
    prompt = SELFANSWER_PROMPT.format(task=task, answer=answer)
    response = task_llm.solve(task=prompt, guidance="Write verification code", guidance_type="CODE")

    code = extract_code(response)
    if not code:
        return {"verdict": "UNKNOWN", "details": "Could not extract verification code"}

    exec_result = execute_code(code, timeout=timeout)
    if not exec_result["success"]:
        return {
            "verdict": "UNKNOWN",
            "details": f"Verification code failed: {exec_result['error'] or exec_result['stderr'][:100]}",
        }

    output = exec_result["stdout"].upper()
    if "CORRECT" in output and "INCORRECT" not in output:
        return {"verdict": "CORRECT", "details": exec_result["stdout"]}
    elif "INCORRECT" in output:
        return {"verdict": "INCORRECT", "details": exec_result["stdout"]}
    else:
        return {"verdict": "UNKNOWN", "details": exec_result["stdout"][:200]}
