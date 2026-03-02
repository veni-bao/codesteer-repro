"""
SymBench task definitions.
Each task provides:
  - generate_question(complexity): generate a random question at given complexity
  - check_answer(question, answer): return True/False
"""

from .game24 import Game24Task
from .sudoku import SudokuTask
from .path_plan import PathPlanTask
from .logical_deduction import LogicalDeductionTask
from .number_multiply import NumberMultiplyTask

TASK_REGISTRY = {
    "game24": Game24Task,
    "sudoku": SudokuTask,
    "path_plan": PathPlanTask,
    "logical_deduction": LogicalDeductionTask,
    "number_multiply": NumberMultiplyTask,
}

ALL_TASKS = list(TASK_REGISTRY.keys())
