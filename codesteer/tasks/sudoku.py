"""Standard Sudoku task."""
import random
import copy


def generate_sudoku(seed=None, remove_count=40):
    """Generate a valid Sudoku puzzle."""
    rng = random.Random(seed)

    # Start with a solved board
    base = list(range(1, 10))
    rng.shuffle(base)
    board = [[0] * 9 for _ in range(9)]

    def valid(board, row, col, num):
        if num in board[row]:
            return False
        if num in [board[r][col] for r in range(9)]:
            return False
        br, bc = 3 * (row // 3), 3 * (col // 3)
        for r in range(br, br + 3):
            for c in range(bc, bc + 3):
                if board[r][c] == num:
                    return False
        return True

    def solve(board):
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0:
                    for num in range(1, 10):
                        if valid(board, r, c, num):
                            board[r][c] = num
                            if solve(board):
                                return True
                            board[r][c] = 0
                    return False
        return True

    # Fill first row with shuffled base
    board[0] = base[:]
    solve(board)
    solution = copy.deepcopy(board)

    # Remove cells
    positions = [(r, c) for r in range(9) for c in range(9)]
    rng.shuffle(positions)
    for r, c in positions[:remove_count]:
        board[r][c] = 0

    return board, solution


def board_to_str(board):
    rows = []
    for r in board:
        rows.append(" ".join(str(x) if x != 0 else "." for x in r))
    return "\n".join(rows)


class SudokuTask:
    name = "sudoku"
    description = "Solve a 9x9 Sudoku puzzle"

    def generate_question(self, complexity: int = 1, seed: int = None) -> dict:
        remove = 30 + complexity * 5
        puzzle, solution = generate_sudoku(seed=seed, remove_count=min(remove, 55))
        puzzle_str = board_to_str(puzzle)
        solution_str = board_to_str(solution)
        return {
            "question": f"Solve this Sudoku puzzle (. = empty cell):\n{puzzle_str}\n\nFill in all cells so each row, column, and 3x3 box contains 1-9 exactly once.",
            "answer": solution_str,
            "metadata": {"puzzle": puzzle, "solution": solution},
        }

    def check_answer(self, question: str, answer: str, metadata: dict = None) -> bool:
        if not metadata:
            return False
        solution = metadata.get("solution", [])
        solution_str = board_to_str(solution)
        # Check if the solution grid appears in the answer
        return solution_str.replace("\n", "") in answer.replace("\n", "").replace(" ", "")
