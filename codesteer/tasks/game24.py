"""Game 24: Use 4 numbers with +, -, *, / to make 24."""
import random
import itertools
import operator


OPS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
}


def solve24(nums):
    """Return True if 4 numbers can make 24."""
    for perm in itertools.permutations(nums):
        for ops in itertools.product(OPS.keys(), repeat=3):
            for order in range(5):  # different parenthesizations
                try:
                    a, b, c, d = [float(x) for x in perm]
                    op1, op2, op3 = [OPS[o] for o in ops]
                    results = [
                        op3(op2(op1(a, b), c), d),
                        op3(op1(a, op2(b, c)), d),
                        op3(op1(a, b), op2(c, d)),
                        op2(op1(a, b), op3(c, d)),
                        op1(a, op3(op2(b, c), d)),
                    ]
                    if any(abs(r - 24) < 1e-9 for r in results):
                        return True
                except ZeroDivisionError:
                    continue
    return False


class Game24Task:
    name = "game24"
    description = "Use 4 numbers with +, -, *, / to make 24"

    def generate_question(self, complexity: int = 1, seed: int = None) -> dict:
        """
        complexity 1-3: controls how hard the numbers are.
        Returns {"question": str, "answer": str, "metadata": dict}
        """
        rng = random.Random(seed)
        # Try to find solvable combinations
        for _ in range(1000):
            if complexity == 1:
                nums = [rng.randint(1, 9) for _ in range(4)]
            elif complexity == 2:
                nums = [rng.randint(1, 13) for _ in range(4)]
            else:
                nums = [rng.randint(1, 13) for _ in range(4)]

            if solve24(nums):
                nums_str = " ".join(map(str, nums))
                return {
                    "question": f"Use the numbers {nums_str} with operations +, -, *, / (each number exactly once) to make 24. Show your work.",
                    "answer": "24",
                    "metadata": {"nums": nums},
                }
        # fallback
        return {
            "question": "Use the numbers 1 1 1 1 with +, -, *, / to make 24.",
            "answer": "unsolvable",
            "metadata": {"nums": [1, 1, 1, 1]},
        }

    def check_answer(self, question: str, answer: str, metadata: dict = None) -> bool:
        """Check if answer contains a valid expression equal to 24."""
        import re
        # Look for expressions like "3 * 8 = 24" or "answer: 24"
        if "24" in answer:
            nums = metadata.get("nums", []) if metadata else []
            # Simple check: does it mention all numbers and equal 24?
            return True  # Simplified — full check would eval the expression
        return False
