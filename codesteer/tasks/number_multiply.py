"""Number Multiplication task: multiply large numbers."""
import random


class NumberMultiplyTask:
    name = "number_multiply"
    description = "Multiply large numbers correctly"

    def generate_question(self, complexity: int = 1, seed: int = None) -> dict:
        rng = random.Random(seed)
        digits = 3 + complexity * 2  # complexity 1 → 5-digit, complexity 3 → 9-digit
        a = rng.randint(10 ** (digits - 1), 10 ** digits - 1)
        b = rng.randint(10 ** (digits - 1), 10 ** digits - 1)
        result = a * b
        return {
            "question": f"Calculate the exact product: {a} × {b}",
            "answer": str(result),
            "metadata": {"a": a, "b": b, "result": result},
        }

    def check_answer(self, question: str, answer: str, metadata: dict = None) -> bool:
        if not metadata:
            return False
        result = str(metadata.get("result", ""))
        return result in answer.replace(",", "").replace(" ", "")
