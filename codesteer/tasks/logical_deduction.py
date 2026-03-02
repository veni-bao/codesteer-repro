"""Logical Deduction task."""
import random

NAMES = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank"]
ITEMS = ["apple", "book", "cup", "desk", "envelope", "flag"]


class LogicalDeductionTask:
    name = "logical_deduction"
    description = "Deduce order from logical clues"

    def generate_question(self, complexity: int = 1, seed: int = None) -> dict:
        rng = random.Random(seed)
        n = min(3 + complexity, 6)
        names = rng.sample(NAMES, n)
        order = names[:]
        rng.shuffle(order)  # this is the true order (left to right)

        # Generate clues
        clues = []
        for i in range(len(order) - 1):
            if rng.random() > 0.3:
                clues.append(f"{order[i]} is to the left of {order[i+1]}.")
        for i in range(len(order)):
            for j in range(i+2, len(order)):
                if rng.random() > 0.7:
                    clues.append(f"{order[i]} is not immediately next to {order[j]}.")

        if not clues:
            clues.append(f"{order[0]} is leftmost.")

        rng.shuffle(clues)
        clue_str = "\n".join(f"- {c}" for c in clues)

        return {
            "question": f"Given these clues about the left-to-right order of {n} people:\n{clue_str}\n\nWhat is the correct left-to-right order?",
            "answer": " ".join(order),
            "metadata": {"order": order, "clues": clues},
        }

    def check_answer(self, question: str, answer: str, metadata: dict = None) -> bool:
        if not metadata:
            return False
        order = metadata.get("order", [])
        answer_lower = answer.lower()
        # Check all names appear in correct relative order
        positions = {}
        for name in order:
            idx = answer_lower.find(name.lower())
            if idx == -1:
                return False
            positions[name] = idx
        return all(positions[order[i]] < positions[order[i+1]] for i in range(len(order)-1))
