"""
TaskLLM wrapper.
Supports OpenAI, MiniMax (OpenAI-compatible), Anthropic, and local vLLM.
"""

from __future__ import annotations


class TaskLLM:
    """Wrapper for the large TaskLLM that solves tasks."""

    def __init__(self, config: dict):
        self.provider = config.get("provider", "openai")
        self.model = config.get("model", "gpt-4o")
        self.api_key = config.get("api_key", "")
        self.base_url = config.get("base_url", "") or None
        self._client = None
        self._init_client()

    def _init_client(self):
        if self.provider in ("openai", "minimax", "local"):
            from openai import OpenAI
            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        elif self.provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)

    def solve(
        self,
        task: str,
        guidance: str,
        guidance_type: str,  # "CODE" | "TEXT"
        history: list[dict] | None = None,
        temperature: float = 0.0,
    ) -> str:
        """
        Generate a solution given guidance.
        Returns raw response text (may include code blocks).
        """
        messages = self._build_messages(task, guidance, guidance_type, history)

        if self.provider in ("openai", "minimax", "local"):
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=2048,
            )
            return response.choices[0].message.content

        elif self.provider == "anthropic":
            response = self._client.messages.create(
                model=self.model,
                messages=messages,
                max_tokens=2048,
            )
            return response.content[0].text

        raise ValueError(f"Unknown provider: {self.provider}")

    def _build_messages(
        self,
        task: str,
        guidance: str,
        guidance_type: str,
        history: list[dict] | None,
    ) -> list[dict]:
        system = (
            "You are an expert problem solver. You will receive a task and guidance "
            "on whether to use code (programmatic computation) or text (reasoning). "
            "Follow the guidance carefully.\n\n"
            "If asked to use CODE: write Python code in a ```python``` block. "
            "Make the code actually compute the answer algorithmically — do NOT hardcode answers.\n"
            "If asked to use TEXT: reason step by step and give a clear final answer."
        )

        messages = [{"role": "system", "content": system}]

        # Add history as alternating user/assistant turns
        for h in (history or []):
            messages.append({"role": "user", "content": h.get("user", "")})
            messages.append({"role": "assistant", "content": h.get("assistant", "")})

        # Current turn
        user_content = f"Task: {task}\n\nGuidance ({guidance_type}): {guidance}\n\nProvide your solution:"
        messages.append({"role": "user", "content": user_content})

        return messages
