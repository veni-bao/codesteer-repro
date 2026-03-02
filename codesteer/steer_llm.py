"""
CodeSteerLLM wrapper.
Supports:
  - Local HuggingFace model (Llama-3.1-8B fine-tuned)
  - OpenAI-compatible API (vLLM, hosted, etc.)
"""

from __future__ import annotations
import re
from typing import Optional


FIRST_TURN_PROMPT = """You are a code/text guidance assistant. Given a task question, decide whether the TaskLLM should solve it using CODE (programmatic/symbolic computation) or TEXT (textual reasoning).

Task: {task}

Current answer from TaskLLM (if any): {current_answer}

Guidance history so far: {history}

Provide your guidance in this format:
GUIDANCE_TYPE: [CODE|TEXT|FINALIZE]
GUIDANCE: <specific instructions for the TaskLLM>

If the answer looks correct and complete, use FINALIZE."""

SUBSEQUENT_TURN_PROMPT = """You are a code/text guidance assistant. Review the TaskLLM's latest answer and history, then provide refined guidance.

Task: {task}
Turn: {turn}/{max_turns}
Latest answer: {current_answer}
Guidance history: {history}

Has the code been verified by Symbolic Checker? {symbolic_result}
Has the answer been verified by Self-answer Checker? {selfanswer_result}

Provide updated guidance:
GUIDANCE_TYPE: [CODE|TEXT|FINALIZE|SWITCH]
GUIDANCE: <specific instructions>"""


class CodeSteerLLM:
    """Wrapper for CodeSteerLLM (local or API-based)."""

    def __init__(self, config: dict):
        self.config = config
        self.mode = config.get("mode", "local")
        self._model = None
        self._tokenizer = None
        self._client = None

        if self.mode == "local":
            self._load_local_model()
        else:
            self._init_api_client()

    def _load_local_model(self):
        """Load fine-tuned Llama-3.1-8B from HuggingFace."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch

        hf_model = self.config.get("hf_model", "yongchao98/CodeSteer-v1.0")
        device = self.config.get("device", "cuda")
        quantize = self.config.get("quantize", None)

        print(f"Loading CodeSteerLLM from {hf_model} (quantize={quantize})...")

        kwargs = {}
        if quantize == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        elif quantize == "8bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kwargs["torch_dtype"] = torch.float16

        max_memory = self.config.get("max_memory", None)
        if max_memory:
            kwargs["max_memory"] = max_memory

        self._tokenizer = AutoTokenizer.from_pretrained(hf_model)
        self._model = AutoModelForCausalLM.from_pretrained(
            hf_model,
            device_map=device if quantize else "auto",
            **kwargs,
        )
        print("CodeSteerLLM loaded.")

    def _init_api_client(self):
        """Initialize OpenAI-compatible API client."""
        from openai import OpenAI

        self._client = OpenAI(
            api_key=self.config.get("api_key", "none"),
            base_url=self.config.get("base_url", ""),
        )
        self._api_model = self.config.get("model", "CodeSteer-v1.0")

    def _generate_local(self, prompt: str, temperature: float = 0.7) -> str:
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        response = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return response

    def _generate_api(self, prompt: str, temperature: float = 0.7) -> str:
        response = self._client.chat.completions.create(
            model=self._api_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=512,
        )
        return response.choices[0].message.content

    def guide(
        self,
        task: str,
        turn: int,
        max_turns: int,
        current_answer: str = "",
        history: list[dict] | None = None,
        symbolic_result: str = "N/A",
        selfanswer_result: str = "N/A",
        temperature: float = 0.7,
    ) -> dict:
        """
        Generate guidance for the TaskLLM.
        Returns: {"type": "CODE"|"TEXT"|"FINALIZE"|"SWITCH", "guidance": str}
        """
        history_str = "\n".join(
            [f"Turn {h['turn']}: {h['type']} - {h['guidance'][:100]}" for h in (history or [])]
        ) or "None"

        if turn == 1:
            prompt = FIRST_TURN_PROMPT.format(
                task=task,
                current_answer=current_answer or "No answer yet",
                history=history_str,
            )
        else:
            prompt = SUBSEQUENT_TURN_PROMPT.format(
                task=task,
                turn=turn,
                max_turns=max_turns,
                current_answer=current_answer,
                history=history_str,
                symbolic_result=symbolic_result,
                selfanswer_result=selfanswer_result,
            )

        if self.mode == "local":
            raw = self._generate_local(prompt, temperature)
        else:
            raw = self._generate_api(prompt, temperature)

        return self._parse_guidance(raw)

    def _parse_guidance(self, raw: str) -> dict:
        """Parse GUIDANCE_TYPE and GUIDANCE from raw output."""
        type_match = re.search(r"GUIDANCE_TYPE:\s*(CODE|TEXT|FINALIZE|SWITCH)", raw, re.IGNORECASE)
        guidance_match = re.search(r"GUIDANCE:\s*(.+?)(?=\n[A-Z_]+:|$)", raw, re.DOTALL)

        g_type = type_match.group(1).upper() if type_match else "TEXT"
        guidance = guidance_match.group(1).strip() if guidance_match else raw.strip()[:300]

        return {"type": g_type, "guidance": guidance}
