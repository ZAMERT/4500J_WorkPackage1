import os
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI


DEFAULT_LLM_MODEL = "deepseek-v4-flash"
DEFAULT_THINKING = "enabled"
DEFAULT_REASONING_EFFORT = "high"


@dataclass
class LLMResult:
    content: str
    reasoning_content: Optional[str]
    model: str
    requested_model: str
    thinking: str
    reasoning_effort: str


def normalize_deepseek_options(model: str, thinking: str) -> tuple[str, str]:
    """Map deprecated compatibility model names to DeepSeek v4 options."""
    if model == "deepseek-chat":
        return "deepseek-v4-flash", "disabled"
    if model == "deepseek-reasoner":
        return "deepseek-v4-flash", "enabled"
    return model, thinking


class DeepSeekLLM:
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.deepseek.com"):
        api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set in the environment or .env file.")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(
        self,
        prompt: str,
        model: str = DEFAULT_LLM_MODEL,
        thinking: str = DEFAULT_THINKING,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    ) -> LLMResult:
        resolved_model, resolved_thinking = normalize_deepseek_options(model, thinking)
        extra_body = {"thinking": {"type": resolved_thinking}}
        if resolved_thinking == "enabled":
            extra_body["reasoning_effort"] = reasoning_effort

        response = self.client.chat.completions.create(
            model=resolved_model,
            messages=[
                {
                    "role": "system",
                    "content": "You generate ABB RAPID code. Be precise and avoid hallucinating APIs.",
                },
                {"role": "user", "content": prompt},
            ],
            extra_body=extra_body,
        )
        message = response.choices[0].message
        return LLMResult(
            content=message.content or "",
            reasoning_content=getattr(message, "reasoning_content", None),
            model=resolved_model,
            requested_model=model,
            thinking=resolved_thinking,
            reasoning_effort=reasoning_effort,
        )
