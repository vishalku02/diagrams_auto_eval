"""Model factory for multi-provider VLM judge evaluations.

Supports OpenAI, Anthropic, Google, and Together (open-source) APIs.
Specify any model from the CLI and the factory routes to the right provider.

Usage:
    from utils.models import get_judge, list_providers

    judge = get_judge("claude-sonnet-4-6")
    result = await judge.call(prompt=..., image_bytes=..., response_format=...)
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Registry: model name prefix -> provider
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, str] = {
    # OpenAI
    "gpt-": "openai",
    "o3": "openai",
    "o4": "openai",
    # Anthropic
    "claude-": "anthropic",
    # Google
    "gemini-": "google",
}

# Anything not matched above goes to Together (open-source catch-all)
DEFAULT_PROVIDER = "together"


def detect_provider(model_name: str) -> str:
    """Detect provider from model name prefix."""
    name = model_name.strip().lower()
    for prefix, provider in MODEL_REGISTRY.items():
        if name.startswith(prefix):
            return provider
    return DEFAULT_PROVIDER


# ---------------------------------------------------------------------------
# Base judge interface
# ---------------------------------------------------------------------------
@dataclass
class JudgeResponse:
    text: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class BaseJudge(ABC):
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name

    @abstractmethod
    async def call(
        self,
        *,
        prompt: str,
        image_bytes: Optional[bytes] = None,
        image_media_type: str = "image/png",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: Optional[dict] = None,
    ) -> JudgeResponse:
        ...


# ---------------------------------------------------------------------------
# OpenAI Judge
# ---------------------------------------------------------------------------
class OpenAIJudge(BaseJudge):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name)
        from openai import OpenAI
        self._client = OpenAI()

    async def call(self, *, prompt, image_bytes=None, image_media_type="image/png",
                   temperature=0.0, max_tokens=4096, response_format=None) -> JudgeResponse:
        content = [{"type": "text", "text": prompt}]
        if image_bytes:
            b64 = base64.b64encode(image_bytes).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{image_media_type};base64,{b64}"}
            })

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format:
            kwargs["response_format"] = response_format

        def _sync_call():
            return self._client.chat.completions.create(**kwargs)

        resp = await asyncio.to_thread(_sync_call)
        choice = resp.choices[0]
        usage = resp.usage

        return JudgeResponse(
            text=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
        )


# ---------------------------------------------------------------------------
# Anthropic Judge
# ---------------------------------------------------------------------------
class AnthropicJudge(BaseJudge):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name)
        import anthropic
        self._client = anthropic.Anthropic()

    async def call(self, *, prompt, image_bytes=None, image_media_type="image/png",
                   temperature=0.0, max_tokens=4096, response_format=None) -> JudgeResponse:
        content = []
        if image_bytes:
            b64 = base64.b64encode(image_bytes).decode()
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": image_media_type, "data": b64}
            })
        content.append({"type": "text", "text": prompt})

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        def _sync_call():
            return self._client.messages.create(**kwargs)

        resp = await asyncio.to_thread(_sync_call)
        text = "".join(block.text for block in resp.content if hasattr(block, "text"))

        return JudgeResponse(
            text=text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            cached_tokens=getattr(resp.usage, "cache_read_input_tokens", None),
            total_tokens=resp.usage.input_tokens + resp.usage.output_tokens,
        )


# ---------------------------------------------------------------------------
# Google Gemini Judge
# ---------------------------------------------------------------------------
class GoogleJudge(BaseJudge):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name)
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        self._model = genai.GenerativeModel(model_name)

    async def call(self, *, prompt, image_bytes=None, image_media_type="image/png",
                   temperature=0.0, max_tokens=4096, response_format=None) -> JudgeResponse:
        import google.generativeai as genai
        from PIL import Image
        import io

        parts = []
        if image_bytes:
            img = Image.open(io.BytesIO(image_bytes))
            parts.append(img)
        parts.append(prompt)

        def _sync_call():
            return self._model.generate_content(
                parts,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )

        resp = await asyncio.to_thread(_sync_call)
        text = resp.text or ""
        usage_meta = getattr(resp, "usage_metadata", None)

        return JudgeResponse(
            text=text,
            input_tokens=getattr(usage_meta, "prompt_token_count", None),
            output_tokens=getattr(usage_meta, "candidates_token_count", None),
            total_tokens=getattr(usage_meta, "total_token_count", None),
        )


# ---------------------------------------------------------------------------
# Together API Judge (open-source models)
# ---------------------------------------------------------------------------
class TogetherJudge(BaseJudge):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name)
        from openai import OpenAI
        self._client = OpenAI(
            api_key=os.environ.get("TOGETHER_API_KEY"),
            base_url="https://api.together.xyz/v1",
        )

    async def call(self, *, prompt, image_bytes=None, image_media_type="image/png",
                   temperature=0.0, max_tokens=4096, response_format=None) -> JudgeResponse:
        content = [{"type": "text", "text": prompt}]
        if image_bytes:
            b64 = base64.b64encode(image_bytes).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{image_media_type};base64,{b64}"}
            })

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format:
            kwargs["response_format"] = response_format

        def _sync_call():
            return self._client.chat.completions.create(**kwargs)

        resp = await asyncio.to_thread(_sync_call)
        choice = resp.choices[0]
        usage = resp.usage

        return JudgeResponse(
            text=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
PROVIDER_MAP: dict[str, type[BaseJudge]] = {
    "openai": OpenAIJudge,
    "anthropic": AnthropicJudge,
    "google": GoogleJudge,
    "together": TogetherJudge,
}


def get_judge(model_name: str, **kwargs) -> BaseJudge:
    """Factory: returns the right judge for the given model name."""
    provider = detect_provider(model_name)
    judge_cls = PROVIDER_MAP.get(provider)
    if judge_cls is None:
        raise ValueError(f"No judge implementation for provider '{provider}' (model: {model_name})")
    return judge_cls(model_name, **kwargs)


def list_providers() -> list[str]:
    return list(PROVIDER_MAP.keys())
