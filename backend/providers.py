"""
WorldLens — Multi-Provider Manager
====================================
Manages LLM/VLM providers with user-selectable preference and automatic
fallback when a provider hits rate limits or errors.

Providers fall into two categories:
  1. **Primary LLM** — powers the main real-time agent conversation.
     Only Gemini Realtime supports this today (voice + vision streaming).
  2. **Supplementary VLMs** — provide additional vision intelligence
     (scene captioning, OCR, detailed VQA).  These include:
       • Gemini VLM   (Google)
       • HuggingFace VLM (Inference API)
       • NVIDIA VLM   (NGC / Cosmos)
       • Grok VLM     (xAI — custom adapter, OpenAI-compat)
       • Azure OpenAI  (GPT-4o — custom adapter)

The fallback chain works as follows:
  1. Try the user's preferred provider.
  2. On failure (rate-limit, network, auth), log the error, record the
     failure event, and try the next provider in the chain.
  3. Expose status via API so the frontend can show toast notifications.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import httpx

logger = logging.getLogger("worldlens.providers")


# ---------------------------------------------------------------------------
# Provider identifiers
# ---------------------------------------------------------------------------
class ProviderID(str, Enum):
    GEMINI = "gemini"
    HUGGINGFACE = "huggingface"
    NVIDIA = "nvidia"
    GROK = "grok"
    AZURE_OPENAI = "azure_openai"


# ---------------------------------------------------------------------------
# Status tracking
# ---------------------------------------------------------------------------
@dataclass
class ProviderStatus:
    """Runtime health status of a single provider."""

    provider: ProviderID
    available: bool = True
    last_error: str | None = None
    last_error_time: float | None = None
    total_calls: int = 0
    total_errors: int = 0
    cooldown_until: float = 0.0  # Unix timestamp — skip provider until then

    @property
    def in_cooldown(self) -> bool:
        return time.time() < self.cooldown_until

    def record_success(self) -> None:
        self.total_calls += 1
        self.available = True

    def record_error(self, error: str, cooldown_seconds: float = 60.0) -> None:
        self.total_calls += 1
        self.total_errors += 1
        self.last_error = error
        self.last_error_time = time.time()
        self.cooldown_until = time.time() + cooldown_seconds
        logger.warning(
            "Provider %s failed: %s (cooldown %ds)",
            self.provider.value,
            error,
            cooldown_seconds,
        )

    def to_dict(self) -> dict:
        return {
            "provider": self.provider.value,
            "available": self.available and not self.in_cooldown,
            "in_cooldown": self.in_cooldown,
            "last_error": self.last_error,
            "total_calls": self.total_calls,
            "total_errors": self.total_errors,
        }


# ---------------------------------------------------------------------------
# Fallback event (sent to frontend via API polling)
# ---------------------------------------------------------------------------
@dataclass
class FallbackEvent:
    """Records a single provider fallback for the frontend toast."""

    original_provider: str
    fallback_provider: str
    error_reason: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "original": self.original_provider,
            "fallback": self.fallback_provider,
            "reason": self.error_reason,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Provider adapters — each wraps a different VLM API
# ---------------------------------------------------------------------------
class _BaseAdapter:
    """Common interface every VLM adapter exposes."""

    provider_id: ProviderID

    async def caption(self, image_bytes: bytes, prompt: str = "Describe this image.") -> str:
        raise NotImplementedError

    async def health_check(self) -> bool:
        """Quick check that the provider credentials work."""
        return True


class GeminiAdapter(_BaseAdapter):
    """Wraps ``vision_agents.plugins.gemini.VLM`` for single-frame captioning."""

    provider_id = ProviderID.GEMINI

    @property
    def _api_key(self) -> str:
        return os.getenv("GOOGLE_API_KEY", "")

    async def caption(self, image_bytes: bytes, prompt: str = "Describe this image.") -> str:
        async with httpx.AsyncClient(timeout=30.0) as client:
            b64 = base64.b64encode(image_bytes).decode()
            resp = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview:generateContent?key={self._api_key}",
                json={
                    "contents": [
                        {
                            "parts": [
                                {"text": prompt},
                                {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                            ]
                        }
                    ]
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]

    async def health_check(self) -> bool:
        return bool(self._api_key)


class HuggingFaceAdapter(_BaseAdapter):
    """Wraps the HuggingFace Inference API for image captioning."""

    provider_id = ProviderID.HUGGINGFACE

    def __init__(self, model: str = "Salesforce/blip-image-captioning-large") -> None:
        self._model = model

    @property
    def _api_key(self) -> str:
        return os.getenv("HF_API_TOKEN", os.getenv("HF_TOKEN", ""))

    async def caption(self, image_bytes: bytes, prompt: str = "Describe this image.") -> str:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"https://api-inference.huggingface.co/models/{self._model}",
                headers={"Authorization": f"Bearer {self._api_key}"},
                content=image_bytes,
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and data:
                return data[0].get("generated_text", "")
            return str(data)

    async def health_check(self) -> bool:
        return bool(self._api_key)


class NvidiaAdapter(_BaseAdapter):
    """Wraps the NVIDIA NGC / Cosmos VLM API."""

    provider_id = ProviderID.NVIDIA

    def __init__(self, model: str = "nvidia/cosmos-reason2-8b") -> None:
        self._model = model

    @property
    def _api_key(self) -> str:
        return os.getenv("NGC_API_KEY", os.getenv("NVIDIA_API_KEY", ""))

    async def caption(self, image_bytes: bytes, prompt: str = "Describe this image.") -> str:
        b64 = base64.b64encode(image_bytes).decode()
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                                },
                            ],
                        }
                    ],
                    "max_tokens": 512,
                    "temperature": 0.2,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    async def health_check(self) -> bool:
        return bool(self._api_key)


class GrokAdapter(_BaseAdapter):
    """Wraps the xAI Grok API (OpenAI-compatible) for vision tasks."""

    provider_id = ProviderID.GROK

    def __init__(self, model: str = "grok-4-latest") -> None:
        self._model = model

    @property
    def _api_key(self) -> str:
        return os.getenv("XAI_API_KEY", "")

    async def caption(self, image_bytes: bytes, prompt: str = "Describe this image.") -> str:
        b64 = base64.b64encode(image_bytes).decode()
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.x.ai/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
                json={
                    "model": self._model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                                },
                            ],
                        }
                    ],
                    "max_tokens": 512,
                    "temperature": 0,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    async def health_check(self) -> bool:
        return bool(self._api_key)


class AzureOpenAIAdapter(_BaseAdapter):
    """Wraps Azure OpenAI GPT-4o for vision tasks."""

    provider_id = ProviderID.AZURE_OPENAI

    @property
    def _endpoint(self) -> str:
        return os.getenv("AZURE_OPENAI_ENDPOINT", "")

    @property
    def _api_key(self) -> str:
        return os.getenv("AZURE_OPENAI_API_KEY", "")

    async def caption(self, image_bytes: bytes, prompt: str = "Describe this image.") -> str:
        b64 = base64.b64encode(image_bytes).decode()
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                self._endpoint,
                headers={
                    "Content-Type": "application/json",
                    "api-key": self._api_key,
                },
                json={
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                                },
                            ],
                        }
                    ],
                    "max_tokens": 512,
                    "temperature": 0.2,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    async def health_check(self) -> bool:
        return bool(self._endpoint and self._api_key)


# ---------------------------------------------------------------------------
# Provider Manager — singleton that orchestrates fallback
# ---------------------------------------------------------------------------
class ProviderManager:
    """Central registry managing VLM provider preference and fallback."""

    # Default fallback order
    DEFAULT_CHAIN: list[ProviderID] = [
        ProviderID.GEMINI,
        ProviderID.GROK,
        ProviderID.AZURE_OPENAI,
        ProviderID.NVIDIA,
        ProviderID.HUGGINGFACE,
    ]

    def __init__(self) -> None:
        self._adapters: dict[ProviderID, _BaseAdapter] = {}
        self._status: dict[ProviderID, ProviderStatus] = {}
        self._preferred: ProviderID = ProviderID.GEMINI
        self._fallback_events: list[FallbackEvent] = []
        self._lock = asyncio.Lock()

        # Register all adapters
        for adapter_cls in (
            GeminiAdapter,
            HuggingFaceAdapter,
            NvidiaAdapter,
            GrokAdapter,
            AzureOpenAIAdapter,
        ):
            adapter = adapter_cls()
            self._adapters[adapter.provider_id] = adapter
            self._status[adapter.provider_id] = ProviderStatus(
                provider=adapter.provider_id
            )

    # ---------- public API ----------

    @property
    def preferred(self) -> ProviderID:
        return self._preferred

    def set_preferred(self, provider_id: str) -> bool:
        """Set the preferred provider. Returns False if unknown."""
        try:
            self._preferred = ProviderID(provider_id)
            logger.info("Preferred VLM provider set to %s", provider_id)
            return True
        except ValueError:
            return False

    def get_fallback_chain(self) -> list[ProviderID]:
        """Return the current fallback order, with the preferred provider first."""
        chain = [self._preferred]
        for pid in self.DEFAULT_CHAIN:
            if pid != self._preferred:
                chain.append(pid)
        return chain

    async def caption(
        self, image_bytes: bytes, prompt: str = "Describe this image."
    ) -> tuple[str, ProviderID]:
        """Run VLM captioning through the fallback chain.

        Returns:
            ``(caption_text, provider_that_succeeded)``

        Raises:
            RuntimeError: if all providers fail.
        """
        chain = self.get_fallback_chain()
        errors: list[str] = []

        for pid in chain:
            status = self._status[pid]
            adapter = self._adapters[pid]

            # Skip if no API key configured
            if not await adapter.health_check():
                continue

            # Skip if in cooldown
            if status.in_cooldown:
                continue

            try:
                result = await adapter.caption(image_bytes, prompt)
                status.record_success()

                # If we fell back, record the event
                if pid != self._preferred:
                    evt = FallbackEvent(
                        original_provider=self._preferred.value,
                        fallback_provider=pid.value,
                        error_reason=errors[-1] if errors else "unknown",
                    )
                    async with self._lock:
                        self._fallback_events.append(evt)
                    logger.info(
                        "Fallback: %s → %s", self._preferred.value, pid.value
                    )

                return result, pid

            except httpx.HTTPStatusError as exc:
                error_msg = f"HTTP {exc.response.status_code}"
                if exc.response.status_code == 429:
                    error_msg = "Rate limit exceeded"
                    status.record_error(error_msg, cooldown_seconds=120.0)
                elif exc.response.status_code in (401, 403):
                    error_msg = "Authentication failed"
                    status.record_error(error_msg, cooldown_seconds=300.0)
                else:
                    status.record_error(error_msg, cooldown_seconds=30.0)
                errors.append(f"{pid.value}: {error_msg}")

            except Exception as exc:
                error_msg = str(exc)[:200]
                status.record_error(error_msg, cooldown_seconds=30.0)
                errors.append(f"{pid.value}: {error_msg}")

        raise RuntimeError(
            f"All VLM providers failed: {'; '.join(errors)}"
        )

    def get_status(self) -> dict:
        """Return status of all providers for the API."""
        return {
            "preferred": self._preferred.value,
            "providers": {
                pid.value: self._status[pid].to_dict()
                for pid in ProviderID
            },
            "fallback_chain": [p.value for p in self.get_fallback_chain()],
        }

    def pop_fallback_events(self) -> list[dict]:
        """Return and clear pending fallback events (for frontend polling)."""
        events = [e.to_dict() for e in self._fallback_events]
        self._fallback_events.clear()
        return events

    async def check_all_providers(self) -> dict[str, bool]:
        """Run health checks on all providers."""
        results: dict[str, bool] = {}
        for pid, adapter in self._adapters.items():
            ok = await adapter.health_check()
            self._status[pid].available = ok
            results[pid.value] = ok
        return results


# Module-level singleton
provider_manager = ProviderManager()
