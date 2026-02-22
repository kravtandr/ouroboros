"""
Ouroboros — LLM client.

The only module that communicates with the LLM API (OpenRouter).
Contract: chat(), default_model(), available_models(), add_usage().

Also provides LocalModelRouter for routing simple tasks to local LM Studio.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

DEFAULT_LIGHT_MODEL = "google/gemini-3.1-pro-preview"


def normalize_reasoning_effort(value: str, default: str = "medium") -> str:
    allowed = {"none", "minimal", "low", "medium", "high", "xhigh"}
    v = str(value or "").strip().lower()
    return v if v in allowed else default


def reasoning_rank(value: str) -> int:
    order = {"none": 0, "minimal": 1, "low": 2, "medium": 3, "high": 4, "xhigh": 5}
    return int(order.get(str(value or "").strip().lower(), 3))


def add_usage(total: Dict[str, Any], usage: Dict[str, Any]) -> None:
    """Accumulate usage from one LLM call into a running total."""
    for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "cache_write_tokens"):
        total[k] = int(total.get(k) or 0) + int(usage.get(k) or 0)
    if usage.get("cost"):
        total["cost"] = float(total.get("cost") or 0) + float(usage["cost"])


def fetch_openrouter_pricing() -> Dict[str, Tuple[float, float, float]]:
    """
    Fetch current pricing from OpenRouter API.

    Returns dict of {model_id: (input_per_1m, cached_per_1m, output_per_1m)}.
    Returns empty dict on failure.
    """
    import logging
    log = logging.getLogger("ouroboros.llm")

    try:
        import requests
    except ImportError:
        log.warning("requests not installed, cannot fetch pricing")
        return {}

    try:
        url = "https://openrouter.ai/api/v1/models"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()

        data = resp.json()
        models = data.get("data", [])

        # Prefixes we care about
        prefixes = ("anthropic/", "openai/", "google/", "meta-llama/", "x-ai/", "qwen/")

        pricing_dict = {}
        for model in models:
            model_id = model.get("id", "")
            if not model_id.startswith(prefixes):
                continue

            pricing = model.get("pricing", {})
            if not pricing or not pricing.get("prompt"):
                continue

            # OpenRouter pricing is in dollars per token (raw values)
            raw_prompt = float(pricing.get("prompt", 0))
            raw_completion = float(pricing.get("completion", 0))
            raw_cached_str = pricing.get("input_cache_read")
            raw_cached = float(raw_cached_str) if raw_cached_str else None

            # Convert to per-million tokens
            prompt_price = round(raw_prompt * 1_000_000, 4)
            completion_price = round(raw_completion * 1_000_000, 4)
            if raw_cached is not None:
                cached_price = round(raw_cached * 1_000_000, 4)
            else:
                cached_price = round(prompt_price * 0.1, 4)  # fallback: 10% of prompt

            # Sanity check: skip obviously wrong prices
            if prompt_price > 1000 or completion_price > 1000:
                log.warning(f"Skipping {model_id}: prices seem wrong (prompt={prompt_price}, completion={completion_price})")
                continue

            pricing_dict[model_id] = (prompt_price, cached_price, completion_price)

        log.info(f"Fetched pricing for {len(pricing_dict)} models from OpenRouter")
        return pricing_dict

    except (requests.RequestException, ValueError, KeyError) as e:
        log.warning(f"Failed to fetch OpenRouter pricing: {e}")
        return {}


class LLMClient:
    """OpenRouter API wrapper. All LLM calls go through this class."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self._base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
                default_headers={
                    "HTTP-Referer": "https://colab.research.google.com/",
                    "X-Title": "Ouroboros",
                },
            )
        return self._client

    def _fetch_generation_cost(self, generation_id: str) -> Optional[float]:
        """Fetch cost from OpenRouter Generation API as fallback."""
        try:
            import requests
            url = f"{self._base_url.rstrip('/')}/generation?id={generation_id}"
            resp = requests.get(url, headers={"Authorization": f"Bearer {self._api_key}"}, timeout=5)
            if resp.status_code == 200:
                data = resp.json().get("data") or {}
                cost = data.get("total_cost") or data.get("usage", {}).get("cost")
                if cost is not None:
                    return float(cost)
            # Generation might not be ready yet — retry once after short delay
            time.sleep(0.5)
            resp = requests.get(url, headers={"Authorization": f"Bearer {self._api_key}"}, timeout=5)
            if resp.status_code == 200:
                data = resp.json().get("data") or {}
                cost = data.get("total_cost") or data.get("usage", {}).get("cost")
                if cost is not None:
                    return float(cost)
        except Exception:
            log.debug("Failed to fetch generation cost from OpenRouter", exc_info=True)
            pass
        return None

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        reasoning_effort: str = "medium",
        max_tokens: int = 16384,
        tool_choice: str = "auto",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Single LLM call. Returns: (response_message_dict, usage_dict with cost)."""
        client = self._get_client()
        effort = normalize_reasoning_effort(reasoning_effort)

        extra_body: Dict[str, Any] = {
            "reasoning": {"effort": effort, "exclude": True},
        }

        # Pin Anthropic models to Anthropic provider for prompt caching
        if model.startswith("anthropic/"):
            extra_body["provider"] = {
                "order": ["Anthropic"],
                "allow_fallbacks": False,
                "require_parameters": True,
            }

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "extra_body": extra_body,
        }
        if tools:
            # Add cache_control to last tool for Anthropic prompt caching
            # This caches all tool schemas (they never change between calls)
            tools_with_cache = [t for t in tools]  # shallow copy
            if tools_with_cache:
                last_tool = {**tools_with_cache[-1]}  # copy last tool
                last_tool["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
                tools_with_cache[-1] = last_tool
            kwargs["tools"] = tools_with_cache
            kwargs["tool_choice"] = tool_choice

        resp = client.chat.completions.create(**kwargs)
        resp_dict = resp.model_dump()
        usage = resp_dict.get("usage") or {}
        choices = resp_dict.get("choices") or [{}]
        msg = (choices[0] if choices else {}).get("message") or {}

        # Extract cached_tokens from prompt_tokens_details if available
        if not usage.get("cached_tokens"):
            prompt_details = usage.get("prompt_tokens_details") or {}
            if isinstance(prompt_details, dict) and prompt_details.get("cached_tokens"):
                usage["cached_tokens"] = int(prompt_details["cached_tokens"])

        # Extract cache_write_tokens from prompt_tokens_details if available
        # OpenRouter: "cache_write_tokens"
        # Native Anthropic: "cache_creation_tokens" or "cache_creation_input_tokens"
        if not usage.get("cache_write_tokens"):
            prompt_details_for_write = usage.get("prompt_tokens_details") or {}
            if isinstance(prompt_details_for_write, dict):
                cache_write = (prompt_details_for_write.get("cache_write_tokens")
                              or prompt_details_for_write.get("cache_creation_tokens")
                              or prompt_details_for_write.get("cache_creation_input_tokens"))
                if cache_write:
                    usage["cache_write_tokens"] = int(cache_write)

        # Ensure cost is present in usage (OpenRouter includes it, but fallback if missing)
        if not usage.get("cost"):
            gen_id = resp_dict.get("id") or ""
            if gen_id:
                cost = self._fetch_generation_cost(gen_id)
                if cost is not None:
                    usage["cost"] = cost

        return msg, usage

    def vision_query(
        self,
        prompt: str,
        images: List[Dict[str, Any]],
        model: str = "anthropic/claude-sonnet-4.6",
        max_tokens: int = 1024,
        reasoning_effort: str = "low",
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Send a vision query to an LLM. Lightweight — no tools, no loop.

        Args:
            prompt: Text instruction for the model
            images: List of image dicts. Each dict must have either:
                - {"url": "https://..."} — for URL images
                - {"base64": "<b64>", "mime": "image/png"} — for base64 images
            model: VLM-capable model ID
            max_tokens: Max response tokens
            reasoning_effort: Effort level

        Returns:
            (text_response, usage_dict)
        """
        # Build multipart content
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in images:
            if "url" in img:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img["url"]},
                })
            elif "base64" in img:
                mime = img.get("mime", "image/png")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{img['base64']}"},
                })
            else:
                log.warning("vision_query: skipping image with unknown format: %s", list(img.keys()))

        messages = [{"role": "user", "content": content}]
        response_msg, usage = self.chat(
            messages=messages,
            model=model,
            tools=None,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
        )
        text = response_msg.get("content") or ""
        return text, usage

    def default_model(self) -> str:
        """Return the single default model from env. LLM switches via tool if needed."""
        return os.environ.get("OUROBOROS_MODEL", "anthropic/claude-sonnet-4.6")

    def available_models(self) -> List[str]:
        """Return list of available models from env (for switch_model tool schema)."""
        main = os.environ.get("OUROBOROS_MODEL", "anthropic/claude-sonnet-4.6")
        code = os.environ.get("OUROBOROS_MODEL_CODE", "")
        light = os.environ.get("OUROBOROS_MODEL_LIGHT", "")
        models = [main]
        if code and code != main:
            models.append(code)
        if light and light != main and light != code:
            models.append(light)
        return models


# ---------------------------------------------------------------------------
# Local LLM Client (for LM Studio / OpenAI-compatible local APIs)
# ---------------------------------------------------------------------------

class LocalLLMClient(LLMClient):
    """
    OpenAI-compatible client for local LM Studio.

    Differences from LLMClient:
    - No OpenRouter-specific extra_body (no reasoning hints, no provider pinning)
    - No cost tracking (local = free)
    - No cache_control headers on tools (local APIs don't support it)
    - Uses a dummy API key ("local")
    """

    def __init__(self, base_url: str = "http://localhost:1234/v1"):
        # Use "local" as dummy key — LM Studio doesn't require auth
        super().__init__(api_key="local", base_url=base_url)

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            # No special headers needed for local API
            self._client = OpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
            )
        return self._client

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        reasoning_effort: str = "medium",
        max_tokens: int = 8192,
        tool_choice: str = "auto",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Simple OpenAI-compatible chat call — no OpenRouter extensions.
        Returns (message_dict, usage_dict). Cost is always 0 (local = free).
        """
        client = self._get_client()

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        # Local models may support tools, but without cache_control
        if tools:
            # Strip any cache_control from tools (local APIs don't support it)
            clean_tools = []
            for t in tools:
                t_clean = {k: v for k, v in t.items() if k != "cache_control"}
                clean_tools.append(t_clean)
            kwargs["tools"] = clean_tools
            kwargs["tool_choice"] = tool_choice

        resp = client.chat.completions.create(**kwargs)
        resp_dict = resp.model_dump()
        usage = resp_dict.get("usage") or {}
        choices = resp_dict.get("choices") or [{}]
        msg = (choices[0] if choices else {}).get("message") or {}

        # Capture reasoning_content from thinking models (Qwen, DeepSeek, etc.).
        # llama.cpp/LM Studio return it as a top-level key on the message object.
        # model_dump() already includes it if present — just ensure it stays in msg.
        reasoning = msg.get("reasoning_content") or ""
        if reasoning:
            log.debug("Local LLM reasoning (%d chars)", len(reasoning))
            # Expose reasoning token count in usage for monitoring
            usage["reasoning_tokens"] = len(reasoning.split())

        # Also check completion_tokens_details for native reasoning_tokens field
        ctd = usage.get("completion_tokens_details") or {}
        if isinstance(ctd, dict) and ctd.get("reasoning_tokens"):
            usage["reasoning_tokens"] = ctd["reasoning_tokens"]

        # Local calls are free — ensure cost is 0
        usage["cost"] = 0.0
        usage["local"] = True

        return msg, usage


# ---------------------------------------------------------------------------
# Local Model Router
# ---------------------------------------------------------------------------

# Tokens threshold above which we don't use local LLM (too slow / not capable)
_LOCAL_MAX_CONTEXT_CHARS = 800_000  # ~200k tokens — Qwen 400B has 262K context

# High-quality model keywords — never route these to local
_HIGH_QUALITY_KEYWORDS = ("opus", "/o3", "/o4", "o3-pro", "gemini-2.5-pro", "gemini-3-pro")


class LocalModelRouter:
    """
    Routes LLM calls between local LM Studio and OpenRouter.

    Rules:
    - If LOCAL_LLM_URL env var is not set → always use OpenRouter
    - If local is unavailable (checked with TTL cache) → use OpenRouter
    - If tools are present → use OpenRouter (tool calling needs capable models)
    - If model is a high-quality model (opus, o3, o4) → use OpenRouter
    - If context is large (>32k chars) → use OpenRouter
    - Otherwise → use local LM Studio

    Thread-safe. Health check cached for 30 seconds.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._available: Optional[bool] = None
        self._last_check: float = 0.0
        self._ttl: float = 30.0  # seconds
        self._local_client: Optional[LocalLLMClient] = None

    def _local_url(self) -> Optional[str]:
        """Returns LOCAL_LLM_URL or LOCAL_LLM_BASE_URL if set, else None."""
        url = os.environ.get("LOCAL_LLM_URL") or os.environ.get("LOCAL_LLM_BASE_URL")
        return url.strip().rstrip("/") if url else None

    def _check_health(self) -> bool:
        """Ping local API /models endpoint. Returns True if reachable."""
        url = self._local_url()
        if not url:
            return False
        try:
            import requests
            endpoint = url.rstrip("/") + "/models"
            resp = requests.get(endpoint, timeout=2)
            return resp.status_code == 200
        except Exception:
            return False

    def is_available(self) -> bool:
        """Check if local LLM is available (cached for TTL seconds)."""
        if not self._local_url():
            return False

        now = time.monotonic()
        with self._lock:
            if self._available is not None and (now - self._last_check) < self._ttl:
                return self._available
            # Cache expired or first check
            result = self._check_health()
            self._available = result
            self._last_check = now
            if result:
                log.info("Local LLM available at %s", self._local_url())
            else:
                log.debug("Local LLM not available at %s", self._local_url())
            return result

    def should_use_local(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """
        Decide whether to route this call to local LM Studio.

        Returns True only if:
        - Local is available
        - No tools required (tool calling → OpenRouter)
        - Model is not a high-quality/reasoning model
        - Context is not too large
        """
        if not self.is_available():
            return False

        # Local Qwen 400B supports tool calling natively — no restriction on tools

        # High-quality model explicitly requested
        model_lower = model.lower()
        if any(kw in model_lower for kw in _HIGH_QUALITY_KEYWORDS):
            return False

        # Estimate context size (rough char count across all messages)
        total_chars = sum(
            len(str(m.get("content") or ""))
            for m in messages
        )
        if total_chars > _LOCAL_MAX_CONTEXT_CHARS:
            return False

        return True

    def get_local_client(self) -> LocalLLMClient:
        """Get (or create) the local LLM client."""
        with self._lock:
            if self._local_client is None:
                url = self._local_url() or "http://localhost:1234/v1"
                self._local_client = LocalLLMClient(base_url=url)
            return self._local_client

    def get_local_model(self) -> str:
        """Return the local model name from env, default 'local'."""
        return os.environ.get("LOCAL_LLM_MODEL", "qwen-400b").strip() or "qwen-400b"

    def invalidate(self) -> None:
        """Force re-check on next call (e.g., after SSH tunnel reconnect)."""
        with self._lock:
            self._available = None
            self._last_check = 0.0


# Alias for backward compatibility (loop.py imports this name)
LocalLLMBackend = LocalLLMClient

# Module-level singleton — import this everywhere
local_router = LocalModelRouter()
