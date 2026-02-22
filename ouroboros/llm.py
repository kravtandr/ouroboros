"""
Ouroboros — LLM client.

The only module that communicates with the LLM API (OpenRouter or local backend).
Contract: chat(), default_model(), available_models(), add_usage().

Local backend support:
- LM Studio (or any OpenAI-compatible server) via LOCAL_LLM_URL env var
- Models prefixed with 'local/' are routed to the local backend
- Local backend is optional and gracefully falls back to cloud if unavailable
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

DEFAULT_LIGHT_MODEL = "google/gemini-3-pro-preview"


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


# ---------------------------------------------------------------------------
# Local LLM Backend (LM Studio / any OpenAI-compatible server)
# ---------------------------------------------------------------------------

class LocalLLMBackend:
    """
    Local LLM backend for LM Studio or any OpenAI-compatible server.

    Configured via environment variables:
    - LOCAL_LLM_URL: Base URL of the local API (e.g. http://localhost:1234/v1)
    - LOCAL_LLM_MODEL: Default model name as known to LM Studio
    - LOCAL_LLM_ENABLED: 'true'/'false', default 'false'

    Models are identified by 'local/' prefix (e.g. 'local/llama-3.3-70b').
    The prefix is stripped when making actual API calls.
    """

    def __init__(self) -> None:
        self._client = None
        self._client_url_cache: str = ""
        self._last_health_check: float = 0.0
        self._last_health_result: bool = False
        self._health_cache_sec: float = 60.0

    @property
    def base_url(self) -> str:
        return os.environ.get("LOCAL_LLM_URL", "http://localhost:1234/v1")

    @property
    def enabled(self) -> bool:
        return os.environ.get("LOCAL_LLM_ENABLED", "false").lower() == "true"

    @property
    def default_model(self) -> str:
        return os.environ.get("LOCAL_LLM_MODEL", "local-model")

    def is_healthy(self) -> bool:
        """
        Check if local LLM server is reachable.
        Result is cached for health_cache_sec seconds.
        """
        if not self.enabled:
            return False
        now = time.time()
        if now - self._last_health_check < self._health_cache_sec:
            return self._last_health_result
        try:
            import requests
            url = self.base_url.rstrip("/")
            # Strip /v1 suffix if present to get /v1/models
            if url.endswith("/v1"):
                models_url = url + "/models"
            else:
                models_url = url + "/v1/models"
            resp = requests.get(
                models_url,
                timeout=3,
                headers={"Authorization": "Bearer lm-studio"},
            )
            result = resp.status_code == 200
        except Exception as e:
            log.debug("Local LLM health check failed: %s", e)
            result = False
        self._last_health_check = now
        self._last_health_result = result
        log.debug("Local LLM health check: %s -> %s", self.base_url, result)
        return result

    def list_models(self) -> List[str]:
        """Return list of model IDs available in LM Studio. Empty on failure."""
        if not self.enabled:
            return []
        try:
            import requests
            url = self.base_url.rstrip("/")
            if url.endswith("/v1"):
                models_url = url + "/models"
            else:
                models_url = url + "/v1/models"
            resp = requests.get(models_url, timeout=3,
                                headers={"Authorization": "Bearer lm-studio"})
            if resp.status_code == 200:
                data = resp.json()
                return [m["id"] for m in data.get("data", []) if m.get("id")]
        except Exception:
            pass
        return []

    def _get_client(self):
        """Lazy-init OpenAI client. Rebuilds if base_url changed (e.g. new ngrok URL)."""
        current_url = self.base_url
        if self._client is None or self._client_url_cache != current_url:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=current_url,
                api_key="lm-studio",  # LM Studio requires any non-empty key
            )
            self._client_url_cache = current_url
        return self._client

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 8192,
        **kwargs,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Chat with local backend.

        Returns (message_dict, usage_dict).
        Cost is always 0.0 for local models.
        """
        client = self._get_client()
        # Strip 'local/' prefix — LM Studio uses its own model IDs
        actual_model = model.removeprefix("local/") if model.startswith("local/") else model

        call_kwargs: Dict[str, Any] = {
            "model": actual_model,
            "messages": messages,
            "max_tokens": max_tokens,
        }

        if tools:
            # Strip cache_control and other OpenRouter-specific extensions
            # Local models use standard OpenAI tool format
            clean_tools = []
            for t in tools:
                clean_t = {k: v for k, v in t.items() if k not in ("cache_control",)}
                clean_tools.append(clean_t)
            call_kwargs["tools"] = clean_tools
            call_kwargs["tool_choice"] = "auto"

        resp = client.chat.completions.create(**call_kwargs)
        resp_dict = resp.model_dump()
        choices = resp_dict.get("choices") or [{}]
        msg = (choices[0] if choices else {}).get("message") or {}
        usage = resp_dict.get("usage") or {}

        # Local models are free — cost is always 0
        usage["cost"] = 0.0
        usage["local"] = True  # mark as local for logging/routing decisions

        return msg, usage


# Module-level singleton — shared health cache across all LLMClient instances
_local_backend = LocalLLMBackend()


# ---------------------------------------------------------------------------
# Main LLM Client
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Multi-backend LLM client.

    Routes requests to:
    - OpenRouter (default) for cloud models
    - LocalLLMBackend for models prefixed with 'local/'

    All LLM calls in Ouroboros go through this class.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self._base_url = base_url
        self._client = None

    @staticmethod
    def is_local_model(model: str) -> bool:
        """Return True if the model should be routed to the local backend."""
        return str(model or "").startswith("local/")

    @property
    def local_backend(self) -> LocalLLMBackend:
        """Access the shared local backend instance."""
        return _local_backend

    def local_status(self) -> Dict[str, Any]:
        """Return a dict with current local backend status."""
        lb = _local_backend
        return {
            "enabled": lb.enabled,
            "url": lb.base_url,
            "default_model": lb.default_model,
            "healthy": lb.is_healthy() if lb.enabled else False,
            "prefer": os.environ.get("LOCAL_LLM_PREFER", "false").lower() == "true",
        }

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
        """
        Single LLM call. Routes to local backend if model starts with 'local/'.

        Returns: (response_message_dict, usage_dict with cost).
        For local models, cost is always 0.0.
        """
        # Route to local backend if requested
        if self.is_local_model(model):
            return self._chat_local(messages, model, tools, max_tokens)

        # Otherwise use OpenRouter
        return self._chat_openrouter(messages, model, tools, reasoning_effort, max_tokens, tool_choice)

    def _chat_local(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]],
        max_tokens: int,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Route call to local LLM backend with fallback to cloud."""
        lb = _local_backend
        if not lb.enabled:
            raise RuntimeError(
                "Local LLM backend is disabled. Set LOCAL_LLM_ENABLED=true in environment."
            )
        if not lb.is_healthy():
            raise RuntimeError(
                f"Local LLM backend is not reachable at {lb.base_url}. "
                "Check that LM Studio is running and the tunnel is active."
            )
        log.debug("Routing to local backend: model=%s url=%s", model, lb.base_url)
        return lb.chat(messages=messages, model=model, tools=tools, max_tokens=max_tokens)

    def _chat_openrouter(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]],
        reasoning_effort: str,
        max_tokens: int,
        tool_choice: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Send chat to OpenRouter."""
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
        """
        Return list of available models (for switch_model tool schema).

        Includes local model if LOCAL_LLM_ENABLED=true and backend is healthy.
        """
        main = os.environ.get("OUROBOROS_MODEL", "anthropic/claude-sonnet-4.6")
        code = os.environ.get("OUROBOROS_MODEL_CODE", "")
        light = os.environ.get("OUROBOROS_MODEL_LIGHT", "")
        models = [main]
        if code and code != main:
            models.append(code)
        if light and light != main and light != code:
            models.append(light)

        # Add local model if backend is healthy
        lb = _local_backend
        if lb.enabled and lb.is_healthy():
            local_model = f"local/{lb.default_model}"
            if local_model not in models:
                models.append(local_model)

        return models
