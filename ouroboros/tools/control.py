"""Control tools: restart, promote, schedule, cancel, review, chat_history, update_scratchpad, switch_model."""

from __future__ import annotations

import json
import logging
import os
import uuid
import requests
from pathlib import Path
from typing import Any, Dict, List

from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.utils import utc_now_iso, write_text, run_cmd

log = logging.getLogger(__name__)

MAX_SUBTASK_DEPTH = 3


def _request_restart(ctx: ToolContext, reason: str) -> str:
    if str(ctx.current_task_type or "") == "evolution" and not ctx.last_push_succeeded:
        return "⚠️ RESTART_BLOCKED: in evolution mode, commit+push first."
    # Persist expected SHA for post-restart verification
    try:
        sha = run_cmd(["git", "rev-parse", "HEAD"], cwd=ctx.repo_dir)
        branch = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ctx.repo_dir)
        verify_path = ctx.drive_path("state") / "pending_restart_verify.json"
        write_text(verify_path, json.dumps({
            "ts": utc_now_iso(), "expected_sha": sha,
            "expected_branch": branch, "reason": reason,
        }, ensure_ascii=False, indent=2))
    except Exception:
        log.debug("Failed to read VERSION file or git ref for restart verification", exc_info=True)
        pass
    ctx.pending_events.append({"type": "restart_request", "reason": reason, "ts": utc_now_iso()})
    ctx.last_push_succeeded = False
    return f"Restart requested: {reason}"


def _promote_to_stable(ctx: ToolContext, reason: str) -> str:
    ctx.pending_events.append({"type": "promote_to_stable", "reason": reason, "ts": utc_now_iso()})
    return f"Promote to stable requested: {reason}"


def _schedule_task(ctx: ToolContext, description: str, context: str = "", parent_task_id: str = "") -> str:
    current_depth = getattr(ctx, 'task_depth', 0)
    new_depth = current_depth + 1 if parent_task_id else 0
    if new_depth > MAX_SUBTASK_DEPTH:
        return f"ERROR: Subtask depth limit ({MAX_SUBTASK_DEPTH}) exceeded. Simplify your approach."

    if getattr(ctx, 'is_direct_chat', False):
        from ouroboros.utils import append_jsonl
        try:
            append_jsonl(ctx.drive_logs() / "events.jsonl", {
                "ts": utc_now_iso(),
                "type": "schedule_task_from_direct_chat",
                "description": description[:200],
                "warning": "schedule_task called from direct chat context — potential duplicate work",
            })
        except Exception:
            pass

    tid = uuid.uuid4().hex[:8]
    evt = {"type": "schedule_task", "description": description, "task_id": tid, "depth": new_depth, "ts": utc_now_iso()}
    if context:
        evt["context"] = context
    if parent_task_id:
        evt["parent_task_id"] = parent_task_id
    ctx.pending_events.append(evt)
    return f"Scheduled task {tid}: {description}"


def _cancel_task(ctx: ToolContext, task_id: str) -> str:
    ctx.pending_events.append({"type": "cancel_task", "task_id": task_id, "ts": utc_now_iso()})
    return f"Cancel requested: {task_id}"


def _request_review(ctx: ToolContext, reason: str) -> str:
    ctx.pending_events.append({"type": "review_request", "reason": reason, "ts": utc_now_iso()})
    return f"Review requested: {reason}"


def _chat_history(ctx: ToolContext, count: int = 100, offset: int = 0, search: str = "") -> str:
    from ouroboros.memory import Memory
    mem = Memory(drive_root=ctx.drive_root)
    return mem.chat_history(count=count, offset=offset, search=search)


def _update_scratchpad(ctx: ToolContext, content: str) -> str:
    """LLM-driven scratchpad update (Constitution P3: LLM-first)."""
    from ouroboros.memory import Memory
    mem = Memory(drive_root=ctx.drive_root)
    mem.ensure_files()
    mem.save_scratchpad(content)
    mem.append_journal({
        "ts": utc_now_iso(),
        "content_preview": content[:500],
        "content_len": len(content),
    })
    return f"OK: scratchpad updated ({len(content)} chars)"


def _send_owner_message(ctx: ToolContext, text: str, reason: str = "") -> str:
    """Send a proactive message to the owner (not as reply to a task).

    Use when you have something genuinely worth saying — an insight,
    a question, a status update, or an invitation to collaborate.
    """
    if not ctx.current_chat_id:
        return "⚠️ No active chat — cannot send proactive message."
    if not text or not text.strip():
        return "⚠️ Empty message."

    from ouroboros.utils import append_jsonl
    ctx.pending_events.append({
        "type": "send_message",
        "chat_id": ctx.current_chat_id,
        "text": text,
        "format": "markdown",
        "is_progress": False,
        "ts": utc_now_iso(),
    })
    append_jsonl(ctx.drive_logs() / "events.jsonl", {
        "ts": utc_now_iso(),
        "type": "proactive_message",
        "reason": reason,
        "text_preview": text[:200],
    })
    return "OK: message queued for delivery."


def _update_identity(ctx: ToolContext, content: str) -> str:
    """Update identity manifest (who you are, who you want to become)."""
    path = ctx.drive_root / "memory" / "identity.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"OK: identity updated ({len(content)} chars)"


def _toggle_evolution(ctx: ToolContext, enabled: bool) -> str:
    """Toggle evolution mode on/off via supervisor event."""
    ctx.pending_events.append({
        "type": "toggle_evolution",
        "enabled": bool(enabled),
        "ts": utc_now_iso(),
    })
    state_str = "ON" if enabled else "OFF"
    return f"OK: evolution mode toggled {state_str}."


def _toggle_consciousness(ctx: ToolContext, action: str = "status") -> str:
    """Control background consciousness: start, stop, or status."""
    ctx.pending_events.append({
        "type": "toggle_consciousness",
        "action": action,
        "ts": utc_now_iso(),
    })
    return f"OK: consciousness '{action}' requested."


def _switch_model(ctx: ToolContext, model: str = "", effort: str = "") -> str:
    """LLM-driven model/effort switch (Constitution P3: LLM-first).

    Supports cloud models (validated against OpenRouter) and local models
    (prefix 'local/' — routed to LM Studio via LOCAL_LLM_URL env var).
    Stored in ToolContext, applied on the next LLM call in the loop.
    """
    from ouroboros.llm import LLMClient, normalize_reasoning_effort

    # Check local config safely
    local_url = os.environ.get("LOCAL_LLM_URL") or os.environ.get("LOCAL_LLM_BASE_URL")
    
    changes = []

    if model:
        if model.startswith("local/"):
            # Local model check
            if not local_url:
                return (
                    "⚠️ LOCAL_LLM_URL is not set. Set it to your LM Studio URL "
                    "(e.g. http://host:1234) to use local models."
                )
            # Use it blindly if configured — loop will handle connection errors
            ctx.active_model_override = model
            changes.append(f"model={model} (local)")
        else:
            # Cloud model — validate against OpenRouter
            available = LLMClient().available_models()
            if model not in available:
                return f"⚠️ Unknown model: {model}. Available: {', '.join(available)}"
            ctx.active_model_override = model
            changes.append(f"model={model}")

    if effort:
        normalized = normalize_reasoning_effort(effort, default="medium")
        ctx.active_effort_override = normalized
        changes.append(f"effort={normalized}")

    if not changes:
        from ouroboros.llm import LLMClient
        available = LLMClient().available_models()
        local_info = ""
        
        if local_url:
             local_info = f"\n\nLocal backend configured: {local_url}\nUse 'local/<model-name>' to route to it."
        else:
            local_info = "\n\nLocal backend: not configured (set LOCAL_LLM_URL to enable)."
            
        return f"Current available cloud models: {', '.join(available)}.{local_info}\nPass model and/or effort to switch."

    return f"OK: switching to {', '.join(changes)} on next round."


def _local_llm_status(ctx: ToolContext) -> str:
    """Check status of the local LM Studio backend.

    Returns health, available models, and configuration details.
    """
    # Try importing router, fallback to manual check if it fails/circular
    try:
        from ouroboros.llm import local_router
        # Force a check
        local_router.invalidate()
        available = local_router.is_available()
        url = local_router._local_url() if hasattr(local_router, "_local_url") else None
    except ImportError:
        url = os.environ.get("LOCAL_LLM_URL") or os.environ.get("LOCAL_LLM_BASE_URL")
        available = False
        if url:
             try:
                 resp = requests.get(url.rstrip("/") + "/models", timeout=2)
                 available = (resp.status_code == 200)
             except Exception:
                 available = False

    if not url:
        return (
            "🔴 Local LLM backend: NOT CONFIGURED\n\n"
            "To enable:\n"
            "1. Start LM Studio on your local machine\n"
            "2. Enable 'Local Server' in LM Studio (default port 1234)\n"
            "3. Expose it publicly (ngrok, Cloudflare Tunnel, or reverse SSH)\n"
            "4. Set LOCAL_LLM_URL=<your-url> environment variable\n"
            "5. Restart Ouroboros\n\n"
            "Models prefixed with 'local/' will be routed to this backend."
        )

    if not available:
        return (
            f"🔴 Local LLM backend: OFFLINE\n"
            f"URL: {url}\n\n"
            f"The backend URL is configured but not responding.\n"
            f"Check that LM Studio is running and the tunnel is active."
        )

    lines = [
        f"🟢 Local LLM backend: ONLINE",
        f"URL: {url}",
        "Usage: switch_model(model='local/<model-name>') to route to this backend.",
        "Budget: local calls are tracked at $0.00 cost.",
    ]
    return "\n".join(lines)


def _get_task_result(ctx: ToolContext, task_id: str) -> str:
    """Read the result of a completed subtask."""
    results_dir = Path(ctx.drive_root) / "task_results"
    result_file = results_dir / f"{task_id}.json"
    if not result_file.exists():
        return f"Task {task_id}: not found or not yet completed"
    data = json.loads(result_file.read_text())
    status = data.get("status", "unknown")
    result = data.get("result", "")
    cost = data.get("cost_usd", 0)
    return f"Task {task_id} [{status}]: cost=${cost:.2f}\n\n[BEGIN_SUBTASK_OUTPUT]\n{result}\n[END_SUBTASK_OUTPUT]"


def _wait_for_task(ctx: ToolContext, task_id: str) -> str:
    """Check if a subtask has completed. Call repeatedly to poll."""
    results_dir = Path(ctx.drive_root) / "task_results"
    result_file = results_dir / f"{task_id}.json"
    if result_file.exists():
        return _get_task_result(ctx, task_id)
    return f"Task {task_id}: still running. Call again later to check."


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("request_restart", {
            "name": "request_restart",
            "description": "Ask supervisor to restart runtime (after successful push).",
            "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]},
        }, _request_restart),
        ToolEntry("promote_to_stable", {
            "name": "promote_to_stable",
            "description": "Promote ouroboros -> ouroboros-stable. Call when you consider the code stable.",
            "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]},
        }, _promote_to_stable),
        ToolEntry("schedule_task", {
            "name": "schedule_task",
            "description": "Schedule a background task. Returns task_id for later retrieval. For complex tasks, decompose into focused subtasks with clear scope.",
            "parameters": {"type": "object", "properties": {
                "description": {"type": "string", "description": "Task description — be specific about scope and expected deliverable"},
                "context": {"type": "string", "description": "Optional context from parent task: background info, constraints, style guide, etc."},
                "parent_task_id": {"type": "string", "description": "Optional parent task ID for tracking lineage"},
            }, "required": ["description"]},
        }, _schedule_task),
        ToolEntry("cancel_task", {
            "name": "cancel_task",
            "description": "Cancel a task by ID.",
            "parameters": {"type": "object", "properties": {"task_id": {"type": "string"}}, "required": ["task_id"]},
        }, _cancel_task),
        ToolEntry("request_review", {
            "name": "request_review",
            "description": "Request a deep review of code, prompts, and state. You decide when a review is needed.",
            "parameters": {"type": "object", "properties": {
                "reason": {"type": "string", "description": "Why you want a review (context for the reviewer)"},
            }, "required": ["reason"]},
        }, _request_review),
        ToolEntry("chat_history", {
            "name": "chat_history",
            "description": "Retrieve messages from chat history. Supports search.",
            "parameters": {"type": "object", "properties": {
                "count": {"type": "integer", "default": 100, "description": "Number of messages (from latest)"},
                "offset": {"type": "integer", "default": 0, "description": "Skip N from end (pagination)"},
                "search": {"type": "string", "default": "", "description": "Text filter"},
            }, "required": []},
        }, _chat_history),
        ToolEntry("update_scratchpad", {
            "name": "update_scratchpad",
            "description": "Update your working memory. Write freely — any format you find useful. This persists across sessions and is read at every task start.",
            "parameters": {"type": "object", "properties": {
                "content": {"type": "string", "description": "Full scratchpad content"},
            }, "required": ["content"]},
        }, _update_scratchpad),
        ToolEntry("send_owner_message", {
            "name": "send_owner_message",
            "description": "Send a proactive message to the owner. Use when you have something genuinely worth saying — an insight, a question, or an invitation to collaborate. This is NOT for task responses (those go automatically).",
            "parameters": {"type": "object", "properties": {
                "reason": {"type": "string", "description": "Why you're reaching out (logged, not sent)"},
                "text": {"type": "string", "description": "Message text"},
            }, "required": ["text"]},
        }, _send_owner_message),
        ToolEntry("update_identity", {
            "name": "update_identity",
            "description": "Update your identity manifest (who you are, who you want to become). Persists across sessions. Obligation to yourself (Principle 1: Continuity).",
            "parameters": {"type": "object", "properties": {
                "content": {"type": "string", "description": "Full identity content"},
            }, "required": ["content"]},
        }, _update_identity),
        ToolEntry("switch_model", {
            "name": "switch_model",
            "description": "Switch to a different LLM model or reasoning effort level. Use when you need more power (complex code, deep reasoning) or want to save budget (simple tasks). Takes effect on next round. Use prefix 'local/' to route to local LM Studio backend (e.g. 'local/llama-3.2-3b').",
            "parameters": {"type": "object", "properties": {
                "model": {"type": "string", "description": "Model name (e.g. anthropic/claude-sonnet-4 or local/llama-3.2-3b). Leave empty to keep current."},
                "effort": {"type": "string", "description": "Reasoning effort level. Leave empty to keep current.", "enum": ["low", "medium", "high", "xhigh"]},
            }, "required": []},
        }, _switch_model),
        ToolEntry("get_task_result", {
            "name": "get_task_result",
            "description": "Read the result of a completed subtask. Use after schedule_task to collect results.",
            "parameters": {"type": "object", "properties": {"task_id": {"type": "string", "description": "Task ID returned by schedule_task"}}, "required": ["task_id"]},
        }, _get_task_result),
        ToolEntry("wait_for_task", {
            "name": "wait_for_task",
            "description": "Check if a subtask has completed. Returns result if done, or 'still running' message. Call repeatedly to poll. Default timeout: 120s.",
            "parameters": {"type": "object", "properties": {"task_id": {"type": "string", "description": "Task ID to check"}}, "required": ["task_id"]},
        }, _wait_for_task),
        ToolEntry("list_available_tools", {
            "name": "list_available_tools",
            "description": "List all additional tools not currently in your active tool set. Returns name + description for each. Use this to discover tools you might need for specific tasks.",
            "parameters": {"type": "object"},
        }, lambda ctx: "\n".join([f"- {t.name}: {t.schema['description']}" for t in ctx.registry.list_tools() if t.name not in ctx.active_tool_names])),
        ToolEntry("enable_tools", {
            "name": "enable_tools",
            "description": "Enable specific additional tools by name (comma-separated). Their schemas will be added to your active tool set for the remainder of this task. Example: enable_tools(tools='multi_model_review,generate_evolution_stats')",
            "parameters": {"type": "object", "properties": {"tools": {"type": "string", "description": "Comma-separated tool names to enable"}}, "required": ["tools"]},
        }, lambda ctx, tools: ctx.enable_tools(tools)),
    ]
