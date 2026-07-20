"""Gateway session context propagation across delegation worker pools."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace


def _gateway_identity() -> tuple[str, str, str]:
    from gateway.session_context import get_session_env

    return (
        get_session_env("HERMES_SESSION_PLATFORM", ""),
        get_session_env("HERMES_SESSION_CHAT_ID", ""),
        get_session_env("HERMES_SESSION_ID", ""),
    )


def _parent_agent(**overrides):
    values = {
        "base_url": "https://example.test/v1",
        "api_key": "test-key",
        "provider": "test-provider",
        "api_mode": "chat_completions",
        "model": "test-model",
        "platform": "api_server",
        "_delegate_depth": 0,
        "_interrupt_requested": False,
        "_delegate_spinner": None,
        "_memory_manager": None,
        "_current_task_id": None,
        "_current_turn_id": "turn-parent",
        "_active_children": [],
        "_active_children_lock": threading.Lock(),
        "session_id": "parent-session",
        "session_estimated_cost_usd": 0.0,
        "session_cost_source": "none",
        "session_cost_status": "unknown",
        "tool_progress_callback": None,
        "thinking_callback": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _ContextRecordingChild:
    def __init__(self, captured: list[tuple[str, str, str]]):
        self._captured = captured
        self._credential_pool = None
        self._subagent_id = None
        self._delegate_role = "leaf"
        self._delegate_saved_tool_names = []
        self.tool_progress_callback = None
        self.model = "test-model"
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_estimated_cost_usd = 0.0

    def run_conversation(self, user_message, task_id=None):
        self._captured.append(_gateway_identity())
        return {
            "final_response": "done",
            "completed": True,
            "interrupted": False,
            "api_calls": 1,
            "messages": [],
        }

    def get_activity_summary(self):
        return {"api_call_count": 1, "max_iterations": 1}

    def close(self):
        return None


def test_single_child_timeout_worker_keeps_originating_api_chat(monkeypatch):
    import tools.delegate_tool as delegate_tool
    from gateway.session_context import clear_session_vars, set_session_vars

    for name in (
        "HERMES_SESSION_PLATFORM",
        "HERMES_SESSION_CHAT_ID",
        "HERMES_SESSION_ID",
    ):
        monkeypatch.delenv(name, raising=False)

    captured: list[tuple[str, str, str]] = []
    child = _ContextRecordingChild(captured)
    parent = _parent_agent(_active_children=[child])
    tokens = set_session_vars(
        platform="api_server",
        chat_id="a" * 32,
        session_id="hermes-session-a",
    )
    try:
        result = delegate_tool._run_single_child(
            task_index=0,
            goal="use an MCP tool",
            child=child,
            parent_agent=parent,
        )
    finally:
        clear_session_vars(tokens)

    assert result["status"] == "completed"
    assert captured == [("api_server", "a" * 32, "hermes-session-a")]


def test_batch_worker_keeps_originating_api_chat(monkeypatch):
    import tools.delegate_tool as delegate_tool
    from gateway.session_context import clear_session_vars, set_session_vars

    barrier = threading.Barrier(2)
    captured: dict[str, tuple[str, str, str]] = {}

    monkeypatch.setattr(delegate_tool, "_get_max_spawn_depth", lambda: 2)
    monkeypatch.setattr(delegate_tool, "_get_max_concurrent_children", lambda: 2)
    monkeypatch.setattr(
        delegate_tool,
        "_load_config",
        lambda: {"max_iterations": 1},
    )
    monkeypatch.setattr(
        delegate_tool,
        "_resolve_delegation_credentials",
        lambda _cfg, _parent: {
            "model": None,
            "provider": None,
            "base_url": None,
            "api_key": None,
            "api_mode": None,
            "command": None,
            "args": None,
        },
    )
    monkeypatch.setattr(
        delegate_tool,
        "_build_child_agent",
        lambda task_index, **_kwargs: SimpleNamespace(
            _delegate_role="leaf",
            session_id=f"child-{task_index}",
        ),
    )

    def _record_batch_context(task_index, goal, child, parent_agent):
        captured[goal] = _gateway_identity()
        barrier.wait(timeout=5)
        return {
            "task_index": task_index,
            "status": "completed",
            "summary": "done",
            "api_calls": 1,
            "duration_seconds": 0,
            "_child_role": "leaf",
            "_child_cost_usd": 0.0,
        }

    monkeypatch.setattr(delegate_tool, "_run_single_child", _record_batch_context)

    tokens = set_session_vars(
        platform="api_server",
        chat_id="b" * 32,
        session_id="hermes-session-b",
    )
    try:
        result = json.loads(
            delegate_tool.delegate_task(
                tasks=[{"goal": "first"}, {"goal": "second"}],
                parent_agent=_parent_agent(),
            )
        )
    finally:
        clear_session_vars(tokens)

    assert [entry["status"] for entry in result["results"]] == [
        "completed",
        "completed",
    ]
    assert captured == {
        "first": ("api_server", "b" * 32, "hermes-session-b"),
        "second": ("api_server", "b" * 32, "hermes-session-b"),
    }


def test_context_copies_are_isolated_between_concurrent_submissions(monkeypatch):
    import tools.delegate_tool as delegate_tool
    from gateway.session_context import clear_session_vars, set_session_vars

    for name in ("HERMES_SESSION_PLATFORM", "HERMES_SESSION_CHAT_ID"):
        monkeypatch.delenv(name, raising=False)

    barrier = threading.Barrier(2)

    def _read_after_both_start():
        barrier.wait(timeout=5)
        return _gateway_identity()[:2]

    futures = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        for chat_id in ("c" * 32, "d" * 32):
            tokens = set_session_vars(platform="api_server", chat_id=chat_id)
            try:
                futures.append(
                    delegate_tool._submit_with_context(executor, _read_after_both_start)
                )
            finally:
                clear_session_vars(tokens)

        observed = {future.result(timeout=5) for future in futures}

    assert observed == {
        ("api_server", "c" * 32),
        ("api_server", "d" * 32),
    }
