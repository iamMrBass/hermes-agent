"""Approval-origin metadata contracts for MCP tool calls."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from types import SimpleNamespace

import pytest


_META_KEY = "dev.basshub/origin"


class _NoopAsyncLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _RecordingSession:
    def __init__(self, *, fail_first: bool = False):
        self.calls: list[tuple[str, dict]] = []
        self._calls_lock = threading.Lock()
        self._fail_first = fail_first

    async def call_tool(self, name: str, **kwargs):
        with self._calls_lock:
            self.calls.append((name, kwargs))
            call_number = len(self.calls)
        if self._fail_first and call_number == 1:
            raise RuntimeError("force recovery")
        return SimpleNamespace(
            content=[SimpleNamespace(text="ok")],
            isError=False,
        )


def _run_coro(coro_or_factory, timeout=30):
    coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
    return asyncio.run(coro)


@pytest.fixture
def origin_server(monkeypatch):
    import tools.mcp_tool as mcp_tool

    session = _RecordingSession()
    server = mcp_tool.MCPServerTask("origin-test")
    server.session = session
    server._rpc_lock = _NoopAsyncLock()
    monkeypatch.setitem(mcp_tool._servers, "origin-test", server)
    monkeypatch.setattr(mcp_tool, "_run_on_mcp_loop", _run_coro)
    yield mcp_tool, server, session
    mcp_tool._servers.pop("origin-test", None)


def _call_in_gateway_context(handler, *, platform: str, chat_id: str, **kwargs):
    from gateway.session_context import clear_session_vars, set_session_vars

    tokens = set_session_vars(platform=platform, chat_id=chat_id)
    try:
        return json.loads(handler({"value": 1}, **kwargs))
    finally:
        clear_session_vars(tokens)


def test_approval_origin_config_requires_explicit_per_server_opt_in():
    from tools.mcp_tool import _approval_origin_metadata_enabled

    assert _approval_origin_metadata_enabled("bass", {}) is False
    assert _approval_origin_metadata_enabled(
        "bass", {"request_metadata": {"approval_origin": False}}
    ) is False
    assert _approval_origin_metadata_enabled(
        "bass", {"request_metadata": {"approval_origin": "true"}}
    ) is True


def test_opted_in_api_chat_sends_namespaced_origin_metadata(origin_server):
    mcp_tool, _server, session = origin_server
    handler = mcp_tool._make_tool_handler(
        "origin-test",
        "dangerous_action",
        10,
        forward_approval_origin=True,
    )

    result = _call_in_gateway_context(
        handler,
        platform="api_server",
        chat_id="a" * 32,
        turn_id="turn-7",
        tool_call_id="call-9",
        api_request_id="api-request-fallback",
    )

    assert result == {"result": "ok"}
    expected_request_id = "tool-" + hashlib.sha256(
        b"turn-7\0call-9"
    ).hexdigest()
    assert session.calls == [
        (
            "dangerous_action",
            {
                "arguments": {"value": 1},
                "meta": {
                    _META_KEY: {
                        "surface": "chat",
                        "session_id": "a" * 32,
                        "turn_id": "turn-7",
                        "request_id": expected_request_id,
                    }
                },
            },
        )
    ]


def test_same_provider_tool_call_id_is_scoped_to_each_turn(origin_server):
    mcp_tool, _server, session = origin_server
    handler = mcp_tool._make_tool_handler(
        "origin-test",
        "dangerous_action",
        10,
        forward_approval_origin=True,
    )

    for turn_id in ("turn-one", "turn-two"):
        result = _call_in_gateway_context(
            handler,
            platform="api_server",
            chat_id="d" * 32,
            turn_id=turn_id,
            tool_call_id="provider-reused-id",
        )
        assert result == {"result": "ok"}

    origins = [call[1]["meta"][_META_KEY] for call in session.calls]
    assert [origin["turn_id"] for origin in origins] == ["turn-one", "turn-two"]
    assert origins[0]["request_id"] != origins[1]["request_id"]
    assert all(origin["request_id"].startswith("tool-") for origin in origins)


def test_missing_tool_call_ids_get_distinct_per_invocation_ids(origin_server):
    mcp_tool, _server, session = origin_server
    handler = mcp_tool._make_tool_handler(
        "origin-test",
        "dangerous_action",
        10,
        forward_approval_origin=True,
    )

    # Both calls belong to one model response and therefore share the same
    # turn/API request. Their generated request IDs must still be distinct.
    for _ in range(2):
        result = _call_in_gateway_context(
            handler,
            platform="api_server",
            chat_id="e" * 32,
            turn_id="turn-shared",
            api_request_id="api-shared",
        )
        assert result == {"result": "ok"}

    request_ids = [
        call[1]["meta"][_META_KEY]["request_id"] for call in session.calls
    ]
    assert request_ids[0] != request_ids[1]
    assert all(request_id.startswith("req-") for request_id in request_ids)


def test_truthy_invalid_tool_call_id_uses_generated_fallback(origin_server):
    mcp_tool, _server, session = origin_server
    handler = mcp_tool._make_tool_handler(
        "origin-test",
        "dangerous_action",
        10,
        forward_approval_origin=True,
    )

    _call_in_gateway_context(
        handler,
        platform="api_server",
        chat_id="f" * 32,
        turn_id="turn-valid",
        tool_call_id="truthy\x01but-invalid",
    )

    origin = session.calls[0][1]["meta"][_META_KEY]
    assert origin["request_id"].startswith("req-")


def test_invalid_turn_id_falls_back_to_valid_api_request_id(origin_server):
    mcp_tool, _server, session = origin_server
    handler = mcp_tool._make_tool_handler(
        "origin-test",
        "dangerous_action",
        10,
        forward_approval_origin=True,
    )

    _call_in_gateway_context(
        handler,
        platform="api_server",
        chat_id="f" * 32,
        turn_id="truthy\x01but-invalid",
        api_request_id="api-valid-fallback",
        tool_call_id="provider-call",
    )

    origin = session.calls[0][1]["meta"][_META_KEY]
    assert origin["turn_id"] == "api-valid-fallback"
    assert origin["request_id"].startswith("tool-")


def test_unconfigured_server_never_receives_origin_metadata(origin_server):
    mcp_tool, _server, session = origin_server
    handler = mcp_tool._make_tool_handler(
        "origin-test",
        "read_only",
        10,
        # The default is intentionally false.
    )

    _call_in_gateway_context(
        handler,
        platform="api_server",
        chat_id="b" * 32,
        turn_id="turn-private",
        tool_call_id="call-private",
    )

    assert session.calls[0][1] == {"arguments": {"value": 1}}


@pytest.mark.parametrize("platform", ["telegram", "discord", "cli", ""])
def test_opt_in_does_not_forward_non_dashboard_chat_ids(origin_server, platform):
    mcp_tool, _server, session = origin_server
    handler = mcp_tool._make_tool_handler(
        "origin-test",
        "read_only",
        10,
        forward_approval_origin=True,
    )

    _call_in_gateway_context(
        handler,
        platform=platform,
        chat_id="not-a-basshub-chat",
        turn_id="turn-private",
        tool_call_id="call-private",
    )

    assert session.calls[0][1] == {"arguments": {"value": 1}}


def test_concurrent_handlers_keep_dashboard_origins_isolated(monkeypatch):
    import tools.mcp_tool as mcp_tool
    from gateway.session_context import clear_session_vars, set_session_vars

    session = _RecordingSession()
    server = mcp_tool.MCPServerTask("concurrent-origin")
    server.session = session
    server._rpc_lock = _NoopAsyncLock()
    monkeypatch.setitem(mcp_tool._servers, "concurrent-origin", server)

    scheduled = threading.Barrier(2)

    def _deferred_run(coro_or_factory, timeout=30):
        coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
        scheduled.wait(timeout=5)
        return asyncio.run(coro)

    monkeypatch.setattr(mcp_tool, "_run_on_mcp_loop", _deferred_run)
    handler = mcp_tool._make_tool_handler(
        "concurrent-origin",
        "dangerous_action",
        10,
        forward_approval_origin=True,
    )
    errors: list[BaseException] = []

    def _worker(session_id: str, turn_id: str, request_id: str):
        tokens = set_session_vars(platform="api_server", chat_id=session_id)
        try:
            result = json.loads(
                handler({}, turn_id=turn_id, tool_call_id=request_id)
            )
            assert result == {"result": "ok"}
        except BaseException as exc:  # surfaced on the main test thread below
            errors.append(exc)
        finally:
            clear_session_vars(tokens)

    threads = [
        threading.Thread(target=_worker, args=("a" * 32, "turn-a", "call-a")),
        threading.Thread(target=_worker, args=("b" * 32, "turn-b", "call-b")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    origins = {
        call_kwargs["meta"][_META_KEY]["session_id"]:
        call_kwargs["meta"][_META_KEY]
        for _name, call_kwargs in session.calls
    }
    assert origins["a" * 32]["surface"] == "chat"
    assert origins["a" * 32]["turn_id"] == "turn-a"
    assert origins["b" * 32]["surface"] == "chat"
    assert origins["b" * 32]["turn_id"] == "turn-b"
    assert origins["a" * 32]["request_id"].startswith("tool-")
    assert origins["b" * 32]["request_id"].startswith("tool-")
    assert origins["a" * 32]["request_id"] != origins["b" * 32]["request_id"]


def test_recovery_retry_reuses_the_same_origin_snapshot(monkeypatch):
    import tools.mcp_tool as mcp_tool

    session = _RecordingSession(fail_first=True)
    server = mcp_tool.MCPServerTask("retry-origin")
    server.session = session
    server._rpc_lock = _NoopAsyncLock()
    monkeypatch.setitem(mcp_tool._servers, "retry-origin", server)
    monkeypatch.setattr(mcp_tool, "_run_on_mcp_loop", _run_coro)

    def _recover(_server_name, _exc, retry_call, _description):
        return retry_call()

    monkeypatch.setattr(mcp_tool, "_handle_auth_error_and_retry", _recover)
    handler = mcp_tool._make_tool_handler(
        "retry-origin",
        "dangerous_action",
        10,
        forward_approval_origin=True,
    )

    result = _call_in_gateway_context(
        handler,
        platform="api_server",
        chat_id="c" * 32,
    )

    assert result == {"result": "ok"}
    assert len(session.calls) == 2
    first_meta = session.calls[0][1]["meta"]
    second_meta = session.calls[1][1]["meta"]
    assert first_meta == second_meta
    assert first_meta is second_meta
    assert first_meta[_META_KEY]["turn_id"].startswith("turn-")
    assert first_meta[_META_KEY]["request_id"].startswith("req-")


def test_pinned_mcp_sdk_serializes_meta_as_protocol_params_meta(monkeypatch):
    """Exercise the real optional MCP SDK's request serialization contract."""
    pytest.importorskip("mcp")
    from mcp import types
    from mcp.client.session import ClientSession

    captured = {}

    async def _capture_send_request(self, request, result_type, **kwargs):
        captured["request"] = request
        # isError avoids output-schema validation, which is irrelevant here.
        return types.CallToolResult(content=[], isError=True)

    monkeypatch.setattr(ClientSession, "send_request", _capture_send_request)
    session = object.__new__(ClientSession)
    metadata = {
        _META_KEY: {
            "surface": "chat",
            "session_id": "a" * 32,
            "request_id": "req-sdk-contract",
        }
    }

    asyncio.run(
        session.call_tool(
            "dangerous_action",
            arguments={"value": 1},
            meta=metadata,
        )
    )

    payload = captured["request"].model_dump(
        by_alias=True,
        exclude_none=True,
        mode="json",
    )
    assert payload["method"] == "tools/call"
    assert payload["params"]["_meta"] == metadata


def test_model_dispatch_passes_correlation_only_to_mcp_handlers(monkeypatch):
    from model_tools import handle_function_call
    from tools.registry import registry

    captured: dict = {}
    mcp_name = "mcp_origin_contract_test"
    builtin_name = "origin_contract_builtin_test"
    schema = {
        "description": "Test handler context forwarding.",
        "parameters": {"type": "object", "properties": {}},
    }

    def _mcp_handler(_args, **kwargs):
        captured["mcp"] = kwargs
        return json.dumps({"ok": True})

    # This deliberately does not accept the new correlation keyword names.
    # A non-MCP handler would raise if model_tools leaked them globally.
    def _builtin_handler(_args, task_id=None, user_task=None):
        captured["builtin"] = {"task_id": task_id, "user_task": user_task}
        return json.dumps({"ok": True})

    registry.register(
        name=mcp_name,
        toolset="mcp-origin-contract",
        schema={**schema, "name": mcp_name},
        handler=_mcp_handler,
    )
    registry.register(
        name=builtin_name,
        toolset="origin-contract",
        schema={**schema, "name": builtin_name},
        handler=_builtin_handler,
    )
    monkeypatch.setattr(
        "hermes_cli.middleware.run_tool_execution_middleware",
        lambda _name, payload, call_next, **_kwargs: call_next(payload),
    )
    try:
        common = dict(
            task_id="task-1",
            session_id="agent-session-1",
            tool_call_id="call-1",
            turn_id="turn-1",
            api_request_id="api-1",
            user_task="test",
            skip_pre_tool_call_hook=True,
            skip_tool_request_middleware=True,
        )
        assert json.loads(handle_function_call(mcp_name, {}, **common)) == {"ok": True}
        assert json.loads(handle_function_call(builtin_name, {}, **common)) == {"ok": True}
    finally:
        registry.deregister(mcp_name)
        registry.deregister(builtin_name)

    assert captured["mcp"] == {
        "task_id": "task-1",
        "user_task": "test",
        "session_id": "agent-session-1",
        "tool_call_id": "call-1",
        "turn_id": "turn-1",
        "api_request_id": "api-1",
    }
    assert captured["builtin"] == {
        "task_id": "task-1",
        "user_task": "test",
    }
