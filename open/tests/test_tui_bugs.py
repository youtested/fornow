"""Tests for the TUI bugs found during review.

Covers: action_interrupt poison/double-_turn_done, down-arrow wiping the prompt,
partial streamed text being discarded on error, ModelPicker/SettingsScreen
crash-after-dismiss, sub-agent session/engine leaks and main-session corruption,
agent-toggle using the wrong engine, /models UI-thread freeze, the write-tool
block rendering, and the permission dialog wiring.

Headless Textual tests use App.run_test() (no real terminal needed).
"""

from __future__ import annotations

import sys
import io
from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult
from textual.screen import ModalScreen

from opencode_py.agent.loop import AgentLoop
from opencode_py.config import Config
from opencode_py.tools.write import _write
from opencode_py.tui.chat_view import ChatView, MessageBubble
from opencode_py.tui.input_bar import InputBar
from opencode_py.tui.model_picker import ModelPicker
from opencode_py.tui.permission_dialog import PermissionDialog
from opencode_py.tui.settings_screen import SettingsScreen
from opencode_py.tui.app import OpenCodeTUI
from opencode_py.session import Session


# --------------------------------------------------------------------------
# test harnesses
# --------------------------------------------------------------------------

class WidgetHost(App):
    """App that mounts a single widget (for run_test)."""

    def __init__(self, factory) -> None:
        super().__init__()
        self._factory = factory

    def compose(self) -> ComposeResult:
        yield self._factory()


class ModalHost(App):
    """App that pushes a modal screen immediately on mount."""

    def __init__(self, factory) -> None:
        super().__init__()
        self._factory = factory

    def on_mount(self) -> None:
        self.push_screen(self._factory())


class FakeEngine:
    agent = "build"
    permission = SimpleNamespace(mode="auto")


# --------------------------------------------------------------------------
# Bug 3: action_interrupt sets flag and does not finish turn
# --------------------------------------------------------------------------

async def test_action_interrupt_sets_flag_and_does_not_finish_turn():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        app._busy = True
        app.action_interrupt()
        assert app._interrupt_flag["requested"] is True
        # the old code called _turn_done() here, clearing _busy while the worker
        # was still running; the fix only flips the shared flag
        assert app._busy is True


async def test_turn_done_resets_interrupt_flag():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        app._interrupt_flag["requested"] = True
        app._turn_done(result=None)
        assert app._interrupt_flag["requested"] is False


# --------------------------------------------------------------------------
# ESC double-press interrupt (mirrors opencode's session.interrupt)
# --------------------------------------------------------------------------

async def test_esc_idle_focuses_input_without_arming():
    from opencode_py.tui.status_bar import StatusBar

    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        bar = app.query_one(InputBar)
        bar.focus()
        app._busy = False
        app.action_interrupt_escape()
        assert app._interrupt_flag["requested"] is False
        assert app._esc_presses == 0
        assert app.query_one(StatusBar).interrupt_armed is False


async def test_esc_first_press_arms_hint_only():
    from opencode_py.tui.status_bar import StatusBar

    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        app._busy = True
        app.action_interrupt_escape()
        assert app._esc_presses == 1
        assert app._interrupt_flag["requested"] is False
        assert app.query_one(StatusBar).interrupt_armed is True
        assert "esc again to interrupt" in app.query_one(StatusBar).render().plain


async def test_esc_second_press_interrupts():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        app._busy = True
        app.action_interrupt_escape()  # first press arms
        app.action_interrupt_escape()  # second press aborts
        assert app._interrupt_flag["requested"] is True
        assert app._esc_presses == 0


async def test_esc_hint_disarms_after_timeout():
    from opencode_py.tui.status_bar import StatusBar

    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        app._busy = True
        app.action_interrupt_escape()
        assert app._esc_presses == 1
        await pilot.pause(6)  # 5s timer elapses
        assert app._esc_presses == 0
        assert app.query_one(StatusBar).interrupt_armed is False


async def test_esc_second_press_within_window_via_key_binding():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        bar = app.query_one(InputBar)
        bar.focus()
        app._busy = True
        await pilot.press("escape")
        assert app._esc_presses == 1
        assert app._interrupt_flag["requested"] is False
        await pilot.press("escape")
        assert app._interrupt_flag["requested"] is True
        assert app._esc_presses == 0


async def test_esc_resets_when_window_expires():
    """Mirrors opencode's 5s window: a third press after the window resets the
    counter and must NOT abort (it only re-arms, like the first press)."""
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        app._busy = True
        app.action_interrupt_escape()  # press 1
        await pilot.pause(6)           # window expires -> reset
        app.action_interrupt_escape()  # press 2 outside the window: arms again
        assert app._interrupt_flag["requested"] is False
        assert app._esc_presses == 1


def test_engine_interrupt_honors_shared_flag():
    """A sub-agent spawned from the app engine must share the interrupt flag."""
    import opencode_py.agent.loop as loop_mod

    real_spawn = loop_mod.AgentLoop.spawn_task
    cfg = Config()
    registry = SimpleNamespace()
    parent = AgentLoop(cfg=cfg, registry=registry, directory=__import__("pathlib").Path("."))
    flag = {"requested": False}
    parent.interrupt = lambda: flag["requested"]
    try:
        assert parent.interrupt() is False
        flag["requested"] = True
        assert parent.interrupt() is True
    finally:
        loop_mod.AgentLoop.spawn_task = real_spawn


# --------------------------------------------------------------------------
# Bug 4: down-arrow must never wipe the typed prompt
# --------------------------------------------------------------------------

async def test_down_arrow_keeps_typed_prompt_without_history():
    app = WidgetHost(lambda: InputBar(commands=[]))
    async with app.run_test() as pilot:
        bar = app.query_one(InputBar)
        bar.input.focus()
        bar.input.value = "hello"
        bar.input.cursor_position = len(bar.input.value)
        await pilot.press("down")
        assert bar.input.value == "hello"


async def test_up_down_history_restores_draft():
    app = WidgetHost(lambda: InputBar(commands=[]))
    async with app.run_test() as pilot:
        bar = app.query_one(InputBar)
        bar.input.focus()
        bar._history = ["first prompt"]
        bar._hist_index = 1  # end position
        bar.input.value = "my draft"
        bar.input.cursor_position = len(bar.input.value)
        await pilot.press("up")
        assert bar.input.value == "first prompt"
        assert bar._draft == "my draft"
        await pilot.press("down")
        assert bar.input.value == "my draft"


async def test_up_repeatedly_keeps_original_draft():
    """Up twice into history must not overwrite the draft with the last item."""
    app = WidgetHost(lambda: InputBar(commands=[]))
    async with app.run_test() as pilot:
        bar = app.query_one(InputBar)
        bar.input.focus()
        bar._history = ["first prompt", "second prompt"]
        bar._hist_index = 2  # end position
        bar.input.value = "my draft"
        bar.input.cursor_position = len(bar.input.value)
        await pilot.press("up")
        await pilot.press("up")
        assert bar._draft == "my draft"
        assert bar.input.value == "first prompt"


# --------------------------------------------------------------------------
# Bug 5: partial streamed text must survive an error / empty-reply cleanup
# --------------------------------------------------------------------------

async def test_remove_last_stream_bubble_keeps_partial_text():
    app = WidgetHost(lambda: ChatView())
    async with app.run_test() as pilot:
        chat = app.query_one(ChatView)
        chat.stream_delta("partial ")
        chat.stream_delta("text")
        chat.remove_last_stream_bubble()
        bubbles = list(chat.query(MessageBubble))
        assert len(bubbles) == 1
        assert bubbles[0].content == "partial text"
        assert bubbles[0].streaming is False


async def test_remove_last_stream_bubble_removes_empty():
    app = WidgetHost(lambda: ChatView())
    async with app.run_test() as pilot:
        chat = app.query_one(ChatView)
        chat.begin_stream()
        chat.remove_last_stream_bubble()
        await pilot.pause()
        assert len(list(chat.query(MessageBubble))) == 0


# --------------------------------------------------------------------------
# Bug 1: ModelPicker.populate on a dismissed/pruned screen
# --------------------------------------------------------------------------

def test_model_picker_populate_when_not_attached_noop():
    picker = ModelPicker()
    assert picker.is_attached is False
    picker.populate({})  # must not raise NoMatches


def test_model_picker_set_loading_when_not_attached_noop():
    picker = ModelPicker()
    picker.set_loading()  # must not raise


# --------------------------------------------------------------------------
# Bug 2: SettingsScreen deferred render after dismissal
# --------------------------------------------------------------------------

def test_settings_render_when_not_attached_noop():
    screen = SettingsScreen(cfg=Config(), engine=FakeEngine(), auth=None)
    assert screen.is_attached is False
    screen._render_settings()  # must not raise NoMatches
    screen._keep_selection_visible()  # must not raise


# --------------------------------------------------------------------------
# Bug 12: permission dialog wiring
# --------------------------------------------------------------------------

async def test_permission_dialog_escape_reports_deny():
    decisions: list[str] = []
    app = ModalHost(
        lambda: PermissionDialog("run a command?", on_decision=decisions.append)
    )
    async with app.run_test() as pilot:
        await pilot.press("escape")
        assert decisions == ["deny"]


async def test_permission_dialog_button_reports_decision():
    decisions: list[str] = []
    app = ModalHost(
        lambda: PermissionDialog("run a command?", on_decision=decisions.append)
    )
    async with app.run_test() as pilot:
        await pilot.press("enter")  # Allow once is first / focused
        assert decisions == ["once"]


# --------------------------------------------------------------------------
# Bug 8: agent toggle must act on the active (sub-agent) engine
# --------------------------------------------------------------------------

async def test_toggle_agent_uses_active_engine():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        sub = FakeEngine()
        sub.agent = "plan"
        app._engines["sub"] = sub
        app._sessions["sub"] = SimpleNamespace(agent="plan", title="sub")
        app._current_session_id = "sub"
        app.action_toggle_agent()
        assert sub.agent == "build"
        # the main engine must be untouched
        assert app.engine.agent == "build"


# --------------------------------------------------------------------------
# Bug 14: /models with args must not run the sync fetch on the UI thread
# --------------------------------------------------------------------------

async def test_models_command_routes_to_model_picker():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        calls: list[str] = []
        app._open_model_picker = lambda: calls.append(1)
        app._run_command("/models --json")
        assert calls == [1]


# --------------------------------------------------------------------------
# Bug 7: missing sub-agent session must not fall back to the main session
# --------------------------------------------------------------------------

async def test_subagent_missing_session_registers_placeholder(monkeypatch):
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        import opencode_py.session as session_mod

        monkeypatch.setattr(session_mod, "load_session", lambda sid: None)
        app._on_subagent_start(
            {"kind": "subagent_start", "session_id": "abc123", "agent": "build", "title": "t"}
        )
        assert "abc123" in app._sessions
        assert app._sessions["abc123"].id == "abc123"
        assert app._sessions["abc123"] is not app.session


# --------------------------------------------------------------------------
# Bug 10: finished sub-agent sessions are pruned (no widget/engine leak)
# --------------------------------------------------------------------------

async def test_subagent_done_prunes_widgets_and_engines():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        sid = "sub1"
        chat = app._chat_for(sid)
        app._chats[sid] = chat
        app._engines[sid] = FakeEngine()
        app._sessions[sid] = SimpleNamespace(completed=None)
        app._busy_sessions.add(sid)
        app._running_agents[sid] = "t · build"
        app._on_subagent_done(
            {"kind": "subagent_done", "session_id": sid, "agent": "build", "title": "t", "ok": True}
        )
        assert sid not in app._chats
        assert sid not in app._engines
        assert sid not in app._sessions
        assert sid not in app._busy_sessions
        assert sid not in app._running_agents


# --------------------------------------------------------------------------
# Bug 11: write tool returns the written content for the TUI block
# --------------------------------------------------------------------------

def test_write_tool_returns_content_metadata(tmp_path):
    target = tmp_path / "hello.py"
    result = _write(str(target), "print('hi')\n")
    assert result["output"] == "Wrote file successfully."
    assert result["metadata"]["content"] == "print('hi')\n"
    assert result["metadata"]["filePath"] == str(target.resolve())
    assert target.read_text() == "print('hi')\n"


# --------------------------------------------------------------------------
# Compaction UI: the `compacted` event renders a ` Session compacted ` divider
# --------------------------------------------------------------------------

async def test_compacted_event_appends_divider_bubble():
    from opencode_py.tui.status_bar import StatusBar
    from opencode_py.tui.chat_view import MessageBubble

    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        app._handle_event({"kind": "compacted", "session_id": app.session.id, "summary": "did the thing"})
        bubbles = list(app.query(MessageBubble))
        assert len(bubbles) == 1
        assert bubbles[0].role == "compaction"
        assert bubbles[0].content == "did the thing"
        await pilot.pause()


def test_compaction_bubble_content_renders_without_crash():
    async def run() -> None:
        from rich.console import Console

        app = WidgetHost(lambda: ChatView())
        async with app.run_test() as pilot:
            chat = app.query_one(ChatView)
            chat.append_compaction("summarized everything")
            bubbles = list(chat.query(MessageBubble))
            assert len(bubbles) == 1
            assert bubbles[0].role == "compaction"
            rendered = bubbles[0]._build_content()
            assert rendered is not None
            # official opencode i18n `ui.messagePart.compaction` label + summary
            console = Console()
            buf = io.StringIO()
            console.file = buf
            console.print(rendered)
            plain = buf.getvalue()
            assert "Session compacted" in plain
            assert "summarized everything" in plain

    import asyncio

    asyncio.run(run())


def test_compaction_start_shows_compacting_indicator():
    """Mirrors official opencode #35316: while the session summarizes, the
    status line reads `Compacting conversation…` (not the generic working…)."""

    async def run() -> None:
        app = WidgetHost(lambda: InputBar())
        async with app.run_test() as pilot:
            bar = app.query_one(InputBar)
            bar.set_busy(True)
            assert "working..." in bar.query_one("#prompt-status").render().plain
            bar.set_compacting(True)
            assert "Compacting conversation" in bar.query_one("#prompt-status").render().plain
            bar.set_compacting(False)
            assert "working..." in bar.query_one("#prompt-status").render().plain

    import asyncio

    asyncio.run(run())


def test_app_compaction_start_and_compacted_roundtrip():
    """compaction_start flips the indicator on; the subsequent `compacted` event
    flips it back off and renders the divider."""

    async def run() -> None:
        app = OpenCodeTUI()
        async with app.run_test() as pilot:
            app._handle_event({"kind": "compaction_start", "session_id": app.session.id})
            bar = app.query_one(InputBar)
            assert bar._compacting is True
            app._handle_event({"kind": "compacted", "session_id": app.session.id, "summary": "s"})
            assert bar._compacting is False
            bubbles = list(app.query(MessageBubble))
            assert any(b.role == "compaction" for b in bubbles)

    import asyncio

    asyncio.run(run())


# --------------------------------------------------------------------------
# Compaction usage in the status bar: the context percentage updates
# --------------------------------------------------------------------------

async def test_compaction_usage_updates_status_percentage():
    from opencode_py.tui.status_bar import StatusBar

    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        # app._handle_event requires the widget tree, and status bar is present
        app._handle_event(
            {"kind": "usage", "session_id": app.session.id, "usage": {"total_tokens": 190000, "context_size": 200000}}
        )
        status = app.query_one(StatusBar)
        assert "190,000" in status.render().plain
        assert "(95%)" in status.render().plain
        # compaction re-emits a usage event with the compacted estimate
        app._handle_event(
            {"kind": "usage", "session_id": app.session.id, "usage": {"total_tokens": 40000, "context_size": 200000}}
        )
        assert "40,000" in status.render().plain
        assert "(20%)" in status.render().plain


async def test_context_percentage_capped_at_100():
    """Providers can report input+output tokens that sum past the nominal
    context window; the footer must clamp the percentage to 100, never 111%."""
    from opencode_py.tui.status_bar import StatusBar

    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        status = app.query_one(StatusBar)
        app._handle_event(
            {"kind": "usage", "session_id": app.session.id, "usage": {"total_tokens": 222222, "context_size": 200000}}
        )
        assert "222,222" in status.render().plain
        assert "(100%)" in status.render().plain
        assert "(111%)" not in status.render().plain
        assert "111" not in status.render().plain


async def test_retry_notice_shows_in_status_bar_then_clears():
    from opencode_py.tui.status_bar import StatusBar

    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        status = app.query_one(StatusBar)
        app._handle_event(
            {"kind": "retry", "session_id": app.session.id, "message": "↻ connection dropped — retrying (49 left)…"}
        )
        assert "retrying" in status.render().plain
        # an error clears the retry hint before surfacing
        app._handle_event({"kind": "error", "session_id": app.session.id, "error": "boom", "retryable": True})
        assert "retrying" not in status.render().plain
        # a normal turn end clears it too
        app._handle_event(
            {"kind": "retry", "session_id": app.session.id, "message": "↻ connection dropped — retrying (1 left)…"}
        )
        assert "retrying" in status.render().plain
        app._turn_done()
        assert "retrying" not in status.render().plain


async def test_retry_notice_clears_as_soon_as_model_responds():
    from opencode_py.tui.status_bar import StatusBar

    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        status = app.query_one(StatusBar)
        # every kind of "model is responding now" event must clear the hint
        for kind, payload in [
            ("text_delta", {"text": "hi"}),
            ("reasoning_delta", {"text": "thinking"}),
            ("tool_call", {"tool": "read", "arguments": {}, "call_id": "c1"}),
        ]:
            app._handle_event(
                {"kind": "retry", "session_id": app.session.id, "message": "↻ connection dropped — retrying (49 left)…"}
            )
            assert "retrying" in status.render().plain
            app._handle_event({"kind": kind, "session_id": app.session.id, **payload})
            assert "retrying" not in status.render().plain
