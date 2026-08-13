"""Tests for the second TUI bug-fix round.

Covers: /undo re-entrant call_from_thread crash (A1), delta batching (A2),
tool_call finalizing the assistant bubble (A3), tool-only turns (A4),
busy-guarded slash commands, pruned-session clicks, tool_denied input rows,
the interrupted event, the permission-dialog exit hang, raw config-key
preservation, the model-picker context formatting, and InputBar history
navigation.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
from pathlib import Path

import pytest
from textual.app import App, ComposeResult

from opencode_py.config import Config, save_config
from opencode_py.tui.app import OpenCodeTUI
from opencode_py.tui.chat_view import ChatView, MessageBubble, collapse_tool_output
from opencode_py.tui.input_bar import InputBar, PromptSubmitted
from opencode_py.tui.model_picker import ModelPicker
from opencode_py.tui.settings_screen import SettingsScreen


class FakeEngine:
    agent = "build"
    permission = type("P", (), {"mode": "auto"})()


class WidgetHost(App):
    def __init__(self, factory) -> None:
        super().__init__()
        self._factory = factory

    def compose(self) -> ComposeResult:
        yield self._factory()


async def _mounted_bubble(run: dict) -> MessageBubble:
    host = WidgetHost(lambda: ChatView())
    async with host.run_test() as pilot:
        chat = host.query_one(ChatView)
        chat.append_tool(run)
        bubbles = list(chat.query(MessageBubble))
        return bubbles[-1]


# --------------------------------------------------------------------------
# A1: /undo (and any command that makes the engine emit) must not crash the UI
# thread via a re-entrant call_from_thread.
# --------------------------------------------------------------------------

async def test_undo_command_from_ui_thread_does_not_crash():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        fd, path = tempfile.mkstemp()
        os.write(fd, b"new")
        os.close(fd)
        app.engine._undo_stack.append({"path": path, "original": b"old"})
        app._run_command("/undo")
        await pilot.pause()
        with open(path, "rb") as fh:
            assert fh.read() == b"old"
        os.unlink(path)


async def test_engine_event_from_ui_thread_handled_inline():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        sid = app.session.id
        app._on_engine_event({"kind": "text_delta", "session_id": sid, "text": "hi"})
        app._flush_deltas()
        await pilot.pause()
        chat = app._chat_for(sid)
        assistants = [b for b in chat.query(MessageBubble) if b.role == "assistant"]
        assert assistants and assistants[-1].content == "hi"


# --------------------------------------------------------------------------
# A2: deltas are batched into a single render instead of one per token.
# --------------------------------------------------------------------------

async def test_delta_batching_renders_once():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        sid = app.session.id
        chat = app._chat_for(sid)
        app._on_engine_event({"kind": "text_delta", "session_id": sid, "text": "a"})
        app._on_engine_event({"kind": "text_delta", "session_id": sid, "text": "b"})
        app._on_engine_event({"kind": "text_delta", "session_id": sid, "text": "c"})
        # buffered, not yet rendered to a bubble
        assert chat._stream_bubble is None
        app._flush_deltas()
        await pilot.pause()
        assistants = [b for b in chat.query(MessageBubble) if b.role == "assistant"]
        assert len(assistants) == 1
        assert assistants[0].content == "abc"


# --------------------------------------------------------------------------
# A3: a tool_call finalizes the assistant bubble; the next step's text must
# land in a fresh bubble (no merged text, no stale cursor).
# --------------------------------------------------------------------------

async def test_tool_call_finalizes_stream_and_new_text_is_new_bubble():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        sid = app.session.id
        chat = app._chat_for(sid)
        app._on_engine_event({"kind": "text_delta", "session_id": sid, "text": "Let me"})
        app._on_engine_event({"kind": "text_delta", "session_id": sid, "text": " check"})
        app._on_engine_event(
            {
                "kind": "tool_call",
                "session_id": sid,
                "tool": "glob",
                "arguments": {"pattern": "*.py"},
                "call_id": "c1",
            }
        )
        await pilot.pause()
        bubbles = list(chat.query(MessageBubble))
        assistants = [b for b in bubbles if b.role == "assistant"]
        assert assistants and assistants[-1].content == "Let me check"
        assert assistants[-1].streaming is False
        tools = [b for b in bubbles if b.role == "tool"]
        assert tools and tools[-1].content.get("tool") == "glob"
        # a new tool-loop step's text must not merge into the previous bubble
        app._on_engine_event({"kind": "text_delta", "session_id": sid, "text": "Found"})
        app._flush_deltas()
        await pilot.pause()
        assistants = [b for b in chat.query(MessageBubble) if b.role == "assistant"]
        assert len(assistants) == 2
        assert assistants[-1].content == "Found"


async def test_reasoning_then_tool_call_does_not_leave_stream_bubble():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        sid = app.session.id
        chat = app._chat_for(sid)
        app._on_engine_event({"kind": "reasoning_delta", "session_id": sid, "text": "think"})
        app._flush_deltas()
        app._on_engine_event(
            {
                "kind": "tool_call",
                "session_id": sid,
                "tool": "bash",
                "arguments": {"command": "ls"},
                "call_id": "c2",
            }
        )
        await pilot.pause()
        bubbles = list(chat.query(MessageBubble))
        # no empty assistant stream bubble lingering above the tool row
        assert not any(b.role == "assistant" and b.content == "" for b in bubbles)


async def test_reasoning_header_shows_thought_duration():
    """The collapsed reasoning header must show how long the model thought:
    `+ Thought for X.Xs` (mirrors opencode), once reasoning ends."""
    from rich.console import Console

    host = WidgetHost(lambda: ChatView())
    async with host.run_test() as pilot:
        chat = host.query_one(ChatView)
        chat.stream_reasoning_delta("think carefully")
        await pilot.pause()
        bubble = chat.last_reasoning()
        assert bubble is not None
        assert bubble.streaming is True
        # while streaming it's the spinner; after end_reasoning the duration shows
        chat.end_reasoning()
        await pilot.pause()
        assert bubble.streaming is False
        assert bubble._thought_seconds is not None
        console = Console(width=120, record=True)
        console.print(bubble._build_content())
        plain = console.export_text()
        assert "Thought for" in plain
        assert f"{bubble._thought_seconds:.1f}s" in plain


async def test_eager_thinking_bubble_appears_immediately():
    """Pressing Enter must mount an immediate `Thinking...` bubble, before the
    provider sends anything back (the eager placeholder)."""
    from rich.console import Console

    host = WidgetHost(lambda: ChatView())
    async with host.run_test() as pilot:
        chat = host.query_one(ChatView)
        chat.begin_thinking()
        await pilot.pause()
        bubble = chat.last_reasoning()
        assert bubble is not None
        assert bubble.streaming is True
        assert bubble.content == ""
        # the spinner + Thinking label is visible right away
        console = Console(width=120, record=True)
        console.print(bubble._build_content())
        assert "Thinking" in console.export_text()
        # if no reasoning ever arrives, end_reasoning drops the placeholder
        chat.end_reasoning()
        await pilot.pause()
        assert list(chat.query(MessageBubble)) == []
        assert chat.last_reasoning() is None


async def test_eager_thinking_receives_real_reasoning():
    """A reasoning delta must stream into the eager placeholder bubble (single
    bubble, not a second one), and end up as a `Thought for X.Xs` header."""
    from rich.console import Console

    host = WidgetHost(lambda: ChatView())
    async with host.run_test() as pilot:
        chat = host.query_one(ChatView)
        chat.begin_thinking()
        chat.stream_reasoning_delta("here is my plan")
        await pilot.pause()
        bubbles = list(chat.query(MessageBubble))
        assert len(bubbles) == 1
        assert bubbles[0].role == "reasoning"
        assert bubbles[0].content == "here is my plan"
        chat.end_reasoning()
        await pilot.pause()
        console = Console(width=120, record=True)
        console.print(bubbles[0]._build_content())
        plain = console.export_text()
        assert "Thought for" in plain


# --------------------------------------------------------------------------
# A4: a tool-only turn must not claim "no reply from the model".
# --------------------------------------------------------------------------

async def test_tool_only_turn_does_not_report_no_reply():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        sid = app.session.id
        app._turn_had_tools = True
        app._turn_done(result=None)
        chat = app._chat_for(sid)
        metas = [b.content for b in chat.query(MessageBubble) if b.role == "meta"]
        assert not any("no reply from the model" in str(m) for m in metas)


async def test_turn_done_still_reports_no_reply_when_nothing_happened():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        app._turn_done(result=None)
        chat = app._chat_for(app.session.id)
        metas = [b.content for b in chat.query(MessageBubble) if b.role == "meta"]
        assert any("no reply from the model" in str(m) for m in metas)


# --------------------------------------------------------------------------
# Busy guard: mutating slash commands are blocked while a turn runs.
# --------------------------------------------------------------------------

async def test_busy_blocks_mutating_command():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        app._busy = True
        app.on_prompt_submitted(PromptSubmitted("/undo"))
        chat = app._chat_for(app.session.id)
        metas = [b.content for b in chat.query(MessageBubble) if b.role == "meta"]
        assert any("still working" in str(m) for m in metas)


async def test_busy_allows_safe_command():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        app._busy = True
        ran: list[str] = []
        app._run_command = lambda line: ran.append(line)
        app.on_prompt_submitted(PromptSubmitted("/help"))
        assert ran == ["/help"]


# --------------------------------------------------------------------------
# Pruned sub-agent: clicking its task row must not open an empty chat wired to
# the main engine.
# --------------------------------------------------------------------------

async def test_switch_to_pruned_session_does_not_switch():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        app._pruned.add("dead")
        app._switch_session("dead")
        assert app._current_session_id == app.session.id


async def test_subagent_done_marks_pruned_session():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        sid = "sub1"
        chat = app._chat_for(sid)
        app._chats[sid] = chat
        app._engines[sid] = FakeEngine()
        app._sessions[sid] = type("S", (), {"completed": None})()
        app._busy_sessions.add(sid)
        app._running_agents[sid] = "t · build"
        app._on_subagent_done(
            {"kind": "subagent_done", "session_id": sid, "agent": "build", "title": "t", "ok": True}
        )
        assert sid in app._pruned
        assert sid not in app._chats


# --------------------------------------------------------------------------
# tool_denied must render a row with the tool input even without a prior
# tool_call event.
# --------------------------------------------------------------------------

async def test_tool_denied_appends_input_row():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        sid = app.session.id
        chat = app._chat_for(sid)
        app._on_engine_event(
            {
                "kind": "tool_denied",
                "session_id": sid,
                "tool": "write",
                "reason": "file not read first",
                "call_id": "c9",
                "input": {"filePath": "x.py"},
            }
        )
        await pilot.pause()
        tools = [b for b in chat.query(MessageBubble) if b.role == "tool"]
        assert tools
        assert tools[-1].content.get("tool") == "write"
        assert tools[-1].content.get("input") == {"filePath": "x.py"}
        assert tools[-1].content.get("output") == "file not read first"


# --------------------------------------------------------------------------
# The interrupted event is surfaced instead of silently dropped.
# --------------------------------------------------------------------------

async def test_interrupted_event_shows_meta_and_marks_turn():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        sid = app.session.id
        chat = app._chat_for(sid)
        app._on_engine_event({"kind": "interrupted", "session_id": sid})
        await pilot.pause()
        assert app._turn_interrupted is True
        metas = [b.content for b in chat.query(MessageBubble) if b.role == "meta"]
        assert any("Interrupted" in str(m) for m in metas)


# --------------------------------------------------------------------------
# Permission dialog: quitting the app must unblock the engine thread quickly.
# --------------------------------------------------------------------------

async def test_permission_ask_unblocks_on_exit():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        holder: dict[str, str] = {}

        def worker() -> None:
            holder["result"] = app._permission_ask("run this command?", [])

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        await asyncio.sleep(0.05)
        app._exit_requested.set()
        t.join(timeout=3)
        assert not t.is_alive(), "permission ask hung after exit"
        assert holder.get("result") == "reject"


# --------------------------------------------------------------------------
# Config: save_config must preserve unknown raw keys (mcpServers/plugins/tools).
# --------------------------------------------------------------------------

def test_save_config_preserves_raw_keys(tmp_path):
    cfg = Config.from_dict(
        {
            "model": "opencode/foo",
            "mcpServers": {"local": {"command": "npx"}},
            "plugins": ["@opencode/plugin-ts"],
            "tools": {"bash": {"deny": "*"}},
        },
        Path("."),
    )
    p = tmp_path / "opencode.json"
    save_config(cfg, path=p)
    data = json.loads(p.read_text())
    assert data["model"] == "opencode/foo"
    assert data["mcpServers"] == {"local": {"command": "npx"}}
    assert data["plugins"] == ["@opencode/plugin-ts"]
    assert data["tools"] == {"bash": {"deny": "*"}}


def test_save_config_known_keys_override_raw():
    cfg = Config.from_dict({"model": "opencode/old", "theme": "solarized"}, Path("."))
    cfg.theme = "opencode"
    p = Path(tempfile.mkdtemp()) / "opencode.json"
    save_config(cfg, path=p)
    data = json.loads(p.read_text())
    assert data["theme"] == "opencode"


# --------------------------------------------------------------------------
# Model picker: "128k"-style context strings must not crash int().
# --------------------------------------------------------------------------

def test_format_context_handles_k_and_junk():
    from opencode_py.tui.model_picker import _format_context

    assert _format_context(128000) == "128,000"
    assert _format_context("128k") == "128,000"
    assert _format_context("1m") == "1,000,000"
    assert _format_context("junk") == "junk"
    assert _format_context(None) == "?"
    assert _format_context(0) == "0"


# --------------------------------------------------------------------------
# Chat view: failed tools surface an error line; long write output collapses.
# --------------------------------------------------------------------------

async def test_error_line_shows_failed_tool_error():
    b = await _mounted_bubble({"tool": "read", "status": "error", "error": "No such file"})
    err = b._error_line(b.content)
    assert err is not None and "No such file" in str(err)


async def test_error_line_hidden_for_denial():
    b = await _mounted_bubble({"tool": "read", "status": "error", "output": "user dismissed"})
    assert b._error_line(b.content) is None


def test_write_render_collapses_long_content():
    long = "\n".join(f"line {i}" for i in range(200))
    collapsed = collapse_tool_output(long, 10, 10 * 80)
    assert collapsed["overflow"] is True
    assert "line 199" not in collapsed["output"]
    short = collapse_tool_output("tiny", 10, 10 * 80)
    assert short["overflow"] is False
    assert short["output"] == "tiny"


async def test_write_tool_block_uses_metadata_content():
    b = await _mounted_bubble(
        {
            "tool": "write",
            "status": "completed",
            "input": {"filePath": "x.py"},
            "metadata": {"content": "print('hi')\n"},
        }
    )
    assert b._tool_block() is True


# --------------------------------------------------------------------------
# Settings: the "small model" picker must not retarget the app engine.
# --------------------------------------------------------------------------

def test_small_model_row_does_not_propagate():
    screen = SettingsScreen(cfg=Config(), engine=FakeEngine(), auth=None)
    rows = screen._build_rows()
    model_row = next(r for r in rows if r.label == "model")
    small_row = next(r for r in rows if r.label == "small model")
    assert model_row.propagate is True
    assert small_row.propagate is False


# --------------------------------------------------------------------------
# InputBar history: repeated Up must not clobber the typed draft, Down must
# restore it, and Down/Up with no history must never wipe the input.
# --------------------------------------------------------------------------

async def _mounted_bar(history: list[str], pilot) -> InputBar:
    bar = pilot.app.query_one(InputBar)
    bar._history = list(history)
    bar._hist_index = len(bar._history)
    return bar


async def test_repeated_up_preserves_typed_draft():
    host = WidgetHost(lambda: InputBar())
    async with host.run_test() as pilot:
        bar = await _mounted_bar(["c1", "c2"], pilot)
        bar.input.value = "my draft"
        assert bar._handle_arrow("up") is True
        assert bar.input.value == "c2"
        assert bar._handle_arrow("up") is True
        assert bar.input.value == "c1"
        assert bar._draft == "my draft"


async def test_down_restores_draft_at_end_of_history():
    host = WidgetHost(lambda: InputBar())
    async with host.run_test() as pilot:
        bar = await _mounted_bar(["c1", "c2"], pilot)
        bar.input.value = "my draft"
        bar._handle_arrow("up")
        bar._handle_arrow("up")
        assert bar._handle_arrow("down") is True
        assert bar.input.value == "c2"
        assert bar._handle_arrow("down") is True
        assert bar.input.value == "my draft"
        assert bar._hist_index == 2


async def test_down_with_no_history_keeps_typed_input():
    host = WidgetHost(lambda: InputBar())
    async with host.run_test() as pilot:
        bar = await _mounted_bar([], pilot)
        bar.input.value = "typed text"
        assert bar._handle_arrow("down") is False
        assert bar.input.value == "typed text"


async def test_up_with_no_history_keeps_typed_input():
    host = WidgetHost(lambda: InputBar())
    async with host.run_test() as pilot:
        bar = await _mounted_bar([], pilot)
        bar.input.value = "typed text"
        assert bar._handle_arrow("up") is False
        assert bar.input.value == "typed text"


# --------------------------------------------------------------------------
# Model picker: providers as headers with models underneath, free first,
# and a search box that filters the rendered rows.
# --------------------------------------------------------------------------

class PickerHost(App):
    """App that mounts a ModelPicker directly (so populate can be inspected)."""

    def __init__(self, factory) -> None:
        super().__init__()
        self._factory = factory

    def compose(self) -> ComposeResult:
        yield self._factory()


class NavPicker(ModelPicker):
    """ModelPicker with the network worker disabled so tests are deterministic."""

    def _start_worker(self) -> None:
        pass


class PickerPushHost(App):
    """App that pushes a NavPicker via push_screen (loads its CSS) and can
    capture the chosen "provider/model" on dismiss."""

    def __init__(self) -> None:
        super().__init__()
        self._picker = NavPicker()
        self.choice = None

    def on_mount(self) -> None:
        self.push_screen(self._picker, self._on_choice)

    def _on_choice(self, choice: str | None) -> None:
        self.choice = choice


def test_picker_row_label_free_and_current():
    from opencode_py.tui.model_picker import _model_row_label

    row = _model_row_label("opencode/deepseek-v4-flash-free", {"id": "deepseek-v4-flash-free", "name": "DeepSeek V4 Flash", "free": True}, "opencode/deepseek-v4-flash-free")
    text = row.render().plain
    assert "FREE" in text
    assert "DeepSeek V4 Flash" in text
    # current model marked with the bullet
    assert "\u25cf" in text


def test_picker_row_label_free_sort_key():
    from opencode_py.tui.model_picker import _model_row_label

    free_row = _model_row_label("p/a", {"id": "a", "free": True}, "")
    paid_row = _model_row_label("p/b", {"id": "b", "free": False}, "")
    assert "FREE" in free_row.render().plain
    assert "FREE" not in paid_row.render().plain


async def test_picker_renders_providers_with_models_and_free_first():
    from opencode_py.tui.model_picker import ModelPicker

    picker = ModelPicker()
    host = PickerHost(lambda: picker)
    async with host.run_test() as pilot:
        await pilot.pause()
        picker.populate(
            {
                "openai": [{"id": "gpt-4o", "name": "GPT-4o", "free": False}],
                "groq": [{"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B", "free": True}],
            }
        )
        await pilot.pause()
        lv = picker.query_one("#models-list")
        rendered = [str(item.children[0].render().plain) for item in lv.query("ListItem")]
        # both provider headers present
        assert any("OpenAI" in r for r in rendered)
        assert any("Groq" in r for r in rendered)
        # free provider's models appear before paid provider's models
        groq_idx = next(i for i, r in enumerate(rendered) if "Groq" in r)
        openai_idx = next(i for i, r in enumerate(rendered) if "OpenAI" in r)
        assert groq_idx < openai_idx
        # free model carries the FREE tag
        assert any("Llama 3.3 70B" in r and "FREE" in r for r in rendered)


async def test_zen_section_groups_free_first_then_family():
    # OpenCode Zen mixes many upstream vendors; free models must come first
    # under a "Free" sub-group, then non-free models grouped by upstream family.
    picker = ModelPicker()
    host = PickerHost(lambda: picker)
    async with host.run_test() as pilot:
        await pilot.pause()
        picker.populate(
            {
                "opencode": [
                    {"id": "gpt-5.6-sol", "name": "GPT-5.6 Sol", "free": False},
                    {"id": "claude-opus-4-5", "name": "Claude Opus 4.5", "free": False},
                    {"id": "deepseek-v4-flash-free", "name": "DeepSeek V4 Flash", "free": True},
                    {"id": "big-pickle", "name": "Big Pickle", "free": True},
                    {"id": "gemini-3.6-flash", "name": "Gemini 3.6 Flash", "free": False},
                ]
            }
        )
        await pilot.pause()
        lv = picker.query_one("#models-list")
        rendered = [str(item.children[0].render().plain) for item in lv.query("ListItem")]
        # header first
        assert rendered[0] == "  OpenCode Zen"
        # free sub-group comes before any family sub-group...
        free_idx = rendered.index("   Free")
        gpt_idx = rendered.index("   OpenAI")
        anthropic_idx = rendered.index("   Anthropic")
        google_idx = rendered.index("   Google")
        assert free_idx < min(gpt_idx, anthropic_idx, google_idx)
        # ...and free models sit right under "Free"
        assert rendered[free_idx + 1] == "   Big Pickle FREE"
        assert rendered[free_idx + 2] == "   DeepSeek V4 Flash FREE"
        # family sub-groups sorted alphabetically by family name
        assert anthropic_idx < google_idx < gpt_idx


async def test_zen_family_maps_upstream_vendor():
    from opencode_py.tui.model_picker import _zen_family

    assert _zen_family("gpt-5.6-sol") == "OpenAI"
    assert _zen_family("claude-opus-4-5") == "Anthropic"
    assert _zen_family("gemini-3.6-flash") == "Google"
    assert _zen_family("kimi-k3") == "Moonshot"
    assert _zen_family("grok-4.6") == "xAI"
    assert _zen_family("qwen3.6-plus") == "Alibaba"
    assert _zen_family("glm-5.2") == "Zhipu AI"
    assert _zen_family("pizza-bot") == "Other"


async def test_picker_search_filters_models():
    from opencode_py.tui.model_picker import ModelPicker
    from textual.widgets import Input

    picker = ModelPicker()
    host = PickerHost(lambda: picker)
    async with host.run_test() as pilot:
        await pilot.pause()
        picker.populate(
            {
                "openai": [
                    {"id": "gpt-4o", "name": "GPT-4o", "free": False},
                    {"id": "gpt-4o-mini", "name": "GPT-4o mini", "free": False},
                ]
            }
        )
        await pilot.pause()
        box = picker.query_one("#models-search", Input)
        box.value = "mini"
        picker._populate_list()
        await pilot.pause()
        lv = picker.query_one("#models-list")
        rendered = [str(item.children[0].render().plain) for item in lv.query("ListItem")]
        assert any("GPT-4o mini" in r for r in rendered)
        assert not any(r.endswith("GPT-4o") for r in rendered if "mini" not in r)
        # header remains so the provider context is visible
        assert any("OpenAI" in r for r in rendered)


# --------------------------------------------------------------------------
# Regression: the picker's list-builder must not be named `_render`, which
# would shadow Textual's internal Widget._render() (returns the widget's
# Visual). When the screen gets flagged dirty, Textual calls `_render()` on it;
# a wrong return type crashes the compositor with
# "AttributeError: 'NoneType' object has no attribute 'render_strips'".
# --------------------------------------------------------------------------

def test_picker_does_not_shadow_textual_render():
    from opencode_py.tui.model_picker import ModelPicker

    assert ModelPicker._render.__qualname__ == "Widget._render"
    # the method that builds the list rows must exist under the renamed id
    assert hasattr(ModelPicker, "_populate_list")


async def test_picker_survives_compositor_rerender():
    from opencode_py.tui.model_picker import ModelPicker

    picker = ModelPicker()
    host = PickerHost(lambda: picker)
    async with host.run_test() as pilot:
        await pilot.pause()
        picker.populate(
            {
                "openai": [{"id": "gpt-4o", "name": "GPT-4o", "free": False}],
            }
        )
        await pilot.pause()
        # force the screen (a widget itself) to re-render its own content as
        # the compositor does after any layout/style invalidation
        for _ in range(3):
            picker.refresh()
            await pilot.pause()
        lv = picker.query_one("#models-list")
        assert len(list(lv.query("ListItem"))) == 2  # header + model row


# --------------------------------------------------------------------------
# Ctrl+M opens the model picker. On most terminals Ctrl+M sends the same
# byte as Enter, so the picker must also open when Enter is pressed on an
# empty prompt; a non-empty Enter still submits instead.
# --------------------------------------------------------------------------

async def _run_models_press(keys: list[str]) -> str:
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        await pilot.pause()
        for key in keys:
            await pilot.press(key)
            await pilot.pause()
        return type(app.screen).__name__


async def test_ctrl_m_opens_model_picker():
    assert (await _run_models_press(["ctrl+m"])) == "ModelPicker"


async def test_empty_enter_opens_model_picker():
    # a real terminal delivers Ctrl+M as the Enter byte (\r), which Textual
    # normalizes to "enter" -- the binding would never fire otherwise
    assert (await _run_models_press(["enter"])) == "ModelPicker"


async def test_typed_enter_submits_not_opens_picker():
    from opencode_py.tui.input_bar import InputBar

    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one(InputBar).input.value = "hello world"
        await pilot.press("enter")
        await pilot.pause()
        assert type(app.screen).__name__ != "ModelPicker"


# --------------------------------------------------------------------------
# Regression: the picker's CSS used to be a module-level constant that was
# never attached to the class, so none of the styling applied (providers and
# models rendered the same default color). Verify the CSS is loaded and the
# key rules (purple provider headers, borderless search bar) take effect.
# --------------------------------------------------------------------------

def test_picker_css_is_attached_to_class():
    from opencode_py.tui.model_picker import ModelPicker

    assert hasattr(ModelPicker, "CSS") and ModelPicker.CSS.strip()
    assert "#models-search" in ModelPicker.CSS
    assert "group-header" in ModelPicker.CSS


async def test_picker_header_and_search_styles_apply():
    # push_screen is what triggers Textual's `_load_screen_css`, i.e. the
    # exact path the real app uses to attach the picker's CSS
    host = PickerPushHost()
    async with host.run_test() as pilot:
        await pilot.pause()
        host._picker.populate(
            {"groq": [{"id": "llama-3.3-70b-versatile", "name": "L", "free": True}]}
        )
        await pilot.pause()
        lv = host._picker.query_one("#models-list")
        header = list(lv.query("ListItem"))[0].children[0]
        assert header.styles.color.rgb == (157, 124, 216)  # #9d7cd8 purple
        box = host._picker.query_one("#models-search")
        assert box.styles.border.top[0] in ("none", "")
        # header row carries the title and the esc hint
        header_row = host._picker.query_one("#models-header")
        assert [c.render().plain for c in header_row.children] == ["Models", "esc"]


# --------------------------------------------------------------------------
# Navigation: the search box keeps focus (like opentui's dialog filter) while
# Up/Down move the highlight across the models (skipping provider headers) and
# Enter selects the highlighted model and dismisses with "provider/model".
# --------------------------------------------------------------------------

PICKER_DATA = {
    "groq": [{"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B", "free": True}],
    "openai": [
        {"id": "gpt-4o", "name": "GPT-4o", "free": False},
        {"id": "gpt-4o-mini", "name": "GPT-4o mini", "free": False},
    ],
}


async def test_picker_arrows_move_between_model_rows():
    host = PickerPushHost()
    async with host.run_test() as pilot:
        await pilot.pause()
        host._picker.populate(PICKER_DATA)
        await pilot.pause()
        lv = host._picker.query_one("#models-list")
        # rows: 0 Groq header, 1 llama, 2 OpenAI header, 3 gpt-4o, 4 gpt-4o-mini
        assert lv.index == 1  # first model row, not the header
        await pilot.press("down")
        await pilot.pause()
        assert lv.index == 3  # skips the OpenAI header
        await pilot.press("down")
        await pilot.pause()
        assert lv.index == 4
        await pilot.press("down")
        await pilot.pause()
        assert lv.index == 4  # clamped at the last model
        await pilot.press("up")
        await pilot.pause()
        assert lv.index == 3


async def test_picker_enter_selects_and_dismisses():
    host = PickerPushHost()
    async with host.run_test() as pilot:
        await pilot.pause()
        host._picker.populate(PICKER_DATA)
        await pilot.pause()
        await pilot.press("down")  # llama -> gpt-4o
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        assert host.choice == "openai/gpt-4o"


async def test_picker_search_filter_enter_selects():
    host = PickerPushHost()
    async with host.run_test() as pilot:
        await pilot.pause()
        host._picker.populate(PICKER_DATA)
        await pilot.pause()
        box = host._picker.query_one("#models-search")
        box.value = "gpt-4o-mini"
        await pilot.pause()
        lv = host._picker.query_one("#models-list")
        # only OpenAI header + gpt-4o-mini remain: the model row is index 1
        assert lv.index == 1
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        assert host.choice == "openai/gpt-4o-mini"


async def test_picker_escape_dismisses_without_choice():
    host = PickerPushHost()
    async with host.run_test() as pilot:
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert host.choice is None
