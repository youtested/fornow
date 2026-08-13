"""Tests for chat scrolling (don't yank the user to the bottom while reading
history) and the per-turn runtime display (`▣ Build · model · 1m 12s`).
"""

from __future__ import annotations

import time

import pytest
from textual.app import App, ComposeResult

from opencode_py.tui.app import OpenCodeTUI
from opencode_py.tui.chat_view import ChatView, MessageBubble
from opencode_py.tui.input_bar import InputBar, format_duration


class WidgetHost(App):
    def __init__(self, factory) -> None:
        super().__init__()
        self._factory = factory

    def compose(self) -> ComposeResult:
        yield self._factory()


# --------------------------------------------------------------------------
# format_duration mirrors opencode's Locale.duration
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0.1, "100ms"),
        (0.312, "312ms"),
        (12.5, "12.5s"),
        (72.3, "1m 12s"),
        (61.0, "1m 1s"),
        (3600.0, "1h 0m"),
        (3900.0, "1h 5m"),
    ],
)
def test_format_duration(seconds, expected):
    assert format_duration(seconds) == expected


# --------------------------------------------------------------------------
# Scroll: the chat must not yank the user to the bottom while reading history
# --------------------------------------------------------------------------

async def _scrolled_chat(pilot, chat) -> None:
    for i in range(30):
        chat.append_meta(f"line {i} " * 30)
    await pilot.pause(0.2)
    chat.focus()
    await pilot.press("pageup")  # user reads earlier history
    await pilot.pause(0.2)
    assert chat.scroll_y < chat.max_scroll_y


async def test_chat_keeps_position_while_scrolled_up():
    app = WidgetHost(lambda: ChatView())
    async with app.run_test() as pilot:
        chat = app.query_one(ChatView)
        await _scrolled_chat(pilot, chat)
        assert chat._follow_bottom is False
        before = chat.scroll_y
        # new streamed/tool content must NOT yank the view back to the bottom
        chat.stream_delta("new output while the user is reading")
        chat.append_tool({"tool": "bash", "status": "running", "label": "run tests"})
        await pilot.pause(0.2)
        assert chat._follow_bottom is False
        assert chat.scroll_y == before


async def test_chat_resumes_following_at_bottom():
    app = WidgetHost(lambda: ChatView())
    async with app.run_test() as pilot:
        chat = app.query_one(ChatView)
        await _scrolled_chat(pilot, chat)
        # user returns to the bottom -> follow resumes
        await pilot.press("end")
        await pilot.pause(0.2)
        assert chat._follow_bottom is True
        chat.append_meta("newest")
        await pilot.pause(0.2)
        assert chat.scroll_y >= chat.max_scroll_y - 1


async def test_chat_follows_at_bottom_by_default():
    app = WidgetHost(lambda: ChatView())
    async with app.run_test() as pilot:
        chat = app.query_one(ChatView)
        for i in range(30):
            chat.append_meta(f"line {i} " * 30)
        await pilot.pause(0.2)
        assert chat._follow_bottom is True
        chat.append_meta("newest")
        await pilot.pause(0.2)
        assert chat.scroll_y >= chat.max_scroll_y - 1


# --------------------------------------------------------------------------
# Runtime: the mode line shows `▣ Build · model · 1m 12s` after a turn
# --------------------------------------------------------------------------

async def test_turn_done_shows_runtime_in_mode_line():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        bar = app.query_one(InputBar)
        app._turn_started = time.monotonic() - 72.3
        app._turn_had_text = True
        app._turn_done()
        await pilot.pause()
        title = bar.query_one("#prompt-title").render()
        text = str(title)
        assert "1m 12s" in text
        assert "Build" in text


async def test_turn_done_no_runtime_without_started():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        bar = app.query_one(InputBar)
        app._turn_started = None
        app._turn_done()
        await pilot.pause()
        title = bar.query_one("#prompt-title").render()
        text = str(title)
        assert "·" not in text.replace("▣", "") or "m " not in text


# The runtime mirrors official opencode: it appears only on the final report
# (a real text answer), and it disappears the moment the model starts doing
# things again.

async def test_turn_done_no_runtime_for_tool_only_turn():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        bar = app.query_one(InputBar)
        app._turn_started = time.monotonic() - 12.5
        app._turn_had_tools = True
        app._turn_had_text = False
        app._turn_done()
        await pilot.pause()
        assert bar.last_duration == ""


async def test_turn_done_no_runtime_for_error_turn():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        bar = app.query_one(InputBar)
        app._turn_started = time.monotonic() - 12.5
        app._turn_had_text = True
        app._turn_had_error = True
        app._turn_done()
        await pilot.pause()
        assert bar.last_duration == ""


async def test_turn_done_no_runtime_for_interrupted_turn():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        bar = app.query_one(InputBar)
        app._turn_started = time.monotonic() - 12.5
        app._turn_had_text = True
        app._turn_interrupted = True
        app._turn_done()
        await pilot.pause()
        assert bar.last_duration == ""


async def test_new_turn_clears_previous_runtime():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        bar = app.query_one(InputBar)
        app._turn_started = time.monotonic() - 72.3
        app._turn_had_text = True
        app._turn_done()
        await pilot.pause()
        assert bar.last_duration == "1m 12s"
        # the model starts doing things again -> the runtime disappears
        app._clear_last_duration()
        await pilot.pause()
        assert bar.last_duration == ""
        title = bar.query_one("#prompt-title").render()
        assert "1m 12s" not in str(title)


async def test_input_bar_set_last_duration_renders():
    app = WidgetHost(lambda: InputBar())
    async with app.run_test() as pilot:
        bar = app.query_one(InputBar)
        bar.set_header(agent="build", model="opencode/x", provider="opencode", permission_mode="auto")
        bar.set_last_duration("1m 12s")
        await pilot.pause()
        title = bar.query_one("#prompt-title").render()
        assert "1m 12s" in str(title)
