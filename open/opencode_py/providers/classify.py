"""Provider error classification (mirrors upstream opencode's provider-error.ts).

A single source of truth for deciding what a provider error *is* so every layer
(the SSE handler, the rotation wrapper, the agent loop) agrees:

- ``is_context_overflow``: the conversation exceeded the model's context window.
  Broad regex coverage of the messages real gateways send (OpenAI, Anthropic,
  OpenRouter, Google, the Zen router, deepseek, …) instead of a few substrings.
- ``is_rate_limit``: a real rate limit / quota exhaustion (the user's lane is
  genuinely done and rotation should move on), kept distinct from overflow.

Crucially ``is_context_overflow`` EXCLUDES rate-limit wording (``rate limit``,
``too many requests``, …) so a 429-style message can never be misread as an
overflow and wrongly trigger compaction instead of failover.
"""

from __future__ import annotations

import re

# Mirrors upstream opencode's context-overflow pattern list (packages/llm/
# src/provider-error.ts) so the same real-world gateway messages are caught.
_CONTEXT_OVERFLOW_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"prompt is too long",
        r"request_too_large",
        r"input is too long for requested model",
        r"exceeds?(?: the)?(\s+the)? context window(?: of [\d,]+ tokens?)?",
        r"exceeds(?: the)? context size",
        r"exceeds (?:the )?(?:model'?s )?maximum context length(?: of [\d,]+ tokens?|\s*\([\d,]+\))?",
        r"input token count.*exceeds(?: the)? max(?:imum)?",
        r"tokens in request more than max tokens allowed",
        r"maximum prompt length is \d+",
        r"reduce the length of the messages",
        r"maximum context length is \d+ tokens",
        r"exceeds (?:the )?maximum allowed input length of [\d,]+ tokens",
        r"input \(\d+ tokens\) is longer than the model'?s context length \(\d+ tokens\)",
        r"exceeds the limit of \d+",
        r"exceeds the available context size",
        r"greater than the context length",
        r"context window exceeds limit",
        r"exceeded model token limit",
        r"context[_ ]?length[_ ]?exceeded",
        r"request entity too large",
        r"context length is only \d+ tokens",
        r"input length.*exceeds.*context length",
        r"prompt too long; exceeded (?:max )?context length",
        r"too large for model with \d+ maximum context length",
        r"prompt has [\d,]+ tokens?, but the configured context size is [\d,]+ tokens",
        r"model_context_window_exceeded",
        r"too many tokens",
        r"token limit exceeded",
        r"context_length_exceeded",
        # "exceeded the max context length of 128000 tokens" (real overflow).
        # Precise: requires the "exceed" verb, so a bare *mention* of "context
        # length" (e.g. "The model context length is 200000…") never misfires.
        r"exceed(?:s|ed|ing)? (?:the )?(?:model'?s )?max(?:imum)? context length",
        r"reduce_other_history",
    )
)

# Messages that merely LOOK like overflow but are really rate limiting.
_CONTEXT_OVERFLOW_EXCLUSIONS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^(throttling error|service unavailable):",
        r"rate limit",
        r"too many requests",
        r"quota",
        r"insufficient",
    )
)

# Messages that identify an in-band provider error as a real rate limit /
# quota exhaustion (the "reached your limit" case). Mirrors upstream opencode's
# classification: the authoritative signal is HTTP 429 (checked by the HTTP
# status path), and the body is only inspected for explicit quota language
# (`insufficient_quota` / `quota exceeded`). We deliberately do NOT substring-
# match the bare word "limit" — free-tier models routinely surface transient
# messages like "model at capacity limit, try later" or "availability is
# limited" that are NOT quota exhaustion, and treating them as rate limits
# would fail over the user's model mid-conversation. Everything else is
# transient and should be retried, not rotated.
_RATE_LIMIT_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\brate\s?limit",  # rate limit / ratelimit / rate-limit / rate_limit
        r"\b429\b",
        r"\btoo many requests\b",
        r"\bthrottl",  # throttled / throttling
        r"\binsufficient[_\-\s]?quota\b",
        r"\bquota\s?(?:exceeded|exhausted|reached|limit)\b",
        r"\btokens?\s+per\s+minute\b",
        r"\brequests?\s+per\s+minute\b",
        r"\bper\s+minute\b",
        r"\busage\s+limit\b",
        r"\bdaily\s+limit\b",
        r"\bmonthly\s+limit\b",
        r"\brequest\s+limit\b",
        r"\bfree\s+(?:tier|plan|model)\s+limit\b",
    )
)


def is_context_overflow(message: str | None) -> bool:
    """True when a provider error means the history overflowed the window."""
    low = (message or "").lower()
    if not low:
        return False
    if any(p.search(low) for p in _CONTEXT_OVERFLOW_EXCLUSIONS):
        return False
    return any(p.search(low) for p in _CONTEXT_OVERFLOW_PATTERNS)


def is_rate_limit(message: str | None) -> bool:
    """True when an in-band error is a real rate limit / quota exhaustion."""
    low = (message or "").lower()
    return any(p.search(low) for p in _RATE_LIMIT_PATTERNS)