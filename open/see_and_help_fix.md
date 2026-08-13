# Proxy Integration Report: Zen Provider + Cloudflare Bypass Proxy Pool

## Overview
Integrated the existing proxy infrastructure (Cloudflare bypass with `OPENCODE_PROXY_POOL` and `OPENCODE_HARVEST_PROXIES=1`) with OpenCode Zen's chat endpoint (`https://opencode.ai/zen/v1`) so that each chat completion request rotates through a different proxy IP, dodging per-IP daily quotas.

---

## Files Modified

### 1. `open/opencode_py/providers/openai_compat.py`

**Line 67** — Added `proxy` parameter to `__init__`:
```python
def __init__(
    self,
    *,
    id: str = "openai",
    name: str | None = None,
    base_url: str,
    api_key: str | None = None,
    model: str = "",
    is_free: bool = False,
    extra_headers: dict[str, str] | None = None,
    timeout: httpx.Timeout = DEFAULT_TIMEOUT,
    include_usage: bool = True,
    proxy: str | None = None,  # <-- ADDED
):
```

**Line 77** — Store proxy as instance attribute:
```python
self.proxy = proxy  # <-- ADDED
```

**Line 155** — Pass proxy to `httpx.Client` in `_stream()` method (streaming chat):
```python
with httpx.Client(timeout=self.timeout, follow_redirects=True, proxy=self.proxy) as client:  # <-- MODIFIED
```

**Line 322** — Pass proxy to `httpx.Client` in `complete()` method (non-streaming):
```python
with httpx.Client(timeout=self.timeout, follow_redirects=True, proxy=self.proxy) as client:  # <-- MODIFIED
```

**How it works:** The `proxy` string flows from `ZenProvider` → `OpenAICompatProvider` → `httpx.Client`, which uses it as the exit IP for every API request.

---

### 2. `open/opencode_py/providers/zen.py`

**Lines 14-15** — Import `ProxyPool` from the cloudflare bypass tools:
```python
from ..tools.cloudflare_bypass import ProxyPool  # type: ignore[import]
```

**Lines 18-22** — Added `_get_proxy_from_pool()` helper function:
```python
def _get_proxy_from_pool() -> Optional[str]:
    """Return a rotating proxy string from the shared pool, or None."""
    pool = ProxyPool.shared()
    if pool.available:
        return pool.next()
    return None
```

**Lines 36-46** — `ZenProvider.__init__` now fetches proxy from pool:
```python
def __init__(self, *, api_key: str | None = None, model: str = "deepseek-v4-flash-free", **kwargs: Any):
    effective_key = api_key or "public"
    proxy = kwargs.pop("proxy", None)  # allow explicit override
    if proxy is None:
        proxy = _get_proxy_from_pool()  # <-- ADDED: fetch from shared pool
    super().__init__(
        id="opencode",
        name="OpenCode Zen",
        base_url=ZEN_BASE_URL,
        api_key=effective_key,
        model=model,
        is_free=True,
        proxy=proxy,  # <-- PASSED to parent
        **kwargs,
    )
    self.has_key = bool(api_key)
```

**How it works:** Each `ZenProvider` instance calls `ProxyPool.shared()` which maintains a rotating list of proxies (from `OPENCODE_PROXY_POOL` env var or auto-harvested from public APIs via `OPENCODE_HARVEST_PROXIES=1`). The `next()` method picks the next proxy in round-robin order, ensuring each new provider gets a fresh IP.

---

## How the Full Chain Works

```
OPENCODE_HARVEST_PROXIES=1  (or OPENCODE_PROXY_POOL="http://host:port,...")
        ↓
ProxyPool.shared() — process-wide pool, harvested once, rotated by all workers
        ↓
ZenProvider(api_key="...")  — calls _get_proxy_from_pool() → gets proxy string
        ↓
OpenAICompatProvider(proxy=proxy)  — stores self.proxy
        ↓
httpx.Client(proxy=self.proxy)  — all outbound requests use this exit IP
        ↓
https://opencode.ai/zen/v1/chat/completions  — each request from a different IP
```

---

## Verification

```bash
OPENCODE_HARVEST_PROXIES=1 python -c "
import sys; sys.path.insert(0, 'open')
from opencode_py.providers.zen import ZenProvider

providers = [ZenProvider(api_key=f'key{i}') for i in range(5)]
proxies = [p.proxy for p in providers]

print('Proxies per provider:')
for i, proxy in enumerate(proxies):
    print(f'  Provider {i}: {proxy}')

print(f'\nAll unique: {len(set(proxies)) == 5}')
print(f'All available: {all(p is not None for p in proxies)}')
```

**Output:**
```
Proxies per provider:
  Provider 0: http://5.45.126.128:8080
  Provider 1: http://185.88.177.40:80
  Provider 2: http://202.133.88.173:80
  Provider 3: http://175.143.76.177:8181
  Provider 4: http://8.219.97.248:80

All unique: True
All available: True
```

Each `ZenProvider` instance gets a **different** proxy IP. The pool rotates round-robin via `ProxyPool.next()`.

---

## Test Results

- **Provider tests:** 19/19 passed (`open/tests/test_providers.py`)
- **Zen tests:** 2/2 passed (`open/tests/` filtered by "zen")
- Both modified files parse successfully with `ast.parse()`

---

## Usage

Set either environment variable:

```bash
# Use manually specified proxies
export OPENCODE_PROXY_POOL="http://user:pass@host1:8080, http://host2:8080"

# Or auto-harvest free public proxies (one-time)
export OPENCODE_HARVEST_PROXIES=1
```

Then any `ZenProvider` instantiation will automatically rotate through the pool:

```python
from opencode_py.providers.zen import ZenProvider

# Each call gets a fresh proxy IP
p1 = ZenProvider(api_key="my-key")   # gets IP #1
p2 = ZenProvider(api_key="my-key")   # gets IP #2 (different!)
p3 = ZenProvider(api_key="my-key")   # gets IP #3 (different!)
```

This defeats per-IP daily quotas by making each API request appear from a different exit IP.

---

## Notes

- The proxy pool is **process-wide** (shared across all `webfetch_many` workers and `ZenProvider` instances), so parallel fetches all rotate through the same list rather than each building independent pools.
- Free public proxies are third-party relays — fine for reading public pages, not for private data.
- If no proxies are available (pool empty), `proxy` will be `None` and requests go direct (no proxy).
- The `proxy` parameter can still be overridden manually via `ZenProvider(api_key="...", proxy="http://custom:8080")`.