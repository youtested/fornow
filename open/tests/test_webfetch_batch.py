"""Parallel webfetch (webfetch_many): fan-out batch fetches + shared proxy pool."""

import os
import threading
from collections import Counter
from unittest import mock

from opencode_py.tools import webfetch as wf


def test_batch_is_parallel_deterministic_and_orders_results():
    # A barrier(4) only releases once four worker threads are inside at once.
    # If webfetch_many degraded to sequential execution the first call would
    # block 3s, break the barrier, and the future would raise — a deterministic
    # concurrency proof that does not depend on wall-clock timing.
    barrier = threading.Barrier(4)
    entered: list[str] = []

    def fake_fetch(url, fmt="markdown", timeout=30):
        barrier.wait(timeout=3)
        entered.append(url)
        return {"output": f"content for {url}", "error": None}

    with mock.patch.object(wf, "_webfetch", side_effect=fake_fetch):
        r = wf.webfetch_many(
            ["http://a", "http://b", "http://c", "http://d"],
            max_concurrent=4,
            timeout=30,
        )

    assert r.get("error") is None
    assert r["metadata"]["count"] == 4
    assert r["metadata"]["succeeded"] == 4
    assert r["metadata"]["failed"] == 0
    assert r["metadata"]["concurrency"] == 4
    assert len(entered) == 4  # all four passed the barrier together

    body = r["output"]
    # results come back in submission order, keyed by URL
    assert body.index("content for http://a") < body.index("content for http://b")


def test_batch_dedupes_and_caps_url_count():
    urls = ["http://a", "http://a", "http://b"] + [f"http://{i}.page" for i in range(60)]
    with mock.patch.object(wf, "_webfetch", return_value={"output": "x", "error": None}) as m:
        r = wf.webfetch_many(urls)
    assert m.call_count == 50  # 2 deduped, then capped at 50
    assert r["metadata"]["count"] == 50
    assert r["metadata"]["dropped"] == 12
    assert len(r["metadata"]["dropped_urls"]) == 12
    assert "dropped" in r["output"]


def test_batch_reports_per_url_failures():
    def fake(url, fmt="markdown", timeout=30):
        if "ok" in url:
            return {"output": "fine", "error": None}
        return {"output": "Fetch failed: HTTP 403", "error": True}

    with mock.patch.object(wf, "_webfetch", side_effect=fake):
        r = wf.webfetch_many(["http://ok", "http://bad"])

    assert r["metadata"]["succeeded"] == 1
    assert r["metadata"]["failed"] == 1
    assert "FAILED" in r["output"]
    assert "Fetch failed: HTTP 403" in r["output"]


def test_batch_requires_nonempty_array():
    assert wf.webfetch_many([]).get("error") is True
    assert wf.webfetch_many("http://not-a-list").get("error") is True
    assert wf.webfetch_many(["http://a", 42]).get("error") is True


def test_batch_content_limit_truncates_each_url():
    with mock.patch.object(wf, "_webfetch", return_value={"output": "A" * 5000, "error": None}):
        r = wf.webfetch_many(["http://a"], content_limit=100)
    assert "[truncated]" in r["output"]
    assert "A" * 100 in r["output"]
    assert "A" * 101 not in r["output"]


def test_batch_tool_is_registered_with_webfetch_permission():
    from opencode_py.tools import build_registry

    reg = build_registry()
    t = reg.get("webfetch_many")
    assert t is not None
    assert t.permission == "webfetch"
    schema = t.parameters
    assert schema["required"] == ["urls"]
    assert schema["properties"]["urls"]["type"] == "array"


# --------------------------------------------------------------------------
# Shared proxy pool: parallel fetches rotate through one process-wide pool and
# it is rebuilt when the env that feeds it changes.
# --------------------------------------------------------------------------

def test_proxy_pool_next_is_thread_safe_round_robin():
    from opencode_py.tools.cloudflare_bypass import ProxyPool

    pool = ProxyPool(proxies=[f"http://p{i}:1" for i in range(10)], harvest=False)
    got: list = []
    lock = threading.Lock()

    def worker():
        for _ in range(50):
            with lock:
                got.append(pool.next())

    ts = [threading.Thread(target=worker) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    assert len(got) == 200
    assert Counter(got) == {f"http://p{i}:1": 20 for i in range(10)}


def test_shared_pool_is_singleton_and_rebuilds_on_env_change():
    from opencode_py.tools.cloudflare_bypass import ProxyPool, _reset_shared_pool

    try:
        with mock.patch.dict(
            os.environ, {"OPENCODE_PROXY_POOL": "http://1.1.1.1:80"}, clear=False
        ):
            _reset_shared_pool()
            p1 = ProxyPool.shared()
            assert p1.available is True
            assert p1.next() == "http://1.1.1.1:80"
            assert ProxyPool.shared() is p1  # cached singleton

        with mock.patch.dict(
            os.environ, {"OPENCODE_PROXY_POOL": "http://2.2.2.2:80"}, clear=False
        ):
            p2 = ProxyPool.shared()
            assert p2 is not p1  # env changed -> rebuilt
            assert p2.next() == "http://2.2.2.2:80"
    finally:
        _reset_shared_pool()


def test_bypass_fetch_hands_the_shared_pool_to_rotation():
    from opencode_py.tools.cloudflare_bypass import _reset_shared_pool

    try:
        with mock.patch.dict(
            os.environ, {"OPENCODE_PROXY_POOL": "http://127.0.0.1:1"}, clear=False
        ):
            _reset_shared_pool()
            with mock.patch.object(wf, "UltimateBypass") as UB:
                UB.return_value.fetch.return_value = {
                    "success": True,
                    "method": "requests",
                    "content": "<h1>ok</h1>",
                }
                r = wf._bypass_fetch("https://example.com", "markdown", 10, False, hint=403)
                assert r.get("error") is None
                # the bypass was given the shared (singleton) pool
                pool = UB.call_args.kwargs["proxy_pool"]
                assert pool.next() == "http://127.0.0.1:1"
    finally:
        _reset_shared_pool()
