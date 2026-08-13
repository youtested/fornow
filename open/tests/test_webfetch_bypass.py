"""webfetch tool: Cloudflare-bypass fallback integration tests."""

import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

from opencode_py.tools import webfetch as wf


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def _send(self, status, body, ctype="text/html"):
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/blocked"):
            self._send(
                403,
                "<html><title>Just a moment...</title>"
                "<body>Checking your browser before accessing. "
                "This process is automatic.</body></html>",
            )
        elif self.path.startswith("/plain"):
            self._send(200, "plain hello")
        else:
            self._send(200, "<h1>Real</h1><p>content here</p>")


def _start_server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, t


def test_normal_fetch_converts_html():
    srv, t = _start_server()
    try:
        url = f"http://127.0.0.1:{srv.server_port}/ok"
        r = wf._webfetch(url, "markdown", 10)
        assert r.get("error") is None
        assert r["output"].strip() == "Real\ncontent here"
    finally:
        srv.shutdown()


def test_plain_text_passthrough():
    srv, t = _start_server()
    try:
        url = f"http://127.0.0.1:{srv.server_port}/plain"
        r = wf._webfetch(url, "text", 10)
        assert r.get("error") is None
        assert r["output"].strip() == "plain hello"
    finally:
        srv.shutdown()


def test_blocked_fallback_bypasses():
    srv, t = _start_server()
    fake = {"success": True, "method": "requests", "content": "<h1>Bypassed</h1><p>ok</p>"}
    try:
        with mock.patch.object(wf, "UltimateBypass") as UB:
            UB.return_value.fetch.return_value = fake
            url = f"http://127.0.0.1:{srv.server_port}/blocked"
            r = wf._webfetch(url, "markdown", 10)
            assert r.get("error") is None
            assert r["output"].strip() == "Bypassed\nok"
            assert r["metadata"]["bypassed"] is True
            assert r["metadata"]["bypass_method"] == "requests"
            UB.return_value.fetch.assert_called_once_with(url)
    finally:
        srv.shutdown()


def test_blocked_fallback_failure_reports_error():
    srv, t = _start_server()
    fake = {"success": False, "method": "requests", "error": "Status: 403"}
    try:
        with mock.patch.object(wf, "UltimateBypass") as UB:
            UB.return_value.fetch.return_value = fake
            url = f"http://127.0.0.1:{srv.server_port}/blocked"
            r = wf._webfetch(url, "markdown", 10)
            assert r.get("error") is True
            assert "bypass failed" in r["output"]
            assert "HTTP 403" in r["output"]
    finally:
        srv.shutdown()


def test_looks_like_block_heuristic():
    assert wf._looks_like_block("Just a moment... checking your browser") is True
    assert wf._looks_like_block("<html>cf-chl challenge</html>") is True
    assert wf._looks_like_block("normal content") is False
    assert wf._looks_like_block("") is False


# --------------------------------------------------------------------------
# Regression: the bypass cascade used GNU-only shell command strings
# (`wget -q ... --user-agent="..."`, `shell=True` with the URL interpolated).
# On Termux/Android the system wget (BusyBox/toybox) rejects the GNU-only
# flags, and because wget was the LAST method its misleading error masked the
# real reason all earlier methods failed. Verify the cascade now shells out
# flag-literal argument lists and wget degrades gracefully when -q/-U are
# unsupported.
# --------------------------------------------------------------------------

def _fake_run(script: list):
    """Return a _run_cmd replacement that simulates BusyBox wget rejecting -q."""

    def fake(cmd, timeout=25, env=None):
        if cmd[0] == "wget" and "-q" in cmd[0:5]:
            return (False, "", "wget: error: no such option: -q")
        if cmd[0] == "wget":
            return (True, "<html><body>" + "real via wget " * 20 + "</body></html>", "")
        if cmd[0] == "curl":
            return (True, "<html><body>" + "real via curl " * 20 + "</body></html>", "")
        return (False, "", "no such binary")

    return fake


def test_scripts_run_as_argument_lists_no_shell():
    from opencode_py.tools.cloudflare_bypass import UltimateBypass

    ub = UltimateBypass()
    with mock.patch.object(subprocess, "run") as run:
        run.return_value.returncode = 0
        run.return_value.stdout = "boom"
        run.return_value.stderr = ""
        ub._run_cmd(["curl", "-A", '"sed-ish"'])  # would fail in a shell
        # never invoked through a shell
        for c in run.call_args.args[:1]:
            assert c == ["curl", "-A", '"sed-ish"']
        assert run.call_args.kwargs.get("shell") is False


def test_wget_skips_unsupported_flag_then_succeeds():
    from opencode_py.tools.cloudflare_bypass import UltimateBypass

    ub = UltimateBypass()
    with mock.patch.object(ub, "_run_cmd", side_effect=_fake_run(None)):
        result = ub.try_wget("https://example.com/page")
        assert result["success"] is True
        assert result["method"] == "wget"
        assert "real via wget" in result["content"]


def test_headers_are_browser_like():
    from opencode_py.tools.cloudflare_bypass import UltimateBypass

    ub = UltimateBypass(user_agent="test-agent")
    ub.use_rotation = False  # keep UA deterministic
    h = ub._get_headers()
    assert h["User-Agent"] == "test-agent"
    for sec in ("sec-ch-ua", "sec-fetch-dest", "sec-fetch-mode", "sec-fetch-site"):
        assert sec.upper() in h or any(sec in k.lower() for k in h)
    assert h["Accept-Language"].startswith("en-US")


def test_webfetch_primary_uses_browser_ua():
    assert wf.BROWSER_UA.startswith("Mozilla/5.0")


def test_bypass_failure_now_reports_real_reason_not_wget_flags():
    srv, t = _start_server()
    try:
        with mock.patch.object(wf, "UltimateBypass") as UB:
            UB.return_value.fetch.return_value = {
                "success": False,
                "error": "Status: 403",
            }
            url = f"http://127.0.0.1:{srv.server_port}/blocked"
            r = wf._webfetch(url, "markdown", 10)
            assert r.get("error") is True
            assert "Status: 403" in r["output"]
            assert "no such option" not in r["output"]
    finally:
        srv.shutdown()


# --------------------------------------------------------------------------
# IP rotation: the cascade can retry a blocked fetch through a fresh proxy so
# per-IP rate limiting can't starve it. Pool is env-driven (OPENCODE_PROXY_POOL
# / OPENCODE_HARVEST_PROXIES) and needs no Tor.
# --------------------------------------------------------------------------

def test_proxy_pool_rotates_from_env():
    from opencode_py.tools.cloudflare_bypass import ProxyPool

    with mock.patch.dict(
        os.environ,
        {"OPENCODE_PROXY_POOL": "http://1.2.3.4:8080, http://5.6.7.8:3128"},
        clear=False,
    ):
        pool = ProxyPool(harvest=False)
        assert pool.available is True
        first = pool.next()
        assert first == "http://1.2.3.4:8080"
        assert pool.next() == "http://5.6.7.8:3128"
        assert pool.next() == "http://1.2.3.4:8080"  # wraps around


def test_proxy_pool_honors_custom_list():
    from opencode_py.tools.cloudflare_bypass import ProxyPool

    pool = ProxyPool(proxies=["http://a:1", "http://b:2"], harvest=False)
    assert [pool.next(), pool.next()] == ["http://a:1", "http://b:2"]


def test_fetch_retries_with_fresh_proxy_when_blocked():
    from opencode_py.tools.cloudflare_bypass import ProxyPool, UltimateBypass

    pool = ProxyPool(proxies=["http://p1:1", "http://p2:2"], harvest=False)
    ub = UltimateBypass(timeout=1, proxy_pool=pool, ip_retries=2)

    def fake_methods(url):
        ub.stats["methods_tried"] += 1
        if ub.proxy == "http://p2:2":
            return {"success": True, "method": "requests", "content": "bypassed via p2"}
        return {"success": False, "method": "requests", "error": "Status: 403"}

    with mock.patch.object(ub, "try_basic_request", side_effect=fake_methods):
        # stub the rest so only requests runs
        for m in ("cloudscraper", "curl_cffi", "curl", "httpx", "wget"):
            mock.patch.object(
                ub, f"try_{m}", return_value={"success": False, "error": "skip"}
            ).start()
        result = ub.fetch("https://example.com")
        assert result["success"] is True
        assert result["proxy"] == "http://p2:2"
        assert result["stats"]["success_method"] == "requests"


def test_rotation_wired_through_bypass_fetch():
    srv, t = _start_server()
    try:
        with mock.patch.dict(
            os.environ,
            {"OPENCODE_PROXY_POOL": "http://127.0.0.1:1"},  # invalid but non-empty
            clear=False,
        ):
            with mock.patch.object(wf, "UltimateBypass") as UB:
                UB.return_value.fetch.return_value = {
                    "success": True,
                    "method": "requests",
                    "content": "<h1>ok</h1>",
                }
                url = f"http://127.0.0.1:{srv.server_port}/blocked"
                r = wf._webfetch(url, "markdown", 10)
                assert r.get("error") is None
                # a pool was constructed and handed to the bypass
                pool = UB.call_args.kwargs["proxy_pool"]
                assert pool.next() == "http://127.0.0.1:1"
    finally:
        srv.shutdown()