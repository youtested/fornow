"""Vendored Cloudflare bypass (HTTP-only).

Adapted from https://github.com/youtested/Cloudflare-bypass (MIT).
Changes for this project:
- HTTP-only: keeps the requests / httpx / curl_cffi(if present) / curl / wget
  methods. The Chrome-based methods (selenium, scrapling, browser, browser-wait)
  are intentionally omitted so this stays pure-Python and armv7-safe for Termux.
- No CLI/progress bars / global logging config: this is consumed as a library
  by the webfetch tool.
"""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, Optional

# Try import requests (hard dep for our use)
try:
    import requests  # type: ignore
    REQUESTS_AVAILABLE = True
except ImportError:  # pragma: no cover
    requests = None  # type: ignore
    REQUESTS_AVAILABLE = False

# Optional modules (armv7-safe only). curl_cffi ships binary wheels and is
# optional; httpx is a hard dep of this project already.
try:
    import httpx  # type: ignore
    HTTPX_AVAILABLE = True
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore
    HTTPX_AVAILABLE = False

try:
    from curl_cffi import requests as curl_requests  # type: ignore
    CURL_CFFI_AVAILABLE = True
except ImportError:
    curl_requests = None  # type: ignore
    CURL_CFFI_AVAILABLE = False

# Pure-Python Cloudflare challenge solver (solves the older JS "cf_chl"
# challenge without a browser). Installs on 32-bit Termux.
try:
    import cloudscraper  # type: ignore
    CLOUDSCRAPER_AVAILABLE = True
except ImportError:
    cloudscraper = None  # type: ignore
    CLOUDSCRAPER_AVAILABLE = False

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/131.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/131.0.0.0",
]


class CookieManager:
    """Save/load cookies per domain (used by the requests method)."""

    def __init__(self, cookie_dir: Optional[str] = None):
        self.cookie_dir = Path(cookie_dir or (Path.home() / ".cache" / "opencode" / "cookies"))
        self.cookie_dir.mkdir(parents=True, exist_ok=True)

    def _get_cookie_path(self, domain: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", domain)
        return self.cookie_dir / f"{safe}.json"

    def load(self, domain: str) -> Optional[Dict]:
        path = self._get_cookie_path(domain)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def save(self, domain: str, cookies: Dict) -> None:
        path = self._get_cookie_path(domain)
        try:
            path.write_text(json.dumps(cookies))
        except OSError:  # pragma: no cover
            pass


def _check_block(content: str, status_code: Optional[int] = None) -> bool:
    """Return True if the response looks like a Cloudflare/JS challenge page."""
    if content is None:
        content = ""
    lower = content.lower()[:1000]
    blockers = [
        "cloudflare",
        "checking your browser",
        "attention required",
        "error 1020",
        "ray id",
        "just a moment",
        "turnstile",
        "captcha",
        "access denied",
        "blocked",
    ]
    if status_code == 403:
        return True
    return any(b in lower for b in blockers)


class ProxyPool:
    """Rotating list of proxies used to fetch through a fresh IP per attempt.

    Sources (no Tor):
      1. OPENCODE_PROXY_POOL env var — comma/space separated proxy URLs
         (e.g. "http://user:pass@host:port, http://host:port").
      2. Open harvest (opt-in): when OPENCODE_HARVEST_PROXIES=1, free public
         HTTP/SOCKS proxies are pulled from public list APIs and filtered to
         reachable, HTTPS-capable ones, so every blocked fetch retries from a
         different IP.
    Rotation picks the next proxy each round. NOTE: free proxies are
    third-party relays — fine for reading public pages, not for private data.
    """

    def __init__(self, proxies: Optional[list] = None, harvest: Optional[bool] = None):
        self._proxies = list(proxies or [])
        self._index = 0
        self._lock = threading.Lock()
        self._load_env()
        if harvest is None:
            harvest = os.environ.get("OPENCODE_HARVEST_PROXIES", "") not in ("", "0", "false")
        if not self._proxies and harvest:
            try:
                self._proxies = self._harvest()
            except Exception:
                self._proxies = []

    def _load_env(self) -> None:
        raw = os.environ.get("OPENCODE_PROXY_POOL", "")
        for item in re.split(r"[,\s]+", raw.strip()):
            if item:
                self._proxies.append(item)

    @property
    def available(self) -> bool:
        return bool(self._proxies)

    def next(self) -> Optional[str]:
        if not self._proxies:
            return None
        # Locked so concurrent fetches (parallel webfetch_many) rotate through
        # the same pool without two workers landing on the same proxy.
        with self._lock:
            proxy = self._proxies[self._index % len(self._proxies)]
            self._index += 1
        return proxy

    def new_identity(self) -> None:
        """With a pool, the next `next()` call is already a different proxy/IP,
        so there is nothing else to do here."""

    @classmethod
    def shared(cls) -> "ProxyPool":
        """Return the process-wide pool (rebuilt automatically when the
        OPENCODE_PROXY_POOL / OPENCODE_HARVEST_PROXIES env changes)."""
        return _shared_pool_get()

    def _harvest(self) -> list:
        """Pull free proxies from public list APIs and keep the live ones."""
        import urllib.request

        candidates: list[str] = []
        endpoints = [
            "https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http&timeout=5000&country=all&ssl=all&anonymity=all",
            "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt",
            "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/http.txt",
            "https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt",
        ]
        for ep in endpoints:
            try:
                req = urllib.request.Request(ep, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=8) as r:
                    for line in (r.read().decode("utf-8", "replace") or "").splitlines():
                        line = line.strip()
                        if re.match(r"^[\w.\-]+:\d{2,5}$", line):
                            candidates.append("http://" + line)
            except Exception:
                continue
        # dedupe + quick reachability filter (keep a handful that answer)
        seen: list[str] = []
        import socket

        def alive(proxy: str) -> bool:
            try:
                host, _, port = proxy.replace("http://", "").partition(":")
                with socket.create_connection((host, int(port)), timeout=2):
                    return True
            except Exception:
                return False

        for p in dict.fromkeys(candidates):
            if len(seen) >= 20:
                break
            if alive(p):
                seen.append(p)
        return seen


# A process-wide shared pool so parallel fetches (webfetch_many) all rotate
# through the same proxy list (and only harvest from the list APIs once) instead
# of each fetch building — and repeatedly re-harvesting — its own pool.
_shared_pool: Optional["ProxyPool"] = None
_shared_pool_sig: tuple = ("", [])
_shared_pool_lock = threading.Lock()


def _pool_signature() -> tuple:
    """Return (harvest_enabled, env_proxy_list) — rebuild the shared pool when
    this changes (env is read at build time, and tests patch it)."""
    harvest = os.environ.get("OPENCODE_HARVEST_PROXIES", "") not in ("", "0", "false")
    raw = os.environ.get("OPENCODE_PROXY_POOL", "")
    proxies = [i for i in re.split(r"[,\s]+", raw.strip()) if i]
    return (harvest, proxies)


def _shared_pool_get() -> "ProxyPool":
    global _shared_pool, _shared_pool_sig
    sig = _pool_signature()
    with _shared_pool_lock:
        if _shared_pool is None or _shared_pool_sig != sig:
            _shared_pool = ProxyPool(harvest=sig[0])
            _shared_pool_sig = sig
        return _shared_pool


def _reset_shared_pool() -> None:
    """Test hook: drop the cached shared pool so the next call rebuilds it."""
    global _shared_pool, _shared_pool_sig
    with _shared_pool_lock:
        _shared_pool = None
        _shared_pool_sig = ("", [])


class UltimateBypass:
    """Fetch a URL through a cascade of HTTP methods to slip past Cloudflare."""

    def __init__(
        self,
        proxy: Optional[str] = None,
        user_agent: Optional[str] = None,
        use_rotation: bool = True,
        timeout: int = 20,
        proxy_pool: Optional[ProxyPool] = None,
        ip_retries: int = 2,
    ):
        self.proxy = proxy
        self.user_agent = user_agent or random.choice(USER_AGENTS)
        self.use_rotation = use_rotation
        self.timeout = timeout
        self.cookie_manager = CookieManager()
        self.stats: Dict = {"methods_tried": 0, "success_method": None}
        self.proxy_pool = proxy_pool or ProxyPool()
        self.ip_retries = ip_retries
        self._last_proxy: Optional[str] = None

    def _get_headers(self) -> Dict:
        ua = random.choice(USER_AGENTS) if self.use_rotation else self.user_agent
        # Only advertise brotli if the installed HTTP stack can decode it;
        # otherwise a br response comes back as undecoded bytes on Termux.
        accept_encoding = "gzip, deflate"
        try:
            import brotli  # noqa: F401

            accept_encoding += ", br"
        except ImportError:
            pass
        # Modern browser header set: the missing sec-ch-ua / sec-fetch-* /
        # accept-language trio is exactly what bot detectors (DataDome, Akamai,
        # Cloudflare) score against, and it applies to the Python-based fetchers
        # where we control headers directly.
        return {
            "User-Agent": ua,
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "image/avif,image/webp,image/apng,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": accept_encoding,
            "Sec-CH-UA": (
                '"Chromium";v="131", "Not_A Brand";v="24", '
                '"Google Chrome";v="131"'
            ),
            "Sec-CH-UA-Mobile": "?0",
            "Sec-CH-UA-Platform": '"Windows"',
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1",
        }

    def _proxy_dict(self) -> Optional[Dict]:
        if not self.proxy:
            return None
        return {"http": self.proxy, "https": self.proxy}

    def _run_cmd(self, cmd: list, timeout: int = 25, env: Optional[Dict] = None) -> tuple:
        """Run an external fetch tool as an argument list (no shell), so flags
        and URLs are never interpreted by a shell and BusyBox/toybox quirks
        can't mangle quoting."""
        try:
            result = subprocess.run(
                cmd,
                shell=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            return (result.returncode == 0, result.stdout or "", result.stderr or "")
        except subprocess.TimeoutExpired:
            return (False, "", "Timeout")
        except Exception as e:  # pragma: no cover
            return (False, "", str(e))

    def try_basic_request(self, url: str) -> Dict:
        if not REQUESTS_AVAILABLE:
            return {"success": False, "method": "requests", "error": "requests not installed"}
        from urllib.parse import urlparse

        try:
            domain = urlparse(url).netloc
            r = requests.get(
                url,
                headers=self._get_headers(),
                proxies=self._proxy_dict(),
                cookies=self.cookie_manager.load(domain),
                timeout=self.timeout,
            )
            if r.status_code == 200 and len(r.text) > 100 and not _check_block(r.text, r.status_code):
                if r.cookies:
                    self.cookie_manager.save(domain, dict(r.cookies))
                return {"success": True, "method": "requests", "content": r.text}
            return {"success": False, "method": "requests", "error": f"Status: {r.status_code}"}
        except Exception as e:
            return {"success": False, "method": "requests", "error": str(e)}

    def try_httpx(self, url: str) -> Dict:
        if not HTTPX_AVAILABLE:
            return {"success": False, "method": "httpx", "error": "httpx not installed"}
        try:
            kwargs: Dict = {"timeout": self.timeout, "follow_redirects": True}
            if self.proxy:
                # httpx >= 0.28 dropped `proxies=` in favour of the single
                # `proxy=` (older versions accept a plain proxy string too).
                kwargs["proxy"] = self.proxy
            with httpx.Client(**kwargs) as client:
                r = client.get(url, headers=self._get_headers())
            if r.status_code == 200 and len(r.text) > 100 and not _check_block(r.text, r.status_code):
                return {"success": True, "method": "httpx", "content": r.text}
            return {"success": False, "method": "httpx", "error": f"Status: {r.status_code}"}
        except Exception as e:
            return {"success": False, "method": "httpx", "error": str(e)}

    def try_curl_cffi(self, url: str) -> Dict:
        if not CURL_CFFI_AVAILABLE:
            return {"success": False, "method": "curl-cffi", "error": "curl_cffi not installed"}
        try:
            session = curl_requests.Session(proxies=self._proxy_dict())
            r = session.get(url, impersonate="chrome", timeout=self.timeout)
            if r.status_code == 200 and len(r.text) > 100 and not _check_block(r.text, r.status_code):
                return {"success": True, "method": "curl-cffi", "content": r.text}
            return {"success": False, "method": "curl-cffi", "error": f"Status: {r.status_code}"}
        except Exception as e:
            return {"success": False, "method": "curl-cffi", "error": str(e)}

    def try_cloudscraper(self, url: str) -> Dict:
        if not CLOUDSCRAPER_AVAILABLE:
            return {"success": False, "method": "cloudscraper", "error": "cloudscraper not installed"}
        try:
            scraper = cloudscraper.create_scraper(
                browser={"browser": "chrome", "platform": "windows", "desktop": True}
            )
            r = scraper.get(url, timeout=self.timeout)
            if r.status_code == 200 and len(r.text) > 100 and not _check_block(r.text, r.status_code):
                return {"success": True, "method": "cloudscraper", "content": r.text}
            return {"success": False, "method": "cloudscraper", "error": f"Status: {r.status_code}"}
        except Exception as e:
            return {"success": False, "method": "cloudscraper", "error": str(e)}

    def try_curl(self, url: str) -> Dict:
        ua = random.choice(USER_AGENTS) if self.use_rotation else self.user_agent
        cmd = ["curl", "-s", "-L", "-A", ua, "--compressed"]
        if self.proxy:
            cmd += ["--proxy", self.proxy]
        # A browser-like header set is what makes curl pass bot walls that a
        # bare UA-only request trips; the host header ordering mirrors Chrome.
        for h in (
            "accept: text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,image/apng,*/*;q=0.8",
            "accept-language: en-US,en;q=0.9",
            "sec-ch-ua: \"Chromium\";v=\"131\", \"Not_A Brand\";v=\"24\", \"Google Chrome\";v=\"131\"",
            "sec-ch-ua-mobile: ?0",
            "sec-ch-ua-platform: \"Windows\"",
            "sec-fetch-dest: document",
            "sec-fetch-mode: navigate",
            "sec-fetch-site: none",
            "sec-fetch-user: ?1",
            "upgrade-insecure-requests: 1",
        ):
            cmd += ["-H", h]
        cmd.append(url)
        success, stdout, stderr = self._run_cmd(cmd, self.timeout)
        if success and stdout and len(stdout) > 100 and not _check_block(stdout):
            return {"success": True, "method": "curl", "content": stdout}
        return {"success": False, "method": "curl", "error": stderr[:100] if stderr else "Empty"}

    def try_wget(self, url: str) -> Dict:
        ua = random.choice(USER_AGENTS) if self.use_rotation else self.user_agent
        # BusyBox/toybox wget (Termux/Android) only understand a subset of GNU
        # wget's flags, so walk a ladder of increasingly minimal invocations and
        # fall through when a flag is rejected. The final variant has no flags
        # beyond the mandatory ones, which every wget supports.
        base = [
            ["wget", "-q", "-O", "-", "--user-agent", ua],
            ["wget", "-q", "-O", "-", "-U", ua],
            ["wget", "-O", "-", "-U", ua],
            ["wget", "-O", "-"],
        ]
        last_err = ""
        for head in base:
            cmd = list(head)
            if self.proxy:
                cmd += ["-e", f"http_proxy={self.proxy}", "-e", f"https_proxy={self.proxy}"]
            cmd.append(url)
            success, stdout, stderr = self._run_cmd(cmd, self.timeout)
            if success and stdout and len(stdout) > 100 and not _check_block(stdout):
                return {"success": True, "method": "wget", "content": stdout}
            last_err = stderr[:100] if stderr else ""
            # Only flag-rejection warrants trying the next variant; a real
            # network failure would recur identically and waste time.
            if "no such option" in stderr or "unrecognized option" in stderr or "invalid option" in stderr:
                continue
            break
        return {"success": False, "method": "wget", "error": last_err or "Empty"}

    def fetch(self, url: str, force_method: Optional[str] = None) -> Dict:
        """Try each method in order until one returns clean content.

        When every method is blocked, rotates to a fresh proxy (a different
        exit IP) and repeats the cascade up to ``ip_retries`` times — this is
        what defeats IP-rate-limited bot walls: the request leaves from a new
        address each round instead of being throttled on one IP.
        """
        methods = [
            "requests",
            "cloudscraper",
            "curl-cffi",
            "curl",
            "httpx",
            "wget",
        ]
        if force_method and force_method in methods:
            methods = [force_method] + [m for m in methods if m != force_method]

        last_result: Dict = {"success": False, "error": "Unknown"}
        for attempt in range(max(1, self.ip_retries + 1)):
            # pick a fresh exit IP each round (rotate within the pool / Tor)
            if self.proxy_pool.available:
                self._last_proxy = self.proxy_pool.next()
                if attempt > 0:
                    self.proxy_pool.new_identity()
            self.proxy = self._last_proxy

            result: Dict = {"success": False, "error": "Unknown"}
            for method in methods:
                self.stats["methods_tried"] += 1
                if method == "requests":
                    result = self.try_basic_request(url)
                elif method == "cloudscraper":
                    result = self.try_cloudscraper(url)
                elif method == "curl-cffi":
                    result = self.try_curl_cffi(url)
                elif method == "curl":
                    result = self.try_curl(url)
                elif method == "httpx":
                    result = self.try_httpx(url)
                elif method == "wget":
                    result = self.try_wget(url)

                if result.get("success"):
                    self.stats["success_method"] = method
                    result["method"] = method
                    result["proxy"] = self._last_proxy
                    result["stats"] = self.stats
                    return result

            last_result = result
            # exhausted the pool? no point retrying on the same IPs
            if not self.proxy_pool.available:
                break

        last_result["stats"] = self.stats
        last_result["rotations"] = attempt + 1 if self.proxy_pool.available else 1
        return last_result


def fetch_url(url: str, timeout: int = 20) -> Dict:
    """Convenience wrapper returning the raw result dict."""
    return UltimateBypass(timeout=timeout).fetch(url)
