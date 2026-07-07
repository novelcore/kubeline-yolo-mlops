"""Browser OIDC login for lakeFS.

lakeFS is fronted by oauth2-proxy(Zitadel). This module gets the developer a
valid ``_lakefs_oauth2`` session WITHOUT copy-pasting cookies out of DevTools:

  1. LOOPBACK (preferred): start a localhost callback, open the browser to
     ``<lakefs>/oauth2/start?rd=http://localhost:<port>/callback``. The user logs
     in via Zitadel; oauth2-proxy sets the cookie and redirects back to the
     loopback, where we read the cookie off the request. Fully automated — no
     paste. Requires the loopback redirect to be allowed on the lakeFS OIDC app
     (the operator registers it; for testing it can be added in Zitadel by hand).

  2. GUIDED PASTE (fallback): if the loopback redirect is not allowed yet, open
     the browser to the lakeFS UI and prompt the user to paste the cookie once.
     Works today with zero platform change.

The captured cookie is cached (0600) at ~/.config/kubecore-ml/lakefs-session.json
so ``validate``/``sync`` reuse it until it expires.
"""
from __future__ import annotations

import http.server
import json
import os
import pathlib
import socket
import sys
import threading
import time
import urllib.parse
import webbrowser
from typing import Optional

from .lakefs_client import COOKIE_NAME, LakeFSClient

SESSION_PATH = pathlib.Path(
    os.environ.get("KUBECORE_ML_HOME", pathlib.Path.home() / ".config" / "kubecore-ml")
) / "lakefs-session.json"

# Fixed loopback port so the redirect URI is stable and can be pre-registered on
# the OIDC app. Overridable for local testing / port clashes.
LOOPBACK_PORT = int(os.environ.get("KUBECORE_ML_LOOPBACK_PORT", "8765"))


# ----------------------------------------------------------------------
# session cache
# ----------------------------------------------------------------------
def save_session(base_url: str, cookie: str) -> None:
    SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    SESSION_PATH.write_text(json.dumps({"base_url": base_url, "cookie": cookie}))
    os.chmod(SESSION_PATH, 0o600)


def load_session(base_url: Optional[str] = None) -> Optional[str]:
    """Return a cached, still-valid cookie for base_url, else None."""
    if not SESSION_PATH.exists():
        return None
    try:
        data = json.loads(SESSION_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if base_url and data.get("base_url") != base_url.rstrip("/"):
        return None
    cookie = data.get("cookie")
    if not cookie:
        return None
    # verify it still authenticates
    client = LakeFSClient(data["base_url"], cookie)
    return cookie if client.check_auth() else None


# ----------------------------------------------------------------------
# loopback capture
# ----------------------------------------------------------------------
class _CookieCatcher(http.server.BaseHTTPRequestHandler):
    captured: dict = {}

    def do_GET(self):  # noqa: N802
        # oauth2-proxy sets _lakefs_oauth2 then 302s to our rd=; the browser
        # sends the cookie back to this same host on the follow-up request.
        cookie_header = self.headers.get("Cookie", "")
        for part in cookie_header.split(";"):
            k, _, v = part.strip().partition("=")
            if k == COOKIE_NAME and v:
                _CookieCatcher.captured["cookie"] = v
        body = (
            b"<html><body style='font-family:sans-serif;padding:3rem'>"
            b"<h2>Logged in.</h2><p>You can close this tab and return to the "
            b"terminal.</p></body></html>"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence the default stderr logging
        pass


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def loopback_login(base_url: str, timeout: int = 180) -> Optional[str]:
    """Open the browser, capture the _lakefs_oauth2 cookie via localhost.

    Returns the cookie value, or None if capture failed (caller falls back).
    """
    base_url = base_url.rstrip("/")
    port = LOOPBACK_PORT
    if not _port_free(port):
        # pick an ephemeral one; note it may not be pre-registered on the app
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

    redirect = f"http://localhost:{port}/callback"
    # oauth2-proxy's start endpoint accepts rd= to return after login.
    start_url = f"{base_url}/oauth2/start?rd={urllib.parse.quote(redirect, safe='')}"

    _CookieCatcher.captured = {}
    server = http.server.HTTPServer(("127.0.0.1", port), _CookieCatcher)
    # serve_forever() in a background thread; stop it cleanly with shutdown().
    t = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.5},
                         daemon=True)
    t.start()

    print("\n🔑  Opening your browser to log in…")
    print(f"    {start_url}")
    print("    (click the link above if the browser didn't open)\n")
    webbrowser.open(start_url)

    try:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if _CookieCatcher.captured.get("cookie"):
                return _CookieCatcher.captured["cookie"]
            time.sleep(0.5)
        return None
    finally:
        server.shutdown()
        server.server_close()


# ----------------------------------------------------------------------
# guided paste fallback
# ----------------------------------------------------------------------
def guided_paste_login(base_url: str) -> Optional[str]:
    base_url = base_url.rstrip("/")
    print("\n🔑  Log in here (opens in your browser):")
    print(f"    {base_url}/\n")
    webbrowser.open(f"{base_url}/")
    print("After you're logged in, copy the `_lakefs_oauth2` cookie value")
    print("(DevTools → Application/Storage → Cookies) and paste it below.\n")
    try:
        cookie = input("_lakefs_oauth2 = ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    return cookie or None


# ----------------------------------------------------------------------
# entry
# ----------------------------------------------------------------------
def login(base_url: str, force: bool = False, prefer_paste: bool = False) -> str:
    """Return a valid cookie for base_url; log in via browser if needed.

    Tries the session cache, then loopback, then guided-paste. Persists the
    cookie on success. Exits the process with a message if all paths fail.
    """
    base_url = base_url.rstrip("/")

    if not force:
        cached = load_session(base_url)
        if cached:
            print("✓ Using cached lakeFS session.")
            return cached

    cookie = None
    if not prefer_paste:
        cookie = loopback_login(base_url)
        if not cookie:
            print("Loopback capture didn't complete "
                  "(the localhost redirect may not be registered yet) — "
                  "falling back to guided paste.")
    if not cookie:
        cookie = guided_paste_login(base_url)

    if not cookie:
        sys.exit("ERROR: no lakeFS session was captured. Aborting.")

    # verify before caching
    client = LakeFSClient(base_url, cookie)
    if not client.check_auth():
        sys.exit(
            "ERROR: the captured session did not authenticate to lakeFS "
            "(cookie invalid or expired). Try again."
        )
    save_session(base_url, cookie)
    print("✓ Logged in — session cached.")
    return cookie
