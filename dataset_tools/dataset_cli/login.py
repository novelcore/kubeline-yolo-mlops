"""Browser OIDC login for lakeFS.

lakeFS is fronted by oauth2-proxy(Zitadel) + an nginx sidecar that injects the
shared lakeFS admin Basic-auth header server-side (lakeFS OSS runs in basic_auth
mode: RBAC:none, single admin, no per-user identity, no mintable credentials).
So the credential the CLI carries through the ingress is the oauth2-proxy session
cookie ``_lakefs_oauth2`` — oauth2-proxy validates it and forwards the request to
nginx, which injects admin Basic before lakeFS. That cookie is Secure + HttpOnly
and scoped to the lakeFS domain, so it can NEVER be captured on http://localhost
directly (the old loopback design that pointed ``rd=`` at localhost hung forever:
the cookie is never sent cross-origin to the loopback).

This module captures the cookie via a **same-origin return page** served on the
lakeFS domain itself:

  1. TOKEN RETURN PAGE (preferred): start a localhost catcher, open the browser to
     ``<lakefs>/oauth2/start?rd=<lakefs>/kubecore-cli/return?port=<port>``. After
     the user logs in via Zitadel, oauth2-proxy redirects to the return page —
     which is served by the nginx sidecar ON the lakeFS domain, so it runs with a
     valid session. The page fetches ``/kubecore-cli/session`` (same-origin; the
     HttpOnly cookie is sent automatically and echoed back server-side by nginx),
     then POSTs the session JSON to ``http://localhost:<port>/callback``. The
     catcher reads the cookie value off that POST. Fully automated — no paste.
     Requires the ``/kubecore-cli/*`` locations on the nginx sidecar (operator
     composition); if they aren't deployed yet, the page's fetch 302s and the
     POST never arrives, so we time out and fall back.

  2. GUIDED PASTE (last-resort fallback): open the browser to the lakeFS UI and
     prompt the user to paste the cookie once. Works with zero platform change.

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

# Fixed loopback port so the return page's POST target is stable. The nginx
# return page reflects whatever ?port= we pass, so any free port works — but a
# stable default keeps the flow predictable. Overridable for clashes.
LOOPBACK_PORT = int(os.environ.get("KUBECORE_ML_LOOPBACK_PORT", "8765"))

# Path (on the lakeFS domain) of the same-origin return page served by the nginx
# auth-injector sidecar. Kept in sync with the operator composition
# (projectmlstack/gcp/composition.yaml, nginx.conf.tmpl `location /kubecore-cli/*`).
RETURN_PATH = "/kubecore-cli/return"


# ----------------------------------------------------------------------
# session cache
# ----------------------------------------------------------------------
def save_session(base_url: str, cookie: str) -> None:
    SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    SESSION_PATH.write_text(json.dumps({"base_url": base_url.rstrip("/"), "cookie": cookie}))
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
# loopback capture (token return page)
# ----------------------------------------------------------------------
def _make_catcher():
    """Build a one-shot handler class that captures the session cookie.

    The same-origin return page (served by nginx on the lakeFS domain) POSTs the
    session JSON to ``/callback``. Because that POST is cross-origin
    (https://lakefs-… → http://localhost:<port>), the browser sends a CORS
    preflight OPTIONS first for the application/json content-type; we answer both
    OPTIONS and POST with permissive CORS so the fetch succeeds.
    """

    class _CookieCatcher(http.server.BaseHTTPRequestHandler):
        captured: dict = {}

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def do_OPTIONS(self):  # noqa: N802 (CORS preflight)
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                payload = {}
            cookie = payload.get("cookie") or ""
            if cookie:
                _CookieCatcher.captured["cookie"] = cookie
            status = 200 if cookie else 400
            self.send_response(status)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}' if cookie else b'{"ok":false}')

        def do_GET(self):  # noqa: N802
            # A human hitting the callback in a browser — friendly note only.
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><body style='font-family:sans-serif;padding:3rem'>"
                b"<h2>kubecore-dataset</h2><p>Waiting for the login return page to "
                b"hand over your session\xe2\x80\xa6 keep this tab open.</p></body></html>"
            )

        def log_message(self, *args):  # silence default stderr logging
            pass

    _CookieCatcher.captured = {}
    return _CookieCatcher


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def loopback_login(base_url: str, timeout: int = 10) -> Optional[str]:
    """Open the browser, capture the _lakefs_oauth2 cookie via the return page.

    Opens ``<lakefs>/oauth2/start?rd=<lakefs>/kubecore-cli/return?port=<port>`` and
    runs a localhost catcher that receives the session JSON POSTed by that page.
    Returns the cookie value, or None if capture failed (caller falls back).
    """
    base_url = base_url.rstrip("/")
    port = LOOPBACK_PORT
    if not _port_free(port):
        # pick an ephemeral one; the return page reflects ?port= so any port works.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

    # The return page is served ON the lakeFS domain (same-origin, so it can read
    # the session), and it POSTs the cookie back to our localhost catcher.
    # rd MUST be RELATIVE (path-only): oauth2-proxy refuses ABSOLUTE cross-host rd=
    # unless the host is in --whitelist-domain, so an absolute https://<lakefs>/... rd
    # silently falls through to the lakeFS UI after login. A relative same-host path is
    # always honored (no whitelist needed) and still resolves against the lakeFS origin,
    # so the /kubecore-cli/return page runs same-origin and can read the session cookie.
    return_url = f"{RETURN_PATH}?port={port}"
    start_url = (
        f"{base_url}/oauth2/start?rd={urllib.parse.quote(return_url, safe='')}"
    )

    catcher = _make_catcher()
    server = http.server.HTTPServer(("127.0.0.1", port), catcher)
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
            if catcher.captured.get("cookie"):
                return catcher.captured["cookie"]
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

    Tries the session cache, then the token return page, then guided-paste.
    Persists the cookie on success. Exits the process if all paths fail.
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
            print("Browser sign-in didn't complete "
                  "(the login return page may not be deployed yet) — "
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
