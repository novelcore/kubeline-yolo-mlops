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
     POST never arrives, so we time out and offer the fallback.

  2. GUIDED PASTE (last-resort fallback, only if the user opts in): open the
     browser to the lakeFS UI and prompt for the cookie once. Works with zero
     platform change. This is NOT entered automatically — reaching it means the
     automated path genuinely failed, and silently demanding a paste hides that.

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

from dataclasses import dataclass

from . import device_flow
from .appconfig import config_value
from .lakefs_client import COOKIE_NAME, LakeFSClient


@dataclass(frozen=True)
class Credential:
    """What the CLI carries to lakeFS.

    ``token`` is a per-user Zitadel bearer from device flow — the normal path.
    ``cookie`` is the legacy oauth2-proxy session, kept only so a cached
    pre-upgrade session and the manual paste escape hatch keep working.
    """
    token: Optional[str] = None
    cookie: Optional[str] = None

    @property
    def kind(self) -> str:
        return "bearer" if self.token else "cookie"

SESSION_PATH = pathlib.Path(
    os.environ.get("KUBECORE_ML_HOME", pathlib.Path.home() / ".config" / "kubecore-ml")
) / "lakefs-session.json"

# Fixed loopback port so the return page's POST target is stable. The nginx
# return page reflects whatever ?port= we pass, so any free port works — but a
# stable default keeps the flow predictable. Overridable for clashes.
LOOPBACK_PORT = int(os.environ.get("KUBECORE_ML_LOOPBACK_PORT", "8765"))

# How long to wait for the browser sign-in to hand back a session. This has to
# cover a real human: Zitadel page load + password + 2FA. Overridable for scripts
# that want to fail fast.
LOOPBACK_TIMEOUT = int(os.environ.get("KUBECORE_ML_LOGIN_TIMEOUT", "180"))

# Path (on the lakeFS domain) of the same-origin return page served by the nginx
# auth-injector sidecar. Kept in sync with the operator composition
# (projectmlstack/gcp/composition.yaml, nginx.conf.tmpl `location /kubecore-cli/*`).
RETURN_PATH = "/kubecore-cli/return"


# ----------------------------------------------------------------------
# session cache
# ----------------------------------------------------------------------
def save_session(base_url: str, cookie: Optional[str] = None,
                 token: Optional[str] = None) -> None:
    SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"base_url": base_url.rstrip("/")}
    if token:
        payload["token"] = token
    if cookie:
        payload["cookie"] = cookie
    SESSION_PATH.write_text(json.dumps(payload))
    os.chmod(SESSION_PATH, 0o600)


def load_session(base_url: Optional[str] = None) -> Optional[Credential]:
    """Return a cached, still-valid credential for base_url, else None.

    Reads both the current {"token": ...} shape and the legacy {"cookie": ...}
    one, so an existing session file keeps working after an upgrade.
    """
    if not SESSION_PATH.exists():
        return None
    try:
        data = json.loads(SESSION_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if base_url and data.get("base_url") != base_url.rstrip("/"):
        return None

    token, cookie = data.get("token"), data.get("cookie")
    if not token and not cookie:
        return None
    cred = Credential(token=token, cookie=cookie)
    # verify it still authenticates (tokens expire; cookies are 4h)
    client = LakeFSClient(data["base_url"], cookie=cookie, token=token)
    return cred if client.check_auth() else None


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


def loopback_login(base_url: str, timeout: int = LOOPBACK_TIMEOUT) -> Optional[str]:
    """Open the browser, capture the _lakefs_oauth2 cookie via the return page.

    Opens ``<lakefs>/oauth2/start?rd=<lakefs>/kubecore-cli/return?port=<port>`` and
    runs a localhost catcher that receives the session JSON POSTed by that page.
    Returns the cookie value, or None if capture failed (caller falls back).

    The timeout budgets a HUMAN sign-in: the clock starts when the browser opens,
    and the user still has to read the Zitadel page, type a password and clear 2FA.
    A short deadline here doesn't "fail fast", it just abandons a capture that was
    about to succeed and drops a paste prompt into the terminal mid-login.
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

    # Only animate on a real terminal; piped/CI output gets no escape codes.
    tty = sys.stdout.isatty()

    def _clear():
        if tty:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()

    try:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if catcher.captured.get("cookie"):
                _clear()  # drop the countdown before the caller prints
                return catcher.captured["cookie"]
            # A silent multi-minute wait is indistinguishable from a hang — show
            # the remaining budget so the user knows we're still listening.
            if tty:
                remaining = int(deadline - time.time())
                sys.stdout.write(
                    f"\r\033[K⏳  Waiting for you to finish signing in… {remaining}s"
                )
                sys.stdout.flush()
            time.sleep(0.5)
        _clear()
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
    print("(DevTools → Application/Storage → Cookies) and paste it below,")
    print("then press Enter.\n")
    # Read via sys.stdin, NOT input(): on macOS, Python's input() uses libedit,
    # which breaks on pastes longer than the terminal width — the line wraps and
    # Enter stops submitting (a session cookie/JWT is hundreds of chars). Reading
    # the raw line has no line-editor, so long pastes + Enter work everywhere.
    sys.stdout.write("_lakefs_oauth2 = ")
    sys.stdout.flush()
    try:
        line = sys.stdin.readline()
    except (EOFError, KeyboardInterrupt):
        return None
    if not line:  # EOF (Ctrl-D)
        return None
    cookie = line.strip()
    return cookie or None


# ----------------------------------------------------------------------
# entry
# ----------------------------------------------------------------------
def oidc_settings(start: Optional[str] = None) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve (issuer, client_id, audience_project_id) for device-flow login.

    Precedence: environment > .kubecore/dataset-config.yaml. The operator
    publishes these per project (KubeProject.status.mlStack.datasetCLIOIDC,
    kubecore-operator #1347) and the render step writes them into the app repo,
    so nothing is hardcoded and every project gets its own app.
    """
    issuer = os.environ.get("KUBECORE_ML_OIDC_ISSUER") or config_value("oidcIssuer", start)
    client_id = os.environ.get("KUBECORE_ML_OIDC_CLIENT_ID") or config_value("oidcClientId", start)
    audience = (os.environ.get("KUBECORE_ML_OIDC_PROJECT_ID")
                or config_value("oidcProjectId", start))
    return issuer, client_id, audience


def device_flow_login(base_url: str) -> Optional[Credential]:
    """Log in via Zitadel device flow. Returns None if it is not configured."""
    issuer, client_id, audience = oidc_settings()
    if not issuer or not client_id:
        return None
    try:
        tokens = device_flow.login(issuer, client_id, audience_project_id=audience)
    except device_flow.DeviceFlowError as exc:
        print(f"\nBrowser sign-in failed: {exc}")
        return None
    token = tokens.get("access_token")
    return Credential(token=token) if token else None


def login(base_url: str, force: bool = False, prefer_paste: bool = False) -> Credential:
    """Return a valid credential for base_url; log in via browser if needed.

    Order: session cache -> Zitadel device flow -> guided paste (opt-in).

    Device flow is the normal path. Nothing passes from the browser to the CLI
    — both talk to Zitadel independently — which is what every earlier design
    got wrong and why users ended up pasting a cookie by hand. The paste
    fallback survives only for projects whose OIDC app is not provisioned yet.
    """
    base_url = base_url.rstrip("/")

    if not force:
        cached = load_session(base_url)
        if cached:
            print(f"✓ Using cached lakeFS session ({cached.kind}).")
            return cached

    cred: Optional[Credential] = None
    if not prefer_paste:
        cred = device_flow_login(base_url)
        if cred is None:
            issuer, client_id, _ = oidc_settings()
            if not issuer or not client_id:
                print("\nDevice-flow sign-in is not configured for this app "
                      "(no oidcIssuer/oidcClientId in .kubecore/dataset-config.yaml).")
            print("Falling back to the manual cookie paste.")
            sys.stdout.write("Continue with a manual paste? [y/N] ")
            sys.stdout.flush()
            try:
                answer = sys.stdin.readline().strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = ""
            if answer not in ("y", "yes"):
                sys.exit("Aborted — not signed in.")

    if cred is None:
        cookie = guided_paste_login(base_url)
        if not cookie:
            sys.exit("ERROR: no lakeFS credential was captured. Aborting.")
        cred = Credential(cookie=cookie)

    # verify before caching
    client = LakeFSClient(base_url, cookie=cred.cookie, token=cred.token)
    if not client.check_auth():
        sys.exit(
            f"ERROR: the captured {cred.kind} did not authenticate to lakeFS "
            "(invalid or expired). Try again."
        )
    save_session(base_url, cookie=cred.cookie, token=cred.token)
    print("✓ Logged in — session cached.")
    return cred
