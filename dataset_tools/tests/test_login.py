"""Loopback login test — the local callback server must capture the
_lakefs_oauth2 cookie the browser sends back, and the redirect URL must carry
the port the server actually bound to."""
from __future__ import annotations

import http.client
import threading
import time
import urllib.parse as up

from dataset_cli import login


def test_loopback_captures_cookie(monkeypatch):
    opened: dict = {}
    monkeypatch.setattr(login.webbrowser, "open", lambda url: opened.setdefault("url", url))

    result: dict = {}

    def run():
        result["cookie"] = login.loopback_login("http://example.invalid", timeout=8)

    th = threading.Thread(target=run)
    th.start()
    time.sleep(1.0)

    # the browser was pointed at oauth2/start?rd=http://localhost:<port>/callback
    rd = up.parse_qs(up.urlparse(opened["url"]).query)["rd"][0]
    port = int(up.urlparse(rd).port)

    # simulate oauth2-proxy's post-login redirect landing on our callback,
    # carrying the session cookie
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", "/callback", headers={"Cookie": "_lakefs_oauth2=UNIT_TEST_COOKIE"})
    resp = conn.getresponse()
    resp.read()
    th.join(timeout=10)

    assert resp.status == 200
    assert result["cookie"] == "UNIT_TEST_COOKIE"
