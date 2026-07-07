"""Token-return-page login tests.

The old loopback design (capture the _lakefs_oauth2 cookie off a localhost GET)
could never work: the cookie is Secure+HttpOnly and scoped to the lakeFS domain,
so the browser never sends it to http://localhost. The new flow captures the
cookie via a same-origin return page (served on the lakeFS domain) that POSTs the
session JSON to the localhost catcher. These tests exercise that POST path plus
the CORS preflight the browser sends for a cross-origin application/json POST.
"""
from __future__ import annotations

import http.client
import json
import threading
import time
import urllib.parse as up

from dataset_cli import login


def _drive_login(monkeypatch, timeout=8):
    """Start loopback_login in a thread; return (thread, result, opened_url)."""
    opened: dict = {}
    monkeypatch.setattr(login.webbrowser, "open",
                        lambda url: opened.setdefault("url", url))
    result: dict = {}

    def run():
        result["cookie"] = login.loopback_login("https://lakefs.invalid", timeout=timeout)

    th = threading.Thread(target=run)
    th.start()
    time.sleep(1.0)  # let the server bind + browser "open"
    return th, result, opened


def test_return_page_target_is_same_origin(monkeypatch):
    """The browser must be pointed at oauth2/start with rd= a SAME-ORIGIN return
    page on the lakeFS domain (not http://localhost), carrying ?port=<bound>."""
    th, result, opened = _drive_login(monkeypatch)
    try:
        parsed = up.urlparse(opened["url"])
        assert parsed.path == "/oauth2/start"
        rd = up.parse_qs(parsed.query)["rd"][0]
        rd_parsed = up.urlparse(rd)
        # return page is served on the lakeFS domain, at /kubecore-cli/return
        assert rd_parsed.scheme == "https"
        assert rd_parsed.netloc == "lakefs.invalid"
        assert rd_parsed.path == login.RETURN_PATH
        port = up.parse_qs(rd_parsed.query)["port"][0]
        assert port.isdigit()
    finally:
        th.join(timeout=10)


def test_post_captures_cookie(monkeypatch):
    """The return page POSTs {"cookie": ...}; the catcher must capture it and the
    login call must return that cookie."""
    th, result, opened = _drive_login(monkeypatch)

    rd = up.parse_qs(up.urlparse(opened["url"]).query)["rd"][0]
    port = int(up.parse_qs(up.urlparse(rd).query)["port"][0])

    # simulate the return page's cross-origin POST of the session JSON
    body = json.dumps({"cookie_name": "_lakefs_oauth2",
                       "cookie": "POSTED_SESSION_COOKIE",
                       "base_url": "https://lakefs.invalid"}).encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("POST", "/callback", body=body,
                 headers={"Content-Type": "application/json",
                          "Origin": "https://lakefs.invalid"})
    resp = conn.getresponse()
    resp.read()
    th.join(timeout=10)

    assert resp.status == 200
    assert result["cookie"] == "POSTED_SESSION_COOKIE"


def test_cors_preflight_answered(monkeypatch):
    """A cross-origin application/json POST triggers a CORS preflight OPTIONS; the
    catcher must answer it with Access-Control-Allow-* so the fetch can proceed."""
    th, result, opened = _drive_login(monkeypatch)

    rd = up.parse_qs(up.urlparse(opened["url"]).query)["rd"][0]
    port = int(up.parse_qs(up.urlparse(rd).query)["port"][0])

    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("OPTIONS", "/callback",
                 headers={"Origin": "https://lakefs.invalid",
                          "Access-Control-Request-Method": "POST"})
    resp = conn.getresponse()
    resp.read()
    assert resp.status in (200, 204)
    assert resp.getheader("Access-Control-Allow-Origin") == "*"
    assert "POST" in (resp.getheader("Access-Control-Allow-Methods") or "")

    # complete the flow so the server thread exits cleanly
    conn2 = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn2.request("POST", "/callback",
                  body=json.dumps({"cookie": "C"}).encode(),
                  headers={"Content-Type": "application/json"})
    conn2.getresponse().read()
    th.join(timeout=10)
    assert result["cookie"] == "C"


def test_empty_post_does_not_capture(monkeypatch):
    """A POST with no cookie must be rejected (400) and not populate the session,
    so the flow keeps waiting rather than caching an empty cookie."""
    th, result, opened = _drive_login(monkeypatch, timeout=3)

    rd = up.parse_qs(up.urlparse(opened["url"]).query)["rd"][0]
    port = int(up.parse_qs(up.urlparse(rd).query)["port"][0])

    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("POST", "/callback", body=b"{}",
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    resp.read()
    assert resp.status == 400
    th.join(timeout=10)
    # nothing captured → loopback_login returns None (timeout), caller falls back
    assert result["cookie"] is None
