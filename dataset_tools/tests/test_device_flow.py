"""Device-flow login tests.

The point of device flow here is that NOTHING passes from the browser to the
CLI — that handoff is what broke every previous login design. These tests pin
the behaviours that keep it that way, plus the RFC 8628 polling semantics that
keep Zitadel from rate-limiting us.
"""
from __future__ import annotations

import json

import pytest

from dataset_cli import device_flow


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise device_flow.requests.HTTPError(f"HTTP {self.status_code}")


def test_discover_endpoints_reads_device_endpoint(monkeypatch):
    doc = {
        "device_authorization_endpoint": "https://z/oauth/v2/device_authorization",
        "token_endpoint": "https://z/oauth/v2/token",
    }
    monkeypatch.setattr(device_flow.requests, "get", lambda *a, **k: _Resp(doc))
    got = device_flow.discover_endpoints("https://z")
    assert got["device_authorization_endpoint"].endswith("/device_authorization")
    assert got["token_endpoint"].endswith("/token")


def test_discover_endpoints_rejects_issuer_without_device_grant(monkeypatch):
    """A Zitadel without the device endpoint must fail loudly, not silently."""
    monkeypatch.setattr(device_flow.requests, "get",
                        lambda *a, **k: _Resp({"token_endpoint": "https://z/t"}))
    with pytest.raises(device_flow.DeviceFlowError, match="device authorization endpoint"):
        device_flow.discover_endpoints("https://z")


def test_request_device_code_sends_project_audience_scope(monkeypatch):
    """Without the project-audience scope oauth2-proxy rejects the bearer.

    --extra-jwt-issuers names the Zitadel PROJECT id as the expected audience,
    so a token minted without this scope is valid but useless against lakeFS.
    """
    seen = {}

    def fake_post(url, data=None, timeout=None):
        seen.update(data or {})
        return _Resp({"device_code": "dc", "user_code": "UC-1",
                      "verification_uri": "https://z/device"})

    monkeypatch.setattr(device_flow.requests, "post", fake_post)
    device_flow.request_device_code("https://z/dev", "client-1",
                                    audience_project_id="proj-9")

    assert "urn:zitadel:iam:org:project:id:proj-9:aud" in seen["scope"]
    # A public client sends NO secret — that is the whole point of the app shape.
    assert "client_secret" not in seen


def test_request_device_code_surfaces_rejection(monkeypatch):
    monkeypatch.setattr(device_flow.requests, "post", lambda *a, **k: _Resp(
        {"error": "unauthorized_client",
         "error_description": 'grant_type "device_code" not allowed'}))
    with pytest.raises(device_flow.DeviceFlowError, match="unauthorized_client"):
        device_flow.request_device_code("https://z/dev", "client-1")


def test_poll_keeps_waiting_through_authorization_pending(monkeypatch):
    """authorization_pending is the normal state while the user is signing in."""
    calls = {"n": 0}

    def fake_post(url, data=None, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            return _Resp({"error": "authorization_pending"})
        return _Resp({"access_token": "tok-abc", "expires_in": 3600})

    monkeypatch.setattr(device_flow.requests, "post", fake_post)
    monkeypatch.setattr(device_flow.time, "sleep", lambda s: None)

    tokens = device_flow.poll_for_token("https://z/t", "c", "dc", interval=1)
    assert tokens["access_token"] == "tok-abc"
    assert calls["n"] == 3


def test_poll_backs_off_on_slow_down(monkeypatch):
    """RFC 8628: slow_down must widen the interval, or Zitadel rate-limits us."""
    waits = []
    calls = {"n": 0}

    def fake_post(url, data=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Resp({"error": "slow_down"})
        return _Resp({"access_token": "tok", "expires_in": 3600})

    monkeypatch.setattr(device_flow.requests, "post", fake_post)
    monkeypatch.setattr(device_flow.time, "sleep", lambda s: waits.append(s))

    device_flow.poll_for_token("https://z/t", "c", "dc", interval=5)
    assert waits[1] > waits[0], f"interval did not widen after slow_down: {waits}"


def test_poll_reports_denial_distinctly(monkeypatch):
    monkeypatch.setattr(device_flow.requests, "post",
                        lambda *a, **k: _Resp({"error": "access_denied"}))
    monkeypatch.setattr(device_flow.time, "sleep", lambda s: None)
    with pytest.raises(device_flow.DeviceFlowError, match="denied"):
        device_flow.poll_for_token("https://z/t", "c", "dc", interval=1)


def test_poll_survives_transient_network_error(monkeypatch):
    """A blip must not abandon an approval the user may already have granted."""
    calls = {"n": 0}

    def fake_post(url, data=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise device_flow.requests.ConnectionError("boom")
        return _Resp({"access_token": "tok", "expires_in": 3600})

    monkeypatch.setattr(device_flow.requests, "post", fake_post)
    monkeypatch.setattr(device_flow.time, "sleep", lambda s: None)

    tokens = device_flow.poll_for_token("https://z/t", "c", "dc", interval=1)
    assert tokens["access_token"] == "tok"


def test_poll_times_out_rather_than_hanging(monkeypatch):
    monkeypatch.setattr(device_flow.requests, "post",
                        lambda *a, **k: _Resp({"error": "authorization_pending"}))
    monkeypatch.setattr(device_flow.time, "sleep", lambda s: None)
    # expires_in=0 means the deadline has already passed on entry.
    with pytest.raises(device_flow.DeviceFlowError, match="timed out"):
        device_flow.poll_for_token("https://z/t", "c", "dc", interval=1, expires_in=0)


def test_login_prints_code_for_headless_use(monkeypatch, capsys):
    """A headless user must be able to finish from another device: the CLI
    never needs anything handed back, so printing the URL+code is sufficient."""
    monkeypatch.setattr(device_flow, "discover_endpoints", lambda *a, **k: {
        "device_authorization_endpoint": "https://z/dev", "token_endpoint": "https://z/t"})
    monkeypatch.setattr(device_flow, "request_device_code", lambda *a, **k: {
        "device_code": "dc", "user_code": "WXYZ-1234",
        "verification_uri": "https://z/device",
        "verification_uri_complete": "https://z/device?user_code=WXYZ-1234",
        "interval": 1, "expires_in": 300})
    monkeypatch.setattr(device_flow, "poll_for_token", lambda *a, **k: {"access_token": "tok"})
    opened = {}
    monkeypatch.setattr(device_flow.webbrowser, "open",
                        lambda u: opened.setdefault("url", u))

    device_flow.login("https://z", "client-1", audience_project_id="p")
    out = capsys.readouterr().out
    assert "WXYZ-1234" in out
    assert "https://z/device" in out
    assert opened["url"].startswith("https://z/device")
