"""OAuth 2.0 device authorization grant (RFC 8628) against Zitadel.

Why this exists: every previous CLI login design tried to hand a credential
from the BROWSER to the CLI, and each one broke on that handoff.

  - Loopback cookie capture: the ``_lakefs_oauth2`` cookie is Secure+HttpOnly
    and scoped to the lakeFS domain, so the browser never sends it to
    http://localhost. Hung forever.
  - Same-origin return page: oauth2-proxy strips its own session cookie before
    forwarding upstream, so the nginx sidecar that was supposed to echo the
    cookie back never sees one. Returns 401 for every user, permanently.

Device flow removes the handoff instead of working around it. The browser and
the CLI each talk to Zitadel independently:

  1. CLI  -> Zitadel:  POST /oauth/v2/device_authorization  (public client_id,
     no secret) -> device_code + user_code + verification_uri.
  2. User -> browser:  opens verification_uri, signs in, approves.
  3. CLI  -> Zitadel:  polls POST /oauth/v2/token until the approval lands.

Nothing passes between the browser and the CLI, so there is nothing to
intercept, strip, or paste. The resulting token is per-user, short-lived,
audience-scoped and revocable in Zitadel — strictly better than the 4h
shared-admin-equivalent session cookie it replaces.

The CLI's Zitadel app is PUBLIC (NATIVE + auth method NONE), so no client
secret ever lives on a developer machine. The operator provisions it per
project and publishes its client_id in
``KubeProject.status.mlStack.datasetCLIOIDC`` (see kubecore-operator #1347).
"""
from __future__ import annotations

import time
import urllib.parse
import webbrowser
from typing import Optional

import requests

# RFC 8628 polling errors that mean "keep waiting", not "give up".
_PENDING = {"authorization_pending", "slow_down"}

# Fallback poll interval when Zitadel omits `interval` from its response.
_DEFAULT_INTERVAL = 5


class DeviceFlowError(RuntimeError):
    """The device authorization flow failed or was denied."""


def discover_endpoints(issuer: str, timeout: int = 30) -> dict:
    """Fetch the OIDC discovery document and return the endpoints we need.

    Discovery is used rather than hardcoded paths so a Zitadel version or
    deployment that moves them does not silently break login.
    """
    url = issuer.rstrip("/")
    if not url.endswith("/.well-known/openid-configuration"):
        url = f"{url}/.well-known/openid-configuration"
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        doc = r.json()
    except (requests.RequestException, ValueError) as exc:
        raise DeviceFlowError(f"could not read OIDC discovery at {url}: {exc}") from exc

    device = doc.get("device_authorization_endpoint")
    token = doc.get("token_endpoint")
    if not device or not token:
        raise DeviceFlowError(
            f"{url} does not advertise a device authorization endpoint "
            "(is this Zitadel, and is the device grant enabled?)"
        )
    return {"device_authorization_endpoint": device, "token_endpoint": token}


def request_device_code(device_endpoint: str, client_id: str,
                        audience_project_id: Optional[str] = None,
                        timeout: int = 30) -> dict:
    """Start the flow. Returns Zitadel's device_code/user_code payload."""
    scopes = ["openid", "profile", "offline_access"]
    if audience_project_id:
        # oauth2-proxy validates the bearer against --extra-jwt-issuers, which
        # names the Zitadel PROJECT id as the expected audience. Without this
        # scope the token is minted but the proxy rejects it.
        scopes.append(f"urn:zitadel:iam:org:project:id:{audience_project_id}:aud")

    try:
        r = requests.post(
            device_endpoint,
            data={"client_id": client_id, "scope": " ".join(scopes)},
            timeout=timeout,
        )
        payload = r.json()
    except (requests.RequestException, ValueError) as exc:
        raise DeviceFlowError(f"device authorization request failed: {exc}") from exc

    if payload.get("error"):
        raise DeviceFlowError(
            f"device authorization rejected: {payload['error']} - "
            f"{payload.get('error_description', '')}"
        )
    if not payload.get("device_code") or not payload.get("user_code"):
        raise DeviceFlowError(f"device authorization returned no code: {payload}")
    return payload


def poll_for_token(token_endpoint: str, client_id: str, device_code: str,
                   interval: int = _DEFAULT_INTERVAL,
                   expires_in: int = 300, timeout: int = 30) -> dict:
    """Poll until the user approves, the code expires, or Zitadel refuses.

    Honours RFC 8628 ``slow_down`` by backing the interval off, which is what
    keeps Zitadel from rate-limiting an over-eager client.
    """
    deadline = time.time() + expires_in
    wait = max(interval, 1)

    while time.time() < deadline:
        time.sleep(wait)
        try:
            r = requests.post(
                token_endpoint,
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "device_code": device_code,
                    "client_id": client_id,
                },
                timeout=timeout,
            )
            payload = r.json()
        except (requests.RequestException, ValueError):
            # A transient network blip must not abandon an approval the user
            # may already have granted; keep polling until the deadline.
            continue

        err = payload.get("error")
        if not err:
            if not payload.get("access_token"):
                raise DeviceFlowError(f"token response carried no access_token: {payload}")
            return payload
        if err == "slow_down":
            wait += 5
            continue
        if err in _PENDING:
            continue
        if err == "access_denied":
            raise DeviceFlowError("sign-in was denied in the browser")
        if err == "expired_token":
            raise DeviceFlowError("the sign-in code expired before it was approved")
        raise DeviceFlowError(
            f"token request failed: {err} - {payload.get('error_description', '')}"
        )

    raise DeviceFlowError("timed out waiting for the browser sign-in to be approved")


def login(issuer: str, client_id: str, audience_project_id: Optional[str] = None,
          open_browser: bool = True) -> dict:
    """Run the whole flow and return Zitadel's token payload.

    Prints the verification URL and user code so a user on a headless machine
    (or whose browser did not open) can complete it from any device — the CLI
    never needs anything handed back to it.
    """
    endpoints = discover_endpoints(issuer)
    device = request_device_code(
        endpoints["device_authorization_endpoint"], client_id, audience_project_id)

    complete = device.get("verification_uri_complete")
    verify = device.get("verification_uri", "")
    code = device["user_code"]

    print("\n🔑  Sign in to continue.")
    print(f"    Code: {code}")
    print(f"    Open: {complete or verify}")
    if complete and verify:
        print(f"    (or go to {verify} and enter the code above)")
    print()

    if open_browser:
        try:
            webbrowser.open(complete or verify)
        except webbrowser.Error:
            # Headless or no browser configured — the printed URL is enough.
            pass

    print("⏳  Waiting for you to approve in the browser…")
    tokens = poll_for_token(
        endpoints["token_endpoint"],
        client_id,
        device["device_code"],
        interval=int(device.get("interval", _DEFAULT_INTERVAL)),
        expires_in=int(device.get("expires_in", 300)),
    )
    print("✓ Signed in.")
    return tokens


def build_verification_prompt(device: dict) -> str:
    """Render the user-facing prompt for a device payload (used by tests)."""
    target = device.get("verification_uri_complete") or device.get("verification_uri", "")
    return f"{device.get('user_code', '')} -> {urllib.parse.urlsplit(target).geturl()}"
