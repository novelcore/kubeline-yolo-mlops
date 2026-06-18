"""Zitadel machine-identity auth for MLflow (bearer JWT, auto-refreshing).

The pipeline authenticates to MLflow the same way a human does: a short-lived
Zitadel-signed JWT validated by the mlflow-oidc-auth plugin via JWKS. There is no
basic-auth password and no credential seeded into the plugin DB — the operator
provisions a per-project Zitadel machine user + key and publishes the key JSON to
the ``{project}-mlflow-svc`` secret, mounted here at ``ZITADEL_MACHINE_KEY_FILE``.

This module registers an MLflow ``RequestAuthProvider`` named ``zitadel`` that mints
a JWT from the machine key (JWT-profile / urn:ietf:params:oauth:grant-type:jwt-bearer)
and **caches it with proactive refresh**, so a long training run never 401s mid-flight:
MLflow calls ``get_auth()`` per request, and the returned auth object always carries a
valid (re-minted when near expiry) bearer token.

Select it by setting ``MLFLOW_TRACKING_AUTH=zitadel`` in the step environment.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Optional

import requests
from authlib.jose import jwt as jose_jwt
from mlflow.tracking.request_auth.abstract_request_auth_provider import (
    RequestAuthProvider,
)

# Re-mint this many seconds before the token actually expires, so an in-flight
# request never carries an about-to-expire token.
_REFRESH_LEEWAY_SECONDS = 120
# Assertion lifetime (the JWT we sign to request the access token). Short — it is
# single-use at the token endpoint.
_ASSERTION_TTL_SECONDS = 60


class _ZitadelTokenSource:
    """Mints and caches a Zitadel access token from a machine key JSON file."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._access_token: Optional[str] = None
        self._exp: float = 0.0

    @staticmethod
    def _key() -> dict:
        path = os.environ.get("ZITADEL_MACHINE_KEY_FILE")
        if not path:
            raise RuntimeError(
                "ZITADEL_MACHINE_KEY_FILE is not set — cannot authenticate to MLflow. "
                "The operator publishes the machine key to the {project}-mlflow-svc secret."
            )
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    @staticmethod
    def _token_endpoint() -> str:
        # Prefer an explicit override; else derive from the OIDC issuer/domain.
        ep = os.environ.get("ZITADEL_TOKEN_ENDPOINT")
        if ep:
            return ep
        domain = os.environ.get("ZITADEL_DOMAIN") or os.environ.get("OIDC_DOMAIN")
        if not domain:
            raise RuntimeError(
                "Set ZITADEL_TOKEN_ENDPOINT or ZITADEL_DOMAIN to reach the token endpoint."
            )
        return f"https://{domain}/oauth/v2/token"

    def _build_assertion(self, key: dict) -> str:
        # Zitadel JWT-profile assertion: iss/sub = userId, aud = issuer/domain,
        # signed RS256 with the machine key, header kid = keyId.
        now = int(time.time())
        domain = os.environ.get("ZITADEL_DOMAIN") or os.environ.get("OIDC_DOMAIN", "")
        audience = os.environ.get("ZITADEL_ISSUER") or f"https://{domain}"
        header = {"alg": "RS256", "kid": key["keyId"]}
        claims = {
            "iss": key["userId"],
            "sub": key["userId"],
            "aud": audience,
            "iat": now,
            "exp": now + _ASSERTION_TTL_SECONDS,
        }
        signed = jose_jwt.encode(header, claims, key["key"])
        return signed.decode("utf-8") if isinstance(signed, bytes) else signed

    def _mint(self) -> None:
        key = self._key()
        assertion = self._build_assertion(key)
        # Request the project audience scope so the access token carries the
        # group claims the mlflow-oidc-auth plugin maps to RBAC.
        scope = os.environ.get(
            "ZITADEL_SCOPE", "openid profile urn:zitadel:iam:org:projects:roles"
        )
        resp = requests.post(
            self._token_endpoint(),
            data={
                "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                "scope": scope,
                "assertion": assertion,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=15,
        )
        resp.raise_for_status()
        body = resp.json()
        self._access_token = body["access_token"]
        # Refresh proactively before the real expiry.
        self._exp = time.time() + int(body.get("expires_in", 3600))

    def token(self) -> str:
        with self._lock:
            if (
                self._access_token is None
                or time.time() >= self._exp - _REFRESH_LEEWAY_SECONDS
            ):
                self._mint()
            assert self._access_token is not None
            return self._access_token


_source = _ZitadelTokenSource()


class _BearerAuth(requests.auth.AuthBase):
    """Attaches a fresh Zitadel bearer token to every outgoing MLflow request."""

    def __call__(
        self, r: requests.PreparedRequest
    ) -> requests.PreparedRequest:  # noqa: D401
        r.headers["Authorization"] = f"Bearer {_source.token()}"
        return r


class ZitadelRequestAuthProvider(RequestAuthProvider):
    """MLflow request-auth provider selected via MLFLOW_TRACKING_AUTH=zitadel."""

    def get_name(self) -> str:
        return "zitadel"

    def get_auth(self) -> requests.auth.AuthBase:
        # Called per request by MLflow; the auth object re-mints on near-expiry,
        # so a multi-hour run never carries an expired token.
        return _BearerAuth()
