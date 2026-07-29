"""Kraken REST authentication: nonce generation and request signing.

Kraken's signature scheme is fiddly and fails opaquely -- a wrong signature
returns ``EAPI:Invalid key``, which is indistinguishable from actually having
the wrong key. The scheme is::

    API-Sign = base64( HMAC-SHA512(
        key     = base64_decode(api_secret),
        message = uri_path || SHA256(nonce || urlencoded_postdata)
    ))

Three details cause most failures:

* The nonce is prepended to the *already URL-encoded* POST body before hashing,
  and it must be the same string that appears inside that body.
* ``uri_path`` is the path only (``/0/private/Balance``), never the full URL.
* The secret is base64-decoded before use as the HMAC key, not used raw.

Isolated from the venue client so it can be tested against Kraken's published
vector without any network access.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import threading
import time
import urllib.parse
from typing import Mapping


class KrakenAuthError(RuntimeError):
    """Malformed credentials, detected before any request is sent."""


class NonceGenerator:
    """Strictly increasing nonces, safe across threads.

    Kraken rejects any nonce not greater than the previous one for that key.
    Two requests issued inside the same millisecond -- routine when analyst
    threads fan out -- would otherwise collide and fail. The counter is
    monotonic rather than purely clock-based so it survives that case and small
    backwards clock adjustments.
    """

    def __init__(self, start: int | None = None) -> None:
        self._lock = threading.Lock()
        self._last = start if start is not None else int(time.time() * 1000)

    def __call__(self) -> int:
        with self._lock:
            candidate = int(time.time() * 1000)
            # Never emit a value we have already used for this key.
            self._last = max(candidate, self._last + 1)
            return self._last


def encode_postdata(payload: Mapping[str, object]) -> str:
    """URL-encode a request body.

    The exact encoded string is hashed, so it must be built once and reused for
    both the signature and the transmitted body -- re-encoding the dict a second
    time can reorder keys and silently invalidate the signature.
    """
    return urllib.parse.urlencode(payload)


def sign_request(uri_path: str, postdata: str, nonce: int | str, api_secret: str) -> str:
    """Compute the ``API-Sign`` header value.

    ``postdata`` must already be URL-encoded and must already contain ``nonce``.
    """
    if not uri_path.startswith("/"):
        raise KrakenAuthError(f"uri_path must be a path, not a URL: {uri_path!r}")

    try:
        secret = base64.b64decode(api_secret, validate=True)
    except (ValueError, TypeError) as exc:
        raise KrakenAuthError(
            "KRAKEN_API_SECRET is not valid base64 -- copy the 'Private key' "
            "value from Kraken exactly, including any trailing '=' padding"
        ) from exc

    sha = hashlib.sha256(f"{nonce}{postdata}".encode()).digest()
    mac = hmac.new(secret, uri_path.encode() + sha, hashlib.sha512)
    return base64.b64encode(mac.digest()).decode()


class KrakenCredentials:
    """API key pair plus the nonce sequence bound to it.

    The nonce generator lives here because Kraken tracks nonces *per key*: two
    credential objects sharing one counter would work, but two counters sharing
    one key would produce rejections.
    """

    def __init__(self, api_key: str, api_secret: str, nonce: NonceGenerator | None = None) -> None:
        api_key = (api_key or "").strip()
        api_secret = (api_secret or "").strip()
        if not api_key or not api_secret:
            raise KrakenAuthError(
                "missing Kraken credentials: set KRAKEN_API_KEY and KRAKEN_API_SECRET"
            )
        # Validate the secret now rather than on the first live order.
        try:
            base64.b64decode(api_secret, validate=True)
        except (ValueError, TypeError) as exc:
            raise KrakenAuthError(
                "KRAKEN_API_SECRET is not valid base64 -- it should be the long "
                "'Private key' string shown once at key creation"
            ) from exc

        self.api_key = api_key
        self._api_secret = api_secret
        self._nonce = nonce or NonceGenerator()

    def signed_request(
        self, uri_path: str, params: Mapping[str, object] | None = None
    ) -> tuple[dict[str, str], str]:
        """Build the headers and encoded body for a private endpoint call.

        Returns ``(headers, body)``. The body must be sent verbatim -- rebuilding
        it from a dict would break the signature.
        """
        payload: dict[str, object] = {"nonce": self._nonce()}
        if params:
            payload.update(params)
        body = encode_postdata(payload)
        signature = sign_request(uri_path, body, payload["nonce"], self._api_secret)
        headers = {
            "API-Key": self.api_key,
            "API-Sign": signature,
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "SpinTrader/0.1",
        }
        return headers, body

    def __repr__(self) -> str:
        # Never let a secret reach a log line or a traceback.
        return f"KrakenCredentials(api_key={self.api_key[:6]}...)"


__all__ = [
    "KrakenAuthError", "KrakenCredentials", "NonceGenerator",
    "encode_postdata", "sign_request",
]
