"""Tests for Kraken request signing.

The headline test is :meth:`SignatureVectorTests.test_official_kraken_vector`,
which checks our implementation against the worked example published in
Kraken's own API docs. A wrong signature is indistinguishable from a wrong key
at the API boundary -- both return ``EAPI:Invalid key`` -- so without a known-
good vector, debugging a live auth failure means guessing.
"""

from __future__ import annotations

import base64
import threading
import unittest

from spintrader.venues.kraken_auth import (
    KrakenAuthError, KrakenCredentials, NonceGenerator, encode_postdata,
    sign_request,
)

# Published test vector from Kraken's REST authentication documentation.
VECTOR_SECRET = (
    "kQH5HW/8p1uGOVjbgWA7FunAmGO8lsSUXNsu3eow76sz84Q18fWxnyRzBHCd3pd5nE9qa99HAZtuZuj6F1huXg=="
)
VECTOR_PATH = "/0/private/AddOrder"
VECTOR_NONCE = 1616492376594
VECTOR_POSTDATA = (
    "nonce=1616492376594&ordertype=limit&pair=XBTUSD&price=37500&type=buy&volume=1.25"
)
VECTOR_SIGNATURE = (
    "4/dpxb3iT4tp/ZCVEwSnEsLxx0bqyhLpdfOpc6fn7OR8+UClSV5n9E6aSS8MPtnRfp32bAb0nmbRn6H8ndwLUQ=="
)

# A syntactically valid secret for tests that do not check the vector.
DUMMY_SECRET = base64.b64encode(b"\x01" * 64).decode()


class SignatureVectorTests(unittest.TestCase):
    def test_official_kraken_vector(self):
        self.assertEqual(
            sign_request(VECTOR_PATH, VECTOR_POSTDATA, VECTOR_NONCE, VECTOR_SECRET),
            VECTOR_SIGNATURE,
        )

    def test_nonce_accepted_as_string(self):
        # The nonce is stringified into the hash; int and str must agree, since
        # the value also appears as text inside postdata.
        self.assertEqual(
            sign_request(VECTOR_PATH, VECTOR_POSTDATA, str(VECTOR_NONCE), VECTOR_SECRET),
            VECTOR_SIGNATURE,
        )

    def test_signature_depends_on_path(self):
        other = sign_request("/0/private/Balance", VECTOR_POSTDATA, VECTOR_NONCE, VECTOR_SECRET)
        self.assertNotEqual(other, VECTOR_SIGNATURE)

    def test_signature_depends_on_nonce(self):
        other = sign_request(VECTOR_PATH, VECTOR_POSTDATA, VECTOR_NONCE + 1, VECTOR_SECRET)
        self.assertNotEqual(other, VECTOR_SIGNATURE)

    def test_signature_depends_on_body(self):
        tampered = VECTOR_POSTDATA.replace("volume=1.25", "volume=12.5")
        self.assertNotEqual(
            sign_request(VECTOR_PATH, tampered, VECTOR_NONCE, VECTOR_SECRET),
            VECTOR_SIGNATURE,
        )

    def test_full_url_rejected(self):
        # Passing the full URL is the most common porting mistake; it must fail
        # loudly here rather than as an opaque "Invalid key" from the API.
        with self.assertRaises(KrakenAuthError) as ctx:
            sign_request("https://api.kraken.com/0/private/Balance",
                         VECTOR_POSTDATA, VECTOR_NONCE, VECTOR_SECRET)
        self.assertIn("must be a path", str(ctx.exception))

    def test_non_base64_secret_rejected_with_guidance(self):
        with self.assertRaises(KrakenAuthError) as ctx:
            sign_request(VECTOR_PATH, VECTOR_POSTDATA, VECTOR_NONCE, "not-base64!!")
        self.assertIn("base64", str(ctx.exception))


class EncodePostdataTests(unittest.TestCase):
    def test_encodes_pairs(self):
        self.assertEqual(encode_postdata({"nonce": 1, "pair": "XBTUSD"}), "nonce=1&pair=XBTUSD")

    def test_preserves_insertion_order(self):
        # Order matters: the encoded string is what gets hashed, so a reordering
        # between signing and sending would invalidate the signature.
        self.assertEqual(
            encode_postdata({"nonce": 1, "b": 2, "a": 3}), "nonce=1&b=2&a=3"
        )

    def test_escapes_special_characters(self):
        self.assertIn("userref=a%2Bb", encode_postdata({"userref": "a+b"}))


class NonceTests(unittest.TestCase):
    def test_strictly_increasing(self):
        gen = NonceGenerator()
        values = [gen() for _ in range(100)]
        self.assertEqual(values, sorted(values))
        self.assertEqual(len(set(values)), 100)

    def test_no_collision_within_one_millisecond(self):
        # Rapid successive calls land in the same millisecond; a purely
        # clock-based nonce would repeat and Kraken would reject the second.
        gen = NonceGenerator(start=1_000_000)
        self.assertNotEqual(gen(), gen())

    def test_survives_backwards_clock_step(self):
        gen = NonceGenerator(start=10**15)   # far ahead of wall clock
        first, second = gen(), gen()
        self.assertGreater(second, first)

    def test_thread_safe(self):
        gen = NonceGenerator()
        collected: list[int] = []
        lock = threading.Lock()

        def worker():
            local = [gen() for _ in range(50)]
            with lock:
                collected.extend(local)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(collected), 400)
        self.assertEqual(len(set(collected)), 400, "nonce collision across threads")


class CredentialsTests(unittest.TestCase):
    def test_missing_credentials_rejected(self):
        for key, secret in [("", DUMMY_SECRET), ("k", ""), ("", "")]:
            with self.subTest(key=key, secret=bool(secret)):
                with self.assertRaises(KrakenAuthError) as ctx:
                    KrakenCredentials(key, secret)
                self.assertIn("KRAKEN_API_KEY", str(ctx.exception))

    def test_bad_secret_rejected_at_construction(self):
        # Fail at startup, not on the first live order.
        with self.assertRaises(KrakenAuthError):
            KrakenCredentials("key", "definitely not base64 ***")

    def test_whitespace_is_stripped(self):
        # Copy-paste from the Kraken UI routinely picks up a trailing newline.
        creds = KrakenCredentials("  key123  \n", f"  {DUMMY_SECRET}  ")
        self.assertEqual(creds.api_key, "key123")

    def test_signed_request_shape(self):
        creds = KrakenCredentials("key123", DUMMY_SECRET)
        headers, body = creds.signed_request("/0/private/Balance")
        self.assertEqual(headers["API-Key"], "key123")
        self.assertTrue(headers["API-Sign"])
        self.assertEqual(headers["Content-Type"], "application/x-www-form-urlencoded")
        self.assertTrue(body.startswith("nonce="))

    def test_params_included_in_body(self):
        creds = KrakenCredentials("key123", DUMMY_SECRET)
        _, body = creds.signed_request("/0/private/AddOrder", {"pair": "XBTUSD", "type": "buy"})
        self.assertIn("pair=XBTUSD", body)
        self.assertIn("type=buy", body)
        self.assertTrue(body.startswith("nonce="))

    def test_body_signature_is_self_consistent(self):
        # Re-deriving the signature from the returned body must reproduce the
        # header, proving we sign exactly what we transmit.
        creds = KrakenCredentials("key123", DUMMY_SECRET)
        headers, body = creds.signed_request("/0/private/Balance")
        nonce = body.split("nonce=", 1)[1].split("&", 1)[0]
        self.assertEqual(
            sign_request("/0/private/Balance", body, nonce, DUMMY_SECRET),
            headers["API-Sign"],
        )

    def test_successive_requests_use_distinct_nonces(self):
        creds = KrakenCredentials("key123", DUMMY_SECRET)
        _, first = creds.signed_request("/0/private/Balance")
        _, second = creds.signed_request("/0/private/Balance")
        self.assertNotEqual(first, second)

    def test_repr_does_not_leak_the_secret(self):
        creds = KrakenCredentials("key1234567890", DUMMY_SECRET)
        text = repr(creds)
        self.assertNotIn(DUMMY_SECRET, text)
        self.assertNotIn("1234567890", text)


if __name__ == "__main__":
    unittest.main()
