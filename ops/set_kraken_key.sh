#!/usr/bin/env bash
# Install Kraken API credentials into .env without exposing them.
#
# Both values are read with echo disabled, so they never appear on screen, in
# shell history, in the process table (`ps` shows arguments, so credentials
# must never be passed as ones), or in any log.
#
# Run it directly on the machine that will hold the key:
#
#     ssh -t -i ~/.ssh/gx10_ed25519 spinner@10.0.0.62 \
#         'cd ~/spintrader-src && ops/set_kraken_key.sh'
#
# The -t is required: without a TTY the silent prompt cannot work.
set -euo pipefail

ENV_FILE="${SPINTRADER_ENV_FILE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.env}"

if [[ ! -t 0 ]]; then
  echo "error: no terminal attached. Re-run with 'ssh -t ...' so the prompt can hide input." >&2
  exit 1
fi

echo "Installing Kraken credentials into: $ENV_FILE"
echo "(input is hidden; nothing is echoed or written to history)"
echo

read -r -s -p "  API key    : " KRAKEN_KEY; echo
read -r -s -p "  Private key: " KRAKEN_SECRET; echo
echo

# Trim whitespace picked up by copy-paste from the Kraken UI.
KRAKEN_KEY="$(printf '%s' "$KRAKEN_KEY" | tr -d '[:space:]')"
KRAKEN_SECRET="$(printf '%s' "$KRAKEN_SECRET" | tr -d '[:space:]')"

if [[ -z "$KRAKEN_KEY" || -z "$KRAKEN_SECRET" ]]; then
  echo "error: both values are required; nothing written." >&2
  exit 1
fi

# Validate the secret decodes as base64 before writing it. Catching this now
# beats an opaque 'EAPI:Invalid key' on the first live call.
if ! printf '%s' "$KRAKEN_SECRET" | base64 -d >/dev/null 2>&1; then
  echo "error: the private key is not valid base64." >&2
  echo "       Copy the long 'Private key' string from Kraken, including any trailing '='." >&2
  exit 1
fi

touch "$ENV_FILE"
chmod 600 "$ENV_FILE"

# Rewrite the two keys in place, preserving every other line and any comments.
# A temp file in the same directory keeps the replacement atomic, so an
# interrupted run cannot leave a half-written .env behind.
TMP="$(mktemp "${ENV_FILE}.XXXXXX")"
chmod 600 "$TMP"
trap 'rm -f "$TMP"' EXIT

awk -v key="$KRAKEN_KEY" -v secret="$KRAKEN_SECRET" '
  /^[[:space:]]*KRAKEN_API_KEY=/    { print "KRAKEN_API_KEY=" key;      seen_key=1;    next }
  /^[[:space:]]*KRAKEN_API_SECRET=/ { print "KRAKEN_API_SECRET=" secret; seen_secret=1; next }
  { print }
  END {
    if (!seen_key)    print "KRAKEN_API_KEY=" key
    if (!seen_secret) print "KRAKEN_API_SECRET=" secret
  }
' "$ENV_FILE" > "$TMP"

mv "$TMP" "$ENV_FILE"
chmod 600 "$ENV_FILE"
trap - EXIT

unset KRAKEN_KEY KRAKEN_SECRET

echo "Written to $ENV_FILE (mode 600)."
echo
echo "Verify with:"
echo "    PYTHONPATH=. .venv/bin/python ops/check_kraken.py"
