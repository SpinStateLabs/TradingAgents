#!/usr/bin/env bash
# Run the held-back eval cases against a pinned agent version and collect verdicts.
# Usage: AGENT_VERSION=<n> ./evals/run-evals.sh   (run from my-agent/)
set -euo pipefail
set -a; source .env; source IDS.env; set +a
BASE=https://api.anthropic.com/v1
H=(-H "x-api-key: $ANTHROPIC_API_KEY" -H "anthropic-version: 2023-06-01" \
   -H "anthropic-beta: managed-agents-2026-04-01" -H "content-type: application/json")
VER="${AGENT_VERSION:?set AGENT_VERSION to the version under test}"
OUT="evals/results-v${VER}.json"; echo "[]" > "$OUT"

for dir in evals/case-*/; do
  [ -f "$dir/input.md" ] || continue
  echo "== $dir =="
  sess=$(curl -sS "$BASE/sessions" "${H[@]}" -d "$(python3 -c "import json;print(json.dumps({'agent':{'type':'agent','id':'$AGENT_ID','version':int('$VER')},'environment_id':'$ENV_ID','title':'eval '+'$dir','resources':[{'type':'memory_store','memory_store_id':'$MEMSTORE_ID','access':'read_only'}]}))")" \
        | python3 -c "import json,sys;print(json.JSONDecoder(strict=False).decode(sys.stdin.read())['id'])")
  evt=$(python3 -c "import json;print(json.dumps({'type':'user.define_outcome','description':open('$dir/input.md').read(),'rubric':{'type':'text','content':open('outcome.md').read()},'max_iterations':3}))")
  curl -sS "$BASE/sessions/$sess/events" "${H[@]}" -d "{\"events\":[$evt]}" >/dev/null
  echo "  session $sess started (poll it; append verdict + usage to $OUT when idle)"
done
echo "Eval sessions launched. Poll each and record outcome_evaluations[].result + usage into $OUT."
