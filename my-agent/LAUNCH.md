# LAUNCH.md — Volatility-Trader (crypto, paper, scheduled)

Resumable launch sequence. Each step sources `IDS.env` first and skips objects that already
exist, so any step can be re-run without creating duplicates. Run from `my-agent/`.

## 0. Setup (every terminal)

```bash
cd my-agent
set -a; source .env; set +a          # ANTHROPIC_API_KEY=sk-ant-...  (never printed, never in chat)
touch IDS.env; set -a; source IDS.env; set +a
BASE=https://api.anthropic.com/v1
H=(-H "x-api-key: $ANTHROPIC_API_KEY" -H "anthropic-version: 2023-06-01" \
   -H "anthropic-beta: managed-agents-2026-04-01" -H "content-type: application/json")
pyid() { python3 -c "import json,sys;print(json.JSONDecoder(strict=False).decode(open(sys.argv[1]).read())[sys.argv[2]])" "$1" "$2"; }
```

## 1. Pick model (newest Opus-class)

```bash
MODEL=$(curl -sS "$BASE/models" "${H[@]:0:4}" | python3 -c "import json,sys; ids=[m['id'] for m in json.load(sys.stdin)['data']]; opus=[i for i in ids if 'opus' in i]; print(sorted(opus)[-1] if opus else ids[0])")
echo "MODEL=$MODEL"
# write MODEL into agent.json (replace PICKED-AT-LAUNCH)
python3 -c "import json; a=json.load(open('agent.json')); a['model']={'id':'$MODEL'}; json.dump(a,open('agent.json','w'),indent=2)"
```

## 2. Environment  (skip if ENV_ID set)

```bash
[ -z "$ENV_ID" ] && { curl -sS --fail-with-body "$BASE/environments" "${H[@]}" -d @environment.json -o /tmp/env.json && ENV_ID=$(pyid /tmp/env.json id) && echo "ENV_ID=$ENV_ID" >> IDS.env; }
echo "ENV_ID=$ENV_ID"
```
`📦 environment` — the sandboxed container the agent runs in (cloud, unrestricted networking so it can reach public crypto data APIs; pandas/numpy for the vol math).

## 3. Memory store (paper portfolio)  (skip if MEMSTORE_ID set)

```bash
[ -z "$MEMSTORE_ID" ] && { curl -sS --fail-with-body "$BASE/memory_stores" "${H[@]}" \
  -d '{"name":"volatility-trader-paper-portfolio","description":"Simulated paper trading account: cash, equity, open positions, closed trades. The agent marks to live market prices and updates it every run."}' \
  -o /tmp/mem.json && MEMSTORE_ID=$(pyid /tmp/mem.json id) && echo "MEMSTORE_ID=$MEMSTORE_ID" >> IDS.env; }
# seed the starting portfolio
curl -sS --fail-with-body "$BASE/memory_stores/$MEMSTORE_ID/memories" "${H[@]}" \
  -d "$(python3 -c "import json;print(json.dumps({'path':'/portfolio.json','content':open('memory-seed/portfolio.json').read()}))")"
echo "MEMSTORE_ID=$MEMSTORE_ID"
```
`🧠 memory store` — the persistent, cross-run state. This is where the simulated account lives; mounted at `/mnt/memory/` in every run.

## 4. Agent  (skip if AGENT_ID set)

```bash
[ -z "$AGENT_ID" ] && { curl -sS --fail-with-body "$BASE/agents" "${H[@]}" -d @agent.json -o /tmp/agent.json \
  && AGENT_ID=$(pyid /tmp/agent.json id) && AGENT_VERSION=$(pyid /tmp/agent.json version) \
  && echo "AGENT_ID=$AGENT_ID" >> IDS.env && echo "AGENT_VERSION=$AGENT_VERSION" >> IDS.env; }
echo "AGENT_ID=$AGENT_ID  v$AGENT_VERSION"
```
`🤖 agent` — the brain: model + system instructions + the full tool set. Versioned; any change is a new version.

## 5. Session + outcome kickoff (first live run, on eval case 1)

```bash
curl -sS --fail-with-body "$BASE/sessions" "${H[@]}" -d "$(python3 -c "
import json
print(json.dumps({
  'agent': '$AGENT_ID',
  'environment_id': '$ENV_ID',
  'title': 'volatility-trader v0 — first run',
  'resources': [{'type':'memory_store','memory_store_id':'$MEMSTORE_ID','access':'read_write','instructions':'Your simulated paper account. Load /portfolio.json at the start of every run, mark to market, honor stops/targets, then save it back.'}]
}))")" -o /tmp/sess.json
SESSION_ID=$(pyid /tmp/sess.json id); echo "SESSION_ID=$SESSION_ID" >> IDS.env
echo "SESSION_ID=$SESSION_ID"

# outcome kickoff: task from first_prompt.txt, rubric from outcome.md
EVT=$(python3 -c "import json;print(json.dumps({'type':'user.define_outcome','description':open('first_prompt.txt').read(),'rubric':{'type':'text','content':open('outcome.md').read()},'max_iterations':3}))")
curl -sS --fail-with-body "$BASE/sessions/$SESSION_ID/events" "${H[@]}" -d "{\"events\":[$EVT]}"
```

Checkpoint (paste each as it lands):
```
✅ 📦 environment  $ENV_ID
✅ 🧠 memory store $MEMSTORE_ID (paper portfolio seeded @ $100k)
✅ 🤖 agent        $AGENT_ID (v$AGENT_VERSION, $MODEL)
✅ ▶️ run started  $SESSION_ID
```
Console (Sessions): https://platform.claude.com/workspaces/default/sessions/$SESSION_ID
(If the key isn't in the `default` workspace, switch workspaces with the Console picker — Settings → API Keys shows which one the key belongs to.)

## 6. Watch it

```bash
# stream (open, then it shows tool calls + the grader verdict); Ctrl-C when idle
curl -sS -N --fail-with-body "$BASE/sessions/$SESSION_ID/events/stream" "${H[@]}" -H "accept: text/event-stream"
# or poll
curl -sS "$BASE/sessions/$SESSION_ID" "${H[@]}" -o /tmp/s.json
python3 -c "import json;d=json.JSONDecoder(strict=False).decode(open('/tmp/s.json').read());print(d['status'],[e.get('result') for e in d.get('outcome_evaluations',[])])"
```

## 7. Fetch the brief

```bash
curl -sS "$BASE/files?scope_id=$SESSION_ID" "${H[@]}" | python3 -c "import json,sys;[print(f['id'],f['filename']) for f in json.load(sys.stdin)['data']]"
curl -sS "$BASE/files/<FILE_ID>/content" "${H[@]}" -o out/brief.md
```

## 8. Scheduled deployment (after a run passes the rubric)

Build `deployment.json` from the same files (relative dates only — replayed every run):
```bash
python3 -c "
import json
kick={'type':'user.define_outcome','description':open('first_prompt.txt').read(),'rubric':{'type':'text','content':open('outcome.md').read()},'max_iterations':3}
dep={'name':'Volatility-Trader daily scan','agent':'$AGENT_ID','environment_id':'$ENV_ID',
     'initial_events':[kick],
     'resources':[{'type':'memory_store','memory_store_id':'$MEMSTORE_ID','access':'read_write','instructions':'Your simulated paper account. Load /portfolio.json, mark to market, honor stops/targets, save it back.'}],
     'schedule':{'type':'cron','expression':'0 8 * * *','timezone':'America/New_York'}}
json.dump(dep,open('deployment.json','w'),indent=2)"
curl -sS --fail-with-body "$BASE/deployments?beta=true" "${H[@]}" -d @deployment.json -o /tmp/dep.json
DEPLOYMENT_ID=$(pyid /tmp/dep.json id); echo "DEPLOYMENT_ID=$DEPLOYMENT_ID" >> IDS.env
python3 -c "import json;d=json.JSONDecoder(strict=False).decode(open('/tmp/dep.json').read());print('upcoming:',d.get('schedule',{}).get('upcoming_runs_at'))"
# prove the cron works now:
curl -sS -X POST -d '{}' "$BASE/deployments/$DEPLOYMENT_ID/run?beta=true" "${H[@]}"
```
`🗓️ deployment` — the cron trigger that fires this exact kickoff every day at 08:00 ET.
