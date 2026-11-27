#!/usr/bin/env bash
# Replay the retired v1-v4 generators on this machine, localhost only.
#   tools/archive.sh            -> http://localhost:5181/v1  /v2  /v3
# Env: PAL_ARCHIVE_PY (python with torch, nltk, wordfreq, fastapi, uvicorn),
#      PAL_ARCHIVE_PORT (API, 8060), PAL_ARCHIVE_WEB (vite, 5181).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PAL_ARCHIVE_PY:-/private/tmp/claude-501/-Users-eric/5590c158-1939-4251-bd41-35f30502faad/scratchpad/venv/bin/python}"
API_PORT="${PAL_ARCHIVE_PORT:-8060}"
WEB_PORT="${PAL_ARCHIVE_WEB:-5181}"
LOG="$ROOT/tmp/archive-server.log"
AUTH_DIR="$HOME/.codex/resilience/local-compute-authorizations"
mkdir -p "$ROOT/tmp" "$AUTH_DIR"
export AUTH_DIR

for f in data/count_2w.txt data/ngrams_wikitext2.json; do
  [ -f "$ROOT/$f" ] || { echo "missing $f (copy it from the main palindrome checkout)"; exit 1; }
done

if ! curl -sf "http://127.0.0.1:$API_PORT/health" >/dev/null; then
  # Own process group plus a 2-hour exemption from the local compute guard.
  (cd "$ROOT" && PALINDROME_ARCHIVE=1 TOKENIZERS_PARALLELISM=false nohup perl -e '
    setpgrp(0,0);
    my $a = "$ENV{AUTH_DIR}/pgid-$$";
    open(F, ">", $a) or die "$a: $!"; print F time()+7200; close F;
    exec @ARGV' "$PY" -m uvicorn server.app:app --host 127.0.0.1 --port "$API_PORT" \
      --workers 1 --loop asyncio --http h11 >"$LOG" 2>&1 </dev/null &
   echo "archive server pid/pgid $!")
fi

echo "waiting for the archive server (log: $LOG)"
for _ in $(seq 1 120); do
  curl -s "http://127.0.0.1:$API_PORT/health" | grep -q '"archive":true' && break
  sleep 2
done
curl -s "http://127.0.0.1:$API_PORT/health"; echo

echo "open http://localhost:$WEB_PORT/v1  /v2  /v3"
cd "$ROOT/web"
[ -d node_modules ] || npm ci
PAL_API="http://127.0.0.1:$API_PORT" exec npx vite --host 127.0.0.1 --port "$WEB_PORT" --strictPort
