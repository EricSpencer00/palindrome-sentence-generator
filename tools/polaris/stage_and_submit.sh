#!/usr/bin/env bash
# Stage the repository to Polaris home and submit the debug-queue job.
#
# Requires a live multiplexed connection; ALCF authentication is
# keyboard-interactive with a one-time passcode, so establish it yourself
# first and this script will reuse the socket:
#
#     ssh -fN polaris
#     ./tools/polaris/stage_and_submit.sh
#
# Storage is $HOME on Polaris by the account holder's instruction, which is
# also why the PBS script asks for filesystems=home only.
set -euo pipefail

REMOTE=polaris
DEST=${PALINDROME_REPO}

if ! ssh -O check "$REMOTE" >/dev/null 2>&1; then
  echo "No control socket for $REMOTE. Run:  ssh -fN $REMOTE" >&2
  exit 1
fi

echo "staging to $REMOTE:$DEST"
ssh "$REMOTE" "mkdir -p $DEST/logs $DEST/runs $DEST/tools/polaris/payload"

# Only what the job reads: the package, the driver, the frozen tables. No
# virtualenv, no runs/, no git history.
rsync -az --delete \
  --exclude '__pycache__' --exclude '*.pyc' \
  llm_palindrome/ "$REMOTE:$DEST/llm_palindrome/"
rsync -az --exclude '__pycache__' \
  tools/polaris/shard_yield.py tools/polaris/yield.pbs \
  "$REMOTE:$DEST/tools/polaris/"
rsync -az tools/polaris/payload/ "$REMOTE:$DEST/tools/polaris/payload/"

echo "checking the payload arrived intact"
ssh "$REMOTE" "cd $DEST && wc -l tools/polaris/payload/vocab30k.txt && \
  /usr/bin/python3.11 -c \"import gzip,json; d=json.load(gzip.open('tools/polaris/payload/brown.json.gz','rt')); \
  print('brown ok:', len(d['table']), 'words', len(d['shapes']), 'shapes')\""

echo "single-rank smoke test on the login node (30s, no allocation charged)"
ssh "$REMOTE" "cd $DEST && PYTHONPATH=$DEST timeout 120 /usr/bin/python3.11 tools/polaris/shard_yield.py \
  --vocab 1200 --lo 12 --hi 16 --seconds 30 --shards 1 --out-dir /tmp/palsmoke"

echo "submitting"
JOBID=$(ssh "$REMOTE" "cd $DEST && qsub tools/polaris/yield.pbs")
echo "job: $JOBID"
# Optional: inspect your own scheduler queue here.
