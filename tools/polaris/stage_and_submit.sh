#!/usr/bin/env bash
# Stage the repository to a configured remote checkout and submit a debug job.
#
# Requires a live multiplexed connection; ALCF authentication is
# keyboard-interactive with a one-time passcode, so establish it yourself
# first and this script will reuse the socket:
#
#     ssh -fN polaris
#     ./tools/polaris/stage_and_submit.sh search_debug.pbs
#
# Set PALINDROME_POLARIS_DEST to a writable remote checkout before running.
# Add your own scheduler account directive to the selected PBS file if needed.
set -euo pipefail

REMOTE=${PALINDROME_POLARIS_HOST:-polaris}
DEST=${PALINDROME_POLARIS_DEST:?Set PALINDROME_POLARIS_DEST to the remote checkout path.}
JOB=${1:-yield.pbs}

case "$JOB" in
  yield.pbs|scaling.pbs|search_debug.pbs|semantic_debug.pbs|diversity_debug.pbs|sentence_bank_debug.pbs|sentence_plan_debug.pbs|sentence_quality_debug.pbs) ;;
  *)
    echo "Usage: $0 [yield.pbs|scaling.pbs|search_debug.pbs|semantic_debug.pbs|diversity_debug.pbs|sentence_bank_debug.pbs|sentence_plan_debug.pbs|sentence_quality_debug.pbs]" >&2
    exit 2
    ;;
esac

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
  tools/polaris/shard_yield.py tools/polaris/search_debug.py \
  tools/polaris/semantic_debug.py tools/polaris/diversity_debug.py \
  tools/polaris/sentence_bank_debug.py tools/polaris/sentence_plan_debug.py \
  "tools/polaris/$JOB" \
  "$REMOTE:$DEST/tools/polaris/"
rsync -az tools/polaris/payload/ "$REMOTE:$DEST/tools/polaris/payload/"
if [[ "$JOB" == "semantic_debug.pbs" || "$JOB" == "diversity_debug.pbs" || "$JOB" == "sentence_bank_debug.pbs" || "$JOB" == "sentence_quality_debug.pbs" ]]; then
  rsync -az data/count_2w.txt "$REMOTE:$DEST/tools/polaris/payload/count_2w.txt"
fi

echo "checking the payload arrived intact"
ssh "$REMOTE" "cd $DEST && wc -l tools/polaris/payload/vocab30k.txt && \
  /usr/bin/python3.11 -c \"import gzip,json; d=json.load(gzip.open('tools/polaris/payload/brown.json.gz','rt')); \
  print('brown ok:', len(d['table']), 'words', len(d['shapes']), 'shapes')\""

echo "single-rank smoke test on the login node (30s, no allocation charged)"
if [[ "$JOB" == "sentence_quality_debug.pbs" ]]; then
  ssh "$REMOTE" "cd $DEST && PYTHONPATH=$DEST timeout 120 /usr/bin/python3.11 tools/polaris/sentence_plan_debug.py \
    --vocab 1200 --min-letters 12 --max-letters 24 --max-units 12 \
    --node-budget 100000 --seconds-per-arm 30 --arms planned_join0,planned_join1 \
    --shards 1 --out-dir /tmp/palsentencequalitysmoke"
elif [[ "$JOB" == "sentence_plan_debug.pbs" ]]; then
  ssh "$REMOTE" "cd $DEST && PYTHONPATH=$DEST timeout 120 /usr/bin/python3.11 tools/polaris/sentence_plan_debug.py \
    --vocab 1200 --min-letters 12 --max-letters 24 --max-units 12 \
    --node-budget 100000 --seconds-per-arm 30 --shards 1 --out-dir /tmp/palsentenceplansmoke"
elif [[ "$JOB" == "sentence_bank_debug.pbs" ]]; then
  ssh "$REMOTE" "cd $DEST && PYTHONPATH=$DEST timeout 120 /usr/bin/python3.11 tools/polaris/sentence_bank_debug.py \
    --vocab 1200 --min-letters 20 --beam 32 --per-parent 8 --opening-pool 128 \
    --candidate-limit 96 --max-steps 120 --seeds-per-rank 1 --shards 1 \
    --out-dir /tmp/palsentencebanksmoke"
elif [[ "$JOB" == "diversity_debug.pbs" ]]; then
  ssh "$REMOTE" "cd $DEST && PYTHONPATH=$DEST timeout 120 /usr/bin/python3.11 tools/polaris/diversity_debug.py \
    --vocab 1200 --min-letters 24 --beam 32 --per-parent 8 --opening-pool 128 \
    --candidate-limit 96 --max-steps 140 --seeds-per-rank 1 --shards 1 \
    --out-dir /tmp/paldiversitysmoke"
elif [[ "$JOB" == "semantic_debug.pbs" ]]; then
  ssh "$REMOTE" "cd $DEST && PYTHONPATH=$DEST timeout 120 /usr/bin/python3.11 tools/polaris/semantic_debug.py \
    --vocab 1200 --weights 0,0.5 --min-letters 30 --beam 24 --per-parent 8 \
    --candidate-limit 96 --max-steps 120 --seeds-per-rank 1 --shards 1 \
    --out-dir /tmp/palsemanticsmoke"
elif [[ "$JOB" == "search_debug.pbs" ]]; then
  ssh "$REMOTE" "cd $DEST && PYTHONPATH=$DEST timeout 120 /usr/bin/python3.11 tools/polaris/search_debug.py \
    --vocab 1200 --min-letters 30 --beam 24 --per-parent 8 --candidate-limit 96 \
    --max-steps 120 --seeds-per-rank 1 --shards 1 --out-dir /tmp/palsearchsmoke"
else
  ssh "$REMOTE" "cd $DEST && PYTHONPATH=$DEST timeout 120 /usr/bin/python3.11 tools/polaris/shard_yield.py \
    --vocab 1200 --lo 12 --hi 16 --seconds 30 --shards 1 --out-dir /tmp/palsmoke"
fi

echo "submitting"
JOBID=$(ssh "$REMOTE" "cd $DEST && qsub tools/polaris/$JOB")
echo "job: $JOBID"
