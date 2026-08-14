#!/bin/bash -l
#PBS -N palindrome_scale
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:55:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -A REPLACE_WITH_PROJECT
#PBS -j oe

# Scaling study: does a larger fluency judge produce more readable palindromes?
# The debug queue caps at 1 hour, so walltime is 55 min with margin.
# Search is CPU-bound and single-threaded per seed; the GPU serves the LM only.

set -euo pipefail
cd "${PBS_O_WORKDIR:-$HOME/palindrome-sentence-generator}"

module use /soft/modulefiles
module load conda
conda activate base

export HF_HOME=/eagle/"${PROJECT:-$USER}"/hf_cache   # login nodes have no outbound HF access
export TOKENIZERS_PARALLELISM=false

python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.device_count())"

for MODEL in gpt2 gpt2-large EleutherAI/gpt-neo-1.3B; do
  echo "=== judge: $MODEL ==="
  python benchmark.py \
    --seeds 24 --min-letters 300 --beam 80 \
    --out "results_$(basename "$MODEL").json" \
    2>&1 | grep -v "Loading weights"
done

echo "done: $(date)"
