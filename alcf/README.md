# Running on ALCF Polaris

## Do we need Argonne at all?

For the current pipeline, no. There is **no model training** in this project —
the palindrome constraint is enforced by search, and the language model is used
only for scoring, at inference. GPT-2 small runs the full benchmark in 5.5
seconds on an M-series Mac; GPT-2 large in about 2.5 minutes.

Argonne becomes worth it for one specific question: **does a much larger judge
produce more readable palindromes?** GPT-2 small and GPT-2 large already select
*different* candidates from the same pool, which suggests the judge's quality is
a real lever. Testing 7B-and-up judges with in-loop pruning — where the model is
called thousands of times per search — is what exceeds a laptop.

That is a debug-queue-sized job: one node, under an hour.

## Submitting

`polaris_debug.sh` needs your project allocation before it will run:

```bash
sed -i 's/REPLACE_WITH_PROJECT/<your_project>/' alcf/polaris_debug.sh
```

Then from a Polaris login node:

```bash
qsub alcf/polaris_debug.sh
```

Check status with `qstat -u $USER`.

## Note on the login step

Logging in requires an ALCF MobilePASS+ one-time password typed at the prompt.
OTPs are single-use and expire in seconds, so the `ssh` has to be run
interactively by you:

```bash
ssh <username>@polaris.alcf.anl.gov
```

## Caveat: compute nodes have no outbound network

Polaris compute nodes cannot reach the Hugging Face Hub. Pre-stage weights from
a login node before submitting:

```bash
export HF_HOME=/eagle/<project>/hf_cache
python -c "from transformers import AutoModelForCausalLM as M, AutoTokenizer as T; \
[ (M.from_pretrained(m), T.from_pretrained(m)) for m in ['gpt2','gpt2-large'] ]"
```

The job script sets the same `HF_HOME` so the cached weights are found offline.
