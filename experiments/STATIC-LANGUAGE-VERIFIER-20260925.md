# Static language verifier: 2026-09-25

## Purpose and limits

This run measures how two language models score a frozen set of eleven selected palindrome texts against fixed controls. It supplies a repeatable scoring protocol and auditable outputs for the NAACL draft. It is not a human readability study, a validated verifier, or a population-level comparison.

The project has documented a concrete weakness of GPT-2 likelihood as a ranker: its score has improved while blind judgments stayed flat or worsened, and character-normalized scores can reward long words. This run therefore uses mean log probability **per GPT-2 token**, reports the same-word controls and the matched prose values item by item, and does not use scores to select or rewrite examples.

## Execution plan

1. Freeze candidate strings, held-out prose offsets, exact text hashes, and same-word controls. **Complete.**
2. Run stock GPT-2 scoring on the configured NUC CPU with pinned packages and an immutable model revision. **Complete.**
3. Compare each candidate with its deterministic shuffle using DeepSeek V3.2 on Amazon Bedrock. **Complete: 11 calls, 9 candidate choices, 2 ties, no shuffle choices.**
4. Join row-level model outputs, test the score audit, add a sourced results table, rebuild the PDF, and inspect the rendered pages. **Complete.**

The separate R1 preflight is archived but excluded from the results: three calls used 789 input and 1,949 output tokens (estimated USD 0.01159); the longest request reached its output ceiling without a visible verdict.

## Frozen sources

- The eleven candidate strings and source-artifact lineage come from [`paper/week_results.json`](../paper/week_results.json), snapshot `2bdd7df301e185e67480f4cc3a0dbec43589b90b`. Each row is an explicitly selected exemplar, not a random sample.
- One intact Brown Corpus span per candidate is taken from the existing held-out report [`runs/readability-length-stratified-sentence-aware-20260925.json`](../runs/readability-length-stratified-sentence-aware-20260925.json). The source document ID, half-open sentence offsets, and SHA-256 are checked when the input set is rebuilt. The corpus source is Ku\v{c}era and Francis (1967); its NLTK loader is documented by Loper and Bird (2002).
- The frozen input contains each candidate and Brown span plus 32 deterministic word shuffles per item. Shuffles preserve the exact word multiset, each word's casing, punctuation, and whitespace. Brown text is materialized locally for scoring and is excluded from the committed artifacts; rebuilding fails if an expected Brown text hash changes.

## Scoring protocol

### GPT-2

Use `openai-community/gpt2` on the NUC CPU at immutable revision `607a30d783dfa663caf39e06633721c8d4cfcd7e` ([model repository snapshot](https://huggingface.co/openai-community/gpt2/tree/607a30d783dfa663caf39e06633721c8d4cfcd7e)). For token IDs $t_1,\ldots,t_N$ from the fast GPT-2 byte-level BPE tokenizer, report $\frac{1}{N-1}\sum_{i=2}^N \log p(t_i\mid t_1,\ldots,t_{i-1})$. The first token is not predicted; all remaining tokens are scored, with no truncation. EOS is used only as the masked padding token. The scorer fails if a passage exceeds GPT-2's context window. Each candidate and matched Brown span is compared with its own 32 deterministic shuffles. We report per-token differences, token counts, and all row-level values; no score is divided by letters or characters.

GPT-2 is a next-token likelihood model, not a reader. This metric is evidence of fit to its learned distribution, not a direct score of syntax, meaning, coherence, or human preference \citep{radford2019language}.

### DeepSeek V3.2

Amazon Bedrock model ID `deepseek.v3.2` receives one counterbalanced pair per candidate: the candidate and its first frozen same-word shuffle. Left/right order is assigned by a stable SHA-256 rule. The prompt asks which reads as better ordinary English and requests a brief observable feature. It does not identify candidate/control roles. Requests use temperature 0, top-$p$ 1, and a 256-token response ceiling. AWS documents the supported model ID, Converse API, regional availability, and model limit \citep{awsbedrockdeepseekv32}.

The provider does not expose a seed that guarantees identical outputs. The prompt, settings, input IDs, side assignment, visible response, parsed choice, model request ID, and reported token use are saved so the exact run can be audited; repeating it makes the same requests but may produce different answers.

The account's in-region quota was read from AWS Service Quotas on 2026-09-25: DeepSeek V3.2 allows 100 requests per minute. The runner retains resumable row-level output. AWS lists V3.2 at USD 0.62 per million input tokens and USD 1.85 per million output tokens; the exact observed 6,327 input and 217 output tokens cost an estimated USD 0.00432419. The R1 preflight used USD 0.01158975 at its published example rates. Combined estimated Bedrock spend was USD 0.01591394. Sources: [AWS DeepSeek V3.2 model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-deepseek-deepseek-v3-2.html), [Amazon Bedrock pricing](https://aws.amazon.com/bedrock/pricing/), and [AWS DeepSeek R1 parameters](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-deepseek.html), checked 2026-09-25.

## Reproduction

From the repository root, install the versions in `paper/readability-requirements.txt` and download NLTK's Brown corpus if it is not already available. Then prepare the local frozen input:

```bash
python3 -m pip install -r paper/readability-requirements.txt
python3 -m nltk.downloader brown
python3 experiments/score_length_stratified_readability.py \
  --manifest paper/week_results.json \
  --expected-report runs/readability-length-stratified-sentence-aware-20260925.json \
  --output /tmp/readability-length-stratified-rerun.json
python3 experiments/static_language_verifier_20260925.py prepare
```

The expected-report check recomputes the Brown corpus stream, selected spans, and saved scores. Brown passages are not redistributed; preparation rebuilds them from the NLTK corpus using the source document IDs, sentence offsets, and SHA-256 checks.

On the NUC, copy the frozen input, scorer, and pinned requirements file, then install and run the CPU-only scorer. Set `NUC_HOST` to the SSH alias or address configured for the NUC:

```bash
NUC_HOST=your-configured-nuc-host
ssh "$NUC_HOST" 'mkdir -p /tmp/static-language-verifier-20260925'
scp runs/static-language-verifier-20260925/input.json \
  experiments/score_gpt2_static_verifier_20260925.py \
  experiments/static-verifier-nuc-requirements.txt \
  "$NUC_HOST":/tmp/static-language-verifier-20260925/
ssh "$NUC_HOST" 'python3 -m venv /tmp/static-language-verifier-20260925/venv'
ssh "$NUC_HOST" '/tmp/static-language-verifier-20260925/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0'
ssh "$NUC_HOST" '/tmp/static-language-verifier-20260925/venv/bin/pip install -r /tmp/static-language-verifier-20260925/static-verifier-nuc-requirements.txt'
ssh "$NUC_HOST" 'HF_HOME=/tmp/static-language-verifier-20260925/hf-home OMP_NUM_THREADS=8 /tmp/static-language-verifier-20260925/venv/bin/python /tmp/static-language-verifier-20260925/score_gpt2_static_verifier_20260925.py --input /tmp/static-language-verifier-20260925/input.json --output /tmp/static-language-verifier-20260925/gpt2.json --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e --batch-size 8'
scp "$NUC_HOST":/tmp/static-language-verifier-20260925/gpt2.json \
  runs/static-language-verifier-20260925/gpt2.json
```

To repeat the Bedrock comparison, install `experiments/static-verifier-bedrock-requirements.txt` in a Python environment that can use the same AWS account login, then run:

```bash
python3 experiments/infer_deepseek_static_verifier_20260925.py \
  --input runs/static-language-verifier-20260925/input.json \
  --output runs/static-language-verifier-20260925/deepseek.json \
  --budget-usd 0.02 --interval-seconds 0.25
```

The default preflight ceiling counts every UTF-8 input byte as an input token and reserves the full 256-token output allowance at the published V3.2 rates. For this frozen input the bound is USD 0.01922532, below the explicit USD 0.02 budget. The runner aborts before inference if the bound exceeds the budget. Exact input/output token counts and the resulting cost estimate are captured from the Bedrock responses. Repeating the requests uses the same prompts and side assignments, but AWS does not expose a seed that guarantees identical model answers.

After retrieving the NUC score file, assemble the row-level report:

```bash
python3 experiments/static_language_verifier_20260925.py summarize
python3 paper/build_static_verifier_table.py
python3 -m unittest tests.test_static_language_verifier_20260925 -v
```

Build the review PDF from `paper/` with `tectonic --outdir ../output/pdf/naacl-ready-review --keep-logs --keep-intermediates naacl2027.tex`, then check `pdfinfo`, confirm all fonts are embedded with `pdffonts`, render with Poppler, and inspect each page. The current PDF is A4, uses the ACL review style, and has three content pages. NAACL 2027 and ARR allow up to four pages for a short paper; limitations, ethics, and references are excluded from that count. See the [NAACL 2027 main-conference call](https://2027.naacl.org/calls/main_conference_papers/), [ARR call](https://aclrollingreview.org/cfp), and [ACLPUB formatting requirements](https://acl-org.github.io/ACLPUB/formatting.html), checked 2026-09-25.

## Outputs

- `runs/static-language-verifier-20260925/gpt2.json`: one untruncated token-likelihood row for each frozen candidate, prose control, and shuffle.
- `runs/static-language-verifier-20260925/deepseek.json`: 11 blinded pairwise outputs, prompt and settings, request IDs, and token use.
- `runs/static-language-verifier-20260925/report.json`: paired candidate/prose/shuffle results with limitations attached.
- `paper/static_verifier_table.tex`: generated paper table with all eleven GPT-2 scores and paired differences plus the V3.2 choices.
- `paper/static-verifier-supplement.zip`: deterministic, hash-manifested supplementary package with sources and row-level reports; the builder scans for identity markers and excludes the reconstructed Brown text input.
- `runs/static-language-verifier-20260925/input.json` is generated locally and ignored by Git because it includes Brown corpus passages. It is reproducible from the corpus IDs, sentence offsets, and hashes in the committed source reports.

The result is 10/11 candidates below their matched Brown text under GPT-2 mean token log probability (mean paired difference -0.995 nats/token); all eleven score above their own 32-shuffle mean. DeepSeek V3.2 selected the candidate in 9/11 pairs and tied in two. These model outputs do not fill the missing human-rating evidence.
