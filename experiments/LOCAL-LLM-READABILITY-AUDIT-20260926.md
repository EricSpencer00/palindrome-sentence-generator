# Local-model language-quality audit (frozen protocol)

**Status:** protocol frozen before inference; results are written to `runs/local-llm-readability-audit-20260926.json`.

## Question

For the 52 unique v4 texts already selected by the search, does a blinded local model assign higher grammar and meaning scores than it assigns to deterministic permutations of the same words? The model judges language quality only. A separate mechanical check establishes letter-level palindrome validity.

This is a finite-set, success-conditioned model audit. It cannot establish that readers find the outputs readable, cannot estimate generation coverage, and cannot replace blind human ratings.

## Frozen inputs and exactness

- Candidates: the 52 unique texts in `paper/human_eval_v4_packet/sealed_key.json`, which are deduplicated from 66 selected rows in `runs/directional-search-resource-sensitivity-20260925/static-verifier-input-aws.json`.
- Before any model call, the script independently removes non-ASCII letters, lowercases, and checks exact reversal. It also counts token-order palindromes separately. All candidates pass the letter-level check; token-order status is reported separately.
- Candidate controls: three archived same-word shuffles per unique text, replicates 00, 01, and 02 from the frozen verifier input. The script verifies the word multiset and punctuation/whitespace template.
- To avoid capitalization leakage, every candidate, shuffle, and calibration text is lowercased and only its first alphabetic character is capitalized. Punctuation and whitespace are preserved.
- Positive calibration: 12 intact sentences reconstructed from the 99-file held-out Brown Corpus manifest in `runs/readability-length-stratified-20260925.json`, four per target-length band. They are selected by closest word count to hash-selected candidate targets, with deterministic SHA-256 tie-breaking and no reuse. One deterministic same-word shuffle is made for each positive control. No model score is used to choose any item.
- The Brown source sentences are not copied into the result artifact. The result stores their corpus file IDs, sentence indices, source manifest hash, and the hash of each exact model input. The Brown Corpus is marked for non-commercial use by its NLTK data package; follow that restriction when reproducing or distributing it.

## Model and request

- Model: installed Ollama `qwen2.5:3b` Q4_K_M checkpoint. The run records its full installed digest, size, model details, and Ollama version.
- The model receives one text per independent request. It does not receive condition labels, item IDs, pairing information, repeat markers, or the fact that the texts are palindrome outputs.
- Randomized presentation order uses seed `20260926`. Six candidate/control pairs, two from each length band, are silently repeated once each (12 repeated requests) to measure exact score agreement.
- Ollama structured output constrains JSON to two integer fields, each 0–3. Decoding is temperature 0, top-p 1, seed 20260926, and at most 128 generated tokens. The raw JSON response, exact prompt, prompt hash, model digest, and per-request token/time metadata are recorded.
- No AWS, hosted model API, or model download is used. Inference runs on the already-configured remote Ollama host through a temporary SSH port forward.

## Rubric and outcomes

The two scales are judged independently:

| Score | Grammaticality | Coherent meaning |
|---|---|---|
| 0 | No recoverable sentence parse / word salad | No recoverable proposition or event |
| 1 | Isolated grammatical phrases with major sentence-level breaks | Word associations without clear relations |
| 2 | Mostly recoverable parse, though strained or awkward | One plausible interpretation, though incomplete or vague |
| 3 | Natural, grammatical connected English | Clear meaning with understandable relations |

The prespecified joint pass is score **at least 2 on both scales**. For each of the 52 unique candidate texts, the primary finite-set contrast is its joint-pass indicator minus the mean joint-pass indicator across its three matched shuffles. Report the mean paired difference, candidate/control joint-pass counts, the candidate-by-shuffle-majority transition table, and separate score distributions. Do not use a population-level p-value or confidence interval for these dependent, construction-related outputs.

The judge calibration gate passes only if at least 8 of 12 held-out Brown sentences jointly score 2 or 3 and no more than 3 of their 12 same-word shuffles do. If it fails, candidate scores are inconclusive under this judge; do not retune the prompt against these results. Passing is a basic control check, not proof that the model agrees with readers. Brown exposure during Qwen pretraining is unknown.

## Reproduction

Use the same installed checkpoint and Brown/NLTK data. Start a temporary SSH local port forward from the coordinator to the machine that already runs Ollama:

```sh
ssh -N -L 127.0.0.1:11435:127.0.0.1:11434 <configured-ollama-host>
```

Then validate the frozen sample without inference and run or resume the scorer:

```sh
python3 experiments/local_llm_readability_audit_20260926.py --prepare-only
python3 experiments/local_llm_readability_audit_20260926.py --base-url http://127.0.0.1:11435
python3 experiments/local_llm_readability_audit_20260926.py --base-url http://127.0.0.1:11435 --resume
```

The second command writes an atomic checkpoint after every response. `--resume` verifies a fingerprint over the inputs, exact presented strings, prompt, schema, decoding parameters, installed model digest, and scorer source hash before continuing. To reproduce the analysis from saved data, recompute the analysis section with the frozen scorer after validating all recorded hashes.

## Sources

- Frozen candidate and shuffle source: the two repository JSON inputs listed above, whose SHA-256 values are embedded in the run result.
- Brown Corpus reference: W. N. Francis and H. Kučera, *Computational Analysis of Present-Day American English* (1967); the paper already cites it as `kucera-francis-1967`. NLTK's source package states that Brown may be used for non-commercial purposes: <https://github.com/nltk/nltk_data/blob/gh-pages/packages/corpora/brown.xml>.
- Model identity and quantization: official Ollama Qwen2.5 model page: <https://ollama.com/library/qwen2.5%3A3b-instruct>.
- Request/schema mechanism: official Ollama structured-output documentation: <https://docs.ollama.com/capabilities/structured-outputs>.
- Prior project calibration motivating absolute, not pairwise, local judgments: `experiments/RESULTS-llm-judge.md`; prior `gpt-oss:20b` results are separate from this Qwen run and are not pooled with it.
