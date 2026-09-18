# Quarantined historical reproducibility archive

This archive description is historical. It does not document a current
readable-palindrome result and must not be used to build or distribute a release.

This archive accompanies *Measuring Reversal Cost in English for Exact
Palindrome Search*. It contains the frozen samples and measurements behind the
paper, the current source needed to audit or rerun them, the two long-form
examples discussed in the paper, and secondary evidence for the exact decoder
case study.

Paths below are relative to the extracted archive. Do not use Python's `-O`
flag: the audit programs intentionally use assertions.

## Audit the primary result

Install `wordfreq` (the archive's `requirements.txt` records the full rerun
environment), then run:

```sh
python3 -m experiments.audit_mirror_cost \
  data/mirror-cost/results.json \
  --output mirror-cost-audit.json
```

The audit checks all 36 model × segmenter × target-length cells, every saved
span and normalized letter stream, the frozen vocabulary hash and filtered
size, both segmentations, lexical coverage, paired standard errors, and all
reported arithmetic. It does not load model weights or recompute logits.

The result uses 150 saved WikiText-2 spans at each target length (20, 30, 40,
60, 80, and 120 letters), three segmentation strategies, GPT-2, and
SmolLM2-135M. Reversal adds 2.14–3.34 bits per letter across the 36 conditions.
Forward lexical coverage is 86.9–91.5%; reversed coverage is 49.6–54.6%.
Within-cell paired standard errors are 0.04–0.11.

The exact samples, corpus snapshot and parquet hash, vocabulary hash, model
commit hashes, boundary-token choices, and package versions are embedded in
`data/mirror-cost/results.json`. `data/mirror-cost/RESULTS.md` is a human-readable
summary.

## Recompute model scores

With the packages in `requirements.txt` and access to the two model revisions:

```sh
python3 experiments/mirror_cost.py \
  --models gpt2 HuggingFaceTB/SmolLM2-135M \
  --strategies unigram fewest greedy \
  --spans 20 30 40 60 80 120 --n 150 \
  --seed 20260911 --batch-size 16 \
  --out rerun/results.json
```

The program expects the WikiText-2 parquet file in the ordinary Hugging Face
cache and verifies its recorded hash in the new output. Downloading models or
the corpus is intentionally not part of the offline audit.

## Audit the long-form examples

```sh
python3 -m experiments.audit_long_form_examples \
  data/long-form/examples.json --root . \
  --output long-form-audit.json
```

This standard-library audit verifies the source-data hashes, mirror relation
for every paired unit, whole-text exactness, word and letter counts, and
generated-versus-catalogue provenance. The generated example has 101 words,
342 letters, and 24 generated mirror pairs. The catalogue example has 72
words, 237 letters, nine borrowed pairs, and one borrowed centre. The audit
does not assess readability.

## Audit the decoder case study

```sh
python3 paper/verify_structural_draft.py --output structural-evidence.json
python3 paper/critique_evidence.py --output critique-evidence.json
python3 -m experiments.audit_controlled_pos_pruning \
  data/controlled --output controlled-audit.json
```

The controlled audit checks 50 fixed-opening paired trials and 2,544 accepted
pair rows. Terminal-only filtering returns 407 accepted pairs; incremental
POS-shape filtering returns 2,137. The paired rate ratio is 5.47 per popped
state (bootstrap 95% interval 2.89–17.65), 4.36 per CPU second (2.30–13.93),
and 0.95 per generated state (0.46–3.20). These are structural search-yield
measurements, not language-quality measurements.

The structural checker also verifies the 90,937-letter dictionary artifact,
including its exact palindrome property, phrase inventory, repetition limits,
and comparison to the identically parsed Norvig reference.

## Archive map

- `data/mirror-cost/`: primary frozen result, audit, and readable summary.
- `data/long-form/`: frozen generated and catalogue examples plus audit.
- `data/controlled/`: fixed-work POS-pruning trials and reports.
- `data/length/`: 90,937-letter dictionary artifact and audit.
- `data/structural/` and `data/inventory/`: secondary exploratory results.
- `inputs/`: frozen vocabulary, Brown payload, long-form banks, and Norvig inputs.
- `experiments/`, `paper/`, `llm_palindrome/`, and `server/`: relevant source.
- `requirements.txt`: packages required for the full model rerun.
- `SOURCE-SNAPSHOT.json`: hashes of included source plus checkout provenance.
- `MANIFEST-SHA256.json`: hashes of every archive payload.
- `TERMS.md` and `licenses/`: attribution and upstream terms.

The archive supports exactness, provenance, likelihood, lexical coverage, and
defined structural-search claims. It contains no human evaluation and supports
no positive claim that the generated long output is grammatical, coherent, or
meaningful.
