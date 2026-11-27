# AWS Qwen3 rubric audit: results

**Purpose:** a model-scored grammaticality and meaning diagnostic on the 52
unique v4 letter-level palindrome outputs. These scores are not human ratings,
a validated readability scale, or evidence that the generator improved prose.

## Result

Amazon Bedrock Qwen3 32B rated all 52 candidate strings `0/3` for both
grammaticality and coherent meaning. None of the 156 matched same-word
shuffles passed the joint threshold of at least 2 on both dimensions either.
The prespecified finite-set difference in joint-pass rates was therefore
exactly zero percentage points. The candidates were lower than the mean of
their three shuffles on grammar for 9/52 items and on meaning for 7/52; they
were never higher. This run shows no improvement signal under this rubric and
has a floor effect on the primary outcome.

The judge passed its narrow control screen: 11/12 held-out Brown prose spans
passed jointly, compared with 0/12 same-word shuffles. The gate was set before
inference at at least 8/12 prose controls and at most 3/12 shuffles. Seven of
the 12 Brown controls join two to four adjacent sentences; all seven passed,
as did 4/5 single-sentence controls. The control set is small and not
human-validated, so this is a model-specific sensitivity check, not validation.
Hidden duplicate presentations received identical two-score ratings in
11/12 pairs (all 6 repeated candidates agreed; 5/6 repeated shuffles agreed).

All 52 source strings are letter-level palindromes under the paper's
case-and-punctuation-insensitive ASCII-letter definition. None reverses word
order. This property is independently checked from the frozen candidate
packet; the model does not score palindrome validity.

## Frozen design and provenance

The run used `qwen.qwen3-32b-v1:0` on Amazon Bedrock in `us-east-2`, one text
per request, temperature 0, top-p 1, no provider seed, and a soft `/no_think`
prompt directive. The 244 presentations comprise 52 unique candidates, 156
matched shuffles, 12 Brown spans, 12 Brown shuffles, and 12 hidden repeats.
Condition labels and item IDs were not sent to the model. The exact rubric,
structured response schema, presentation order, response hashes, model/API
metadata, and analysis are saved in the run record. The Bedrock model ID and
endpoint details are documented by [AWS's Qwen3 32B model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-qwen-qwen3-32b.html).
We cannot verify whether Qwen3 saw Brown text during pretraining, and Bedrock
does not expose a model-weight digest.

The primary source files and hashes recorded by the run are:

| Artifact | SHA-256 |
| --- | --- |
| Candidate review packet | `efeb367df0eace41ce797ef0e7843537182abf44d6bdc74a88f2e09c2ffd1312` |
| Frozen static-verifier input | `9229cdfdf761dace9824686bd2f9ed285b30a12c798045995d6fa834e3130bf0` |
| Brown control manifest | `b572f86edf96ddf2b9e46f06b173f95dd0009c6540a91d58298af366a39d1f0f` |
| Scorer script | `3c62052cfac78ab40320b34e9778c7dcb4cee9d3431c677dfa5211c1373b48e5` |
| Frozen rubric/presentation script | `67a72cc85d7379bbf7ea61825097e746ecd4203eabb18277d2c7e328e03a8199` |
| AWS price snapshot | `551d955b80aa08cfd8edf3e25adbb2255f18b417af4a953e14dddd8c12f63069` |
| Redacted archived run record | `05a26714bd60e9294347d1e31f9d39394494dc3b5423374d812810529af3dd87` |

The saved record is the metadata-redacted
`runs/aws-qwen3-32b-readability-audit-20260926.json` member of
`paper/v4-reproducibility.zip`. The published copy removes transient AWS
request and runtime-identity metadata; presentations, saved model responses,
score fields, and analysis are preserved.
AWS's standard on-demand price snapshot records $0.000150/1K input tokens
and $0.000600/1K output tokens, effective 2026-09-01; the snapshot was
retrieved from the AWS Price List API and is retained alongside the scorer.
The 244 successful responses account for a known token-cost subtotal of
$0.01321965. Two earlier requests were interrupted, so their billing status
is unknown and the run correctly marks its total usage-based cost estimate
incomplete. Cumulative worst-case reservations for all 246 attempts were
$1.228032 under the run's $2 cap. No provisioned endpoint was used.

## Reproduction

At the original repository revision and with the pinned candidate/control
inputs, this checks the frozen presentation manifest without inference:

```sh
python3 experiments/aws_qwen3_readability_audit_20260926.py --prepare-only
python3 experiments/summarize_aws_qwen3_readability_audit_20260926.py \
  --input runs/aws-qwen3-32b-readability-audit-20260926.json --verify-sources
```

The summary command checks all 244 response-to-presentation IDs and text
hashes, validates scores against the 0--3 schema, recomputes the full analysis
from saved responses, and verifies the three input-file hashes. It makes no
model requests. A new Bedrock run requires authorized AWS credentials and
model access; Bedrock does not expose a request seed, so a fresh run is not
guaranteed to return identical judgments. The Brown corpus text is not copied
into the run record; the frozen manifest retains locations and hashes so the
licensed local corpus can reconstruct and verify each control.

For archive-only reanalysis, extract `paper/v4-reproducibility.zip` and run
the same summarizer from the extracted root without `--verify-sources`. The
public archive uses anonymized input projections, whose hashes differ from
the original private workspace files; the archive manifest and
`supplement/QWEN-EXPORT-PROVENANCE.json` verify those projected hashes. This
also makes no model requests.

## Interpretation boundary

The all-zero candidate ratings are a result about this model's responses to
this finite, success-conditioned set under this rubric. The zero joint-pass
difference is not positive evidence for a readable output, not a human
readability estimate, and not a population-level comparison. The control gate
does not repair the 7 multi-sentence examples or turn Brown spans into human
gold labels. The existing Qwen2.5-3B run is a separate model and failed its
own prespecified control gate; these results do not revise that run.
