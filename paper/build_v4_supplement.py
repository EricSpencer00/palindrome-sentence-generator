"""Build a deterministic, self-contained v4 evidence and source archive."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "paper" / "v4-reproducibility.zip"

FILES = (
    "data/known_palindromes.json",
    "data/lexicon.txt",
    "experiments/__init__.py",
    "experiments/DIRECTIONAL-SEARCH-RESOURCE-SENSITIVITY-20260925.md",
    "experiments/DIRECTIONAL-SEARCH-BENCHMARK-20260925.md",
    "experiments/analyze_directional_search_resource_sensitivity_20260925.py",
    "experiments/analyze_paired_directional_search_benchmark_20260925.py",
    "experiments/audit_programmatic_readability.py",
    "experiments/infer_deepseek_static_verifier_20260925.py",
    "experiments/paired_directional_search_benchmark_20260925.py",
    "experiments/paired_directional_search_resource_sensitivity_20260925.py",
    "experiments/prepare_directional_static_verifier_20260925.py",
    "experiments/prepare_directional_static_verifier_aws_20260925.py",
    "experiments/score_directional_search_brown_20260925.py",
    "experiments/score_gpt2_static_verifier_20260925.py",
    "experiments/score_gpt2_static_verifier_aws_20260925.py",
    "experiments/score_gpt2_static_verifier_20260925.py",
    "experiments/score_length_stratified_readability.py",
    "experiments/static-verifier-bedrock-requirements.txt",
    "experiments/static-verifier-nuc-requirements.txt",
    "experiments/static_language_verifier_20260925.py",
    "experiments/summarize_directional_static_verifier_20260925.py",
    "llm_palindrome/__init__.py",
    "llm_palindrome/admission.py",
    "llm_palindrome/centerout.py",
    "llm_palindrome/generate.py",
    "llm_palindrome/lexicon.py",
    "llm_palindrome/overhang.py",
    "llm_palindrome/scoring.py",
    "llm_palindrome/search.py",
    "llm_palindrome/safe_vocab.py",
    "llm_palindrome/shortwords.py",
    "llm_palindrome/textify.py",
    "llm_palindrome/validator.py",
    "paper/acl.sty",
    "paper/acl_natbib.bst",
    "paper/build_v4_supplement.py",
    "paper/naacl2027.tex",
    "paper/readability-requirements.txt",
    "paper/refs.bib",
    "runs/directional-search-benchmark-20260925/confirmatory-v3-native-closure.json",
    "runs/directional-search-benchmark-20260925/manifest.json",
    "runs/directional-search-resource-sensitivity-20260925/brown-v4-aws.json",
    "runs/directional-search-resource-sensitivity-20260925/confirmatory-v4-resource-sensitivity.json",
    "runs/directional-search-resource-sensitivity-20260925/deepseek-v4.json",
    "runs/directional-search-resource-sensitivity-20260925/gpt2-v4-aws.json",
    "runs/directional-search-resource-sensitivity-20260925/manifest.json",
    "runs/directional-search-resource-sensitivity-20260925/static-verifier-input-aws.json",
    "runs/directional-search-resource-sensitivity-20260925/static-verifier-summary-v4-aws.json",
    "runs/directional-search-resource-sensitivity-20260925/summary-v4.json",
    "tests/test_initial_state_search.py",
    "tools/polaris/payload/vocab30k.txt",
)

README = """# v4 supplementary evidence and reproduction

This archive accompanies `paper/naacl2027.tex`. Extract it at a clean checkout
of the repository root, preserving its paths. It contains the frozen v4
manifest and raw results, v3 comparison used to motivate the adaptive follow-
up, all 66 selected texts and their 32 fixed shuffles, model outputs, source
code, vocabulary/lexicon inputs, and pinned dependency lists. `MANIFEST.sha256`
records each archived file hash.

## Search replay

Use Python 3.10+ and install the package versions recorded in the requirements
files. From the repository root, run:

```sh
python3 experiments/paired_directional_search_resource_sensitivity_20260925.py run \\
  --manifest runs/directional-search-resource-sensitivity-20260925/manifest.json \\
  --phase confirmatory --output /tmp/replayed-v4.json --workers 6
python3 experiments/analyze_directional_search_resource_sensitivity_20260925.py \\
  --manifest runs/directional-search-resource-sensitivity-20260925/manifest.json \\
  --results /tmp/replayed-v4.json \\
  --static-verifier-summary runs/directional-search-resource-sensitivity-20260925/static-verifier-summary-v4-aws.json \\
  --json-output /tmp/v4-summary.json --markdown-output /tmp/v4-summary.md
```

The search runner checks every source hash and the pinned `wordfreq` version
before it runs. Its random seeds, task order, vocabulary, gates, and resource
settings are frozen in the manifest. Execution time can vary by host; no run
in the reported experiment reached its cooperative deadline.

## Language diagnostics

Install `paper/readability-requirements.txt` and the Brown corpus to rerun the
Brown scorer; install `experiments/static-verifier-nuc-requirements.txt` and
use the pinned GPT-2 revision shown in the paper to rerun GPT-2. Example:

```sh
NLTK_DATA=/path/to/nltk_data python3 -m experiments.score_directional_search_brown_20260925 \\
  --input runs/directional-search-resource-sensitivity-20260925/static-verifier-input-aws.json \\
  --output /tmp/brown-v4.json
HF_HOME=/path/to/hf-cache OMP_NUM_THREADS=4 python3 -m experiments.score_gpt2_static_verifier_aws_20260925 \\
  --input runs/directional-search-resource-sensitivity-20260925/static-verifier-input-aws.json \\
  --output /tmp/gpt2-v4.json --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e \\
  --batch-size 8 --execution-host 'authorized CPU host'
python3 experiments/infer_deepseek_static_verifier_20260925.py \\
  --input runs/directional-search-resource-sensitivity-20260925/static-verifier-input-aws.json \\
  --output /tmp/deepseek-v4.json --budget-usd 0.20 --interval-seconds 0.75
python3 experiments/summarize_directional_static_verifier_20260925.py \\
  --input runs/directional-search-resource-sensitivity-20260925/static-verifier-input-aws.json \\
  --brown /tmp/brown-v4.json --gpt2 /tmp/gpt2-v4.json --deepseek /tmp/deepseek-v4.json \\
  --output /tmp/static-summary-v4.json
```

These scores are deterministic for the archived inputs and model/corpus
versions. A fresh
DeepSeek V3.2 request requires an authorized Bedrock account and
`experiments/static-verifier-bedrock-requirements.txt`; the archived requests
and responses can be re-parsed without a new call. To obtain new responses,
run the included cost-capped scorer with `--budget-usd 0.20`; Bedrock did not provide a
model seed, so a new response is not guaranteed to match. The reported call
used 11,538 input tokens and 1,410 output tokens. Its estimated cost was USD
0.00976206, computed from recorded usage and published [Bedrock rates](https://aws.amazon.com/bedrock/pricing/).

The outputs are automatic diagnostics, not readability labels. No human
ratings are in this archive.
"""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(2026, 9, 25, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def build(output: Path = DEFAULT_OUTPUT) -> dict:
    missing = [name for name in FILES if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError("supplement inputs missing: " + ", ".join(missing))

    content = {name: (ROOT / name).read_bytes() for name in FILES}
    content["supplement/README.md"] = README.encode("utf-8")
    hashes = "".join(
        f"{sha256(content[name])}  {name}\n" for name in sorted(content)
    ).encode("ascii")
    content["supplement/MANIFEST.sha256"] = hashes

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for name in sorted(content):
            archive.writestr(zip_info(name), content[name])
    return {
        "output": str(output),
        "member_count": len(content),
        "uncompressed_bytes": sum(map(len, content.values())),
        "sha256": sha256(output.read_bytes()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    import json
    print(json.dumps(build(args.output), indent=2))


if __name__ == "__main__":
    main()
