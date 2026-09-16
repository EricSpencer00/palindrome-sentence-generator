"""Role-aware reversible lexical reservoir for the center-out solver.

The preceding run had a correct centre-residual rule but a finite hand list.
This repair derives reversible word pairs from an independent frequency/POS
inventory, then injects each member into its attested grammatical role before
the same debt-carrying equation is solved.  It is not a replay of a fixed
semordnilap list or a beam-width sweep.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib, json, importlib.util
from pathlib import Path

from nltk.corpus import brown
from wordfreq import top_n_list, zipf_frequency

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "experiments" / "pos_template_centerout_longform_repair_20260915.py"
spec = importlib.util.spec_from_file_location("parent_centerout", PARENT)
assert spec and spec.loader
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)

EXPERIMENT_ID = "role-aware-reversible-reservoir-centerout-20260915"
SIGNATURE = "role-aware-reversible-reservoir-centerout|corpus-derived-semordnilap-pairs|palindromic-center-residual|typed-pos-template|independent-audit"
ROLES = {"NOUN", "VERB", "ADJ", "ADV", "ADP", "PRON", "DET"}


def reservoir(by: dict[str, list[str]]) -> tuple[dict[str, list[str]], dict]:
    counts: Counter[tuple[str, str]] = Counter()
    for sent in brown.tagged_sents(tagset="universal"):
        for raw, tag in sent:
            word = raw.casefold()
            if word.isascii() and word.isalpha() and tag in ROLES:
                counts[(word, tag)] += 1
    canonical: dict[str, str] = {}
    for (word, tag), count in counts.items():
        if word not in canonical or count > counts[(word, canonical[word])]:
            canonical[word] = tag
    words = {
        word for word in top_n_list("en", 160_000)
        if word.isascii() and word.isalpha() and len(word) >= 3
        and zipf_frequency(word, "en") >= 3.45 and word in canonical
    }
    pairs = []
    for word in words:
        reverse = word[::-1]
        if reverse in words and word < reverse:
            left_tag, right_tag = canonical.get(word), canonical.get(reverse)
            if left_tag in ROLES and right_tag in ROLES:
                pairs.append((word, reverse, left_tag, right_tag))
                by.setdefault(left_tag, []).append(word)
                by.setdefault(right_tag, []).append(reverse)
    for tag, vals in by.items():
        by[tag] = list(dict.fromkeys(vals))
    return by, {"eligible_pair_count": len(pairs), "pairs": [list(p) for p in pairs], "frequency_floor": 3.45}


def run() -> dict:
    by, reservoir_info = reservoir(parent._brown_lexicon())
    rows = []
    stats = Counter()
    for name, tags in parent.TEMPLATES.items():
        found, state = parent.solve(tags, by, budget=2_000_000)
        for row in found:
            row["template_name"] = name
        rows.extend(found)
        stats.update({f"{name}.{k}": v for k, v in state.items()})
    unique = {}
    for row in rows:
        tape = row["audit"]["normalized_tape"]
        if tape not in unique or row["lm_prior"] > unique[tape]["lm_prior"]:
            unique[tape] = row
    rendered = sorted(unique.values(), key=lambda r: (r["mechanically_admitted"], r["lm_prior"], r["audit"]["letters"]), reverse=True)
    admitted = [row for row in rendered if row["mechanically_admitted"]]
    return {
        "experiment_id": EXPERIMENT_ID,
        "signature": SIGNATURE,
        "status": "completed_no_reader_promotion" if not admitted else "exact_hits_pending_blinded_readers",
        "parent_repair": "pos-template-centerout-longform-center-residual-repair-20260915",
        "repair": "derive reversible lexical reservoir by independent frequency/POS attestation before center-out solving",
        "config": {"templates": {k: list(v) for k, v in parent.TEMPLATES.items()}, "inventory_sizes": {k: len(v) for k, v in by.items()}, "budget_per_template": 2_000_000, "catalogue_text_copied": False},
        "reservoir": reservoir_info,
        "stats": {**stats, "unique_terminal_rows": len(rendered), "mechanically_admitted": len(admitted), "reader_eligible": 0},
        "rendered_candidates": rendered[:100],
        "exact_candidates": admitted,
        "next_repair": "Use the strongest surviving role pair to build a two-clause tense/agreement transition, preserving independent character audits.",
        "provenance": {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "parent_script_sha256": hashlib.sha256(PARENT.read_bytes()).hexdigest(), "source": "Brown universal POS counts and wordfreq reservoir; no intact source sentences", "programmatic_readability_claim": False},
    }


if __name__ == "__main__":
    out = run()
    path = ROOT / "runs" / "role-aware-reversible-reservoir-centerout-20260915.json"
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"status": out["status"], "stats": out["stats"], "path": str(path)}, indent=2))
