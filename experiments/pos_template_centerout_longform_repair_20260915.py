"""Long-form POS-template center-out lexicalization.

Unlike the fixed-slot clause products, this route uses a 14--16 word POS
template and carries an unmatched character *debt* across word boundaries
while lexicalizing from both outer edges.  Word boundaries are therefore
variables in the equation.  The inventories are fresh Brown/word-frequency
lexical items; no known palindrome text is supplied to the constructor.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib, json, math, re, sys
from pathlib import Path

from nltk.corpus import brown
from wordfreq import top_n_list, zipf_frequency

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

EXPERIMENT_ID = "pos-template-centerout-longform-repair-20260915"
SIGNATURE = "pos-template-centerout-longform-repair|debt-carrying-character-equation|variable-word-boundary|fresh-lexicalization|independent-tape-audit"

TEMPLATES = {
    "long_report": ("NOUN","NOUN","PRON","VERB","DET","ADJ","ADV","VERB","DET","NOUN","PRON","VERB","ADP","NOUN"),
    "long_report_adj": ("NOUN","NOUN","PRON","VERB","DET","ADJ","ADV","VERB","DET","ADJ","NOUN","PRON","VERB","ADP","NOUN","NOUN"),
    "long_result": ("NOUN","VERB","PRON","VERB","DET","NOUN","ADV","VERB","DET","NOUN","PRON","VERB","ADP","NOUN"),
}

FUNCTIONS = set("a an the i we he she it they you me us them my our your this that and or but if as of to in on at by for with from is was are were be been have has had do does did can could will would may might should not no never".split())
EXPLICIT = {
    "NOUN": "cod dog god pot top rat tar deer reed drawer reward mood doom mail rail map pan nap note stone room moor wolf flow pool loop star rats parts strap pets step nuts stun straw warts denim mined lever revel trap part bats stab pals slap gum mug bed tab bat gas sag cap pac liar rail spam maps tips spit tops spot saw was man nam son nos name eman time emit evil live".split(),
    "VERB": "saw was live emit draw deliver stressed repaid keep peek stop part trap edit dial map ban nab spot show use open close call help visit want like work play run walk talk look know tell ask give get put set let eat win meet read write make take find send hold need love".split(),
    "PRON": "i we he she it they you me us them".split(),
    "DET": "a an the".split(),
    "ADJ": "big red old new small quiet young green bright kind smart raw evil avid fast deep dark calm warm cool clear true open final local major minor".split(),
    "ADV": "now never soon away here there often well again today later just very still".split(),
    "ADP": "in on at by to for with from near over under of".split(),
}


def _brown_lexicon() -> dict[str, list[str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for sent in brown.tagged_sents(tagset="universal"):
        for raw, tag in sent:
            w = raw.casefold()
            if w.isascii() and w.isalpha():
                counts[(w, tag)] += 1
    canon: dict[str, str] = {}
    for (w, tag), count in counts.items():
        if w not in canon or count > counts[(w, canon[w])]:
            canon[w] = tag
    common = {
        w.casefold() for w in top_n_list("en", 100_000)
        if w.isascii() and w.isalpha() and (len(w) >= 3 or w in FUNCTIONS)
    }
    out: dict[str, list[str]] = defaultdict(list)
    for w in common:
        tag = canon.get(w)
        if tag in TEMPLATES["long_report"] and (w in FUNCTIONS or zipf_frequency(w, "en") >= 3.35):
            out[tag].append(w)
    for tag, vals in EXPLICIT.items():
        out[tag].extend(vals)
    # Keep a reproducible, diverse inventory: high-frequency words first, then
    # explicit reversibles that are otherwise rare but grammatical.
    result: dict[str, list[str]] = {}
    for tag in set(x for t in TEMPLATES.values() for x in t):
        vals = {w for w in out[tag] if w.isascii() and w.isalpha() and w != w[::-1]}
        vals = sorted(vals, key=lambda w: (-zipf_frequency(w, "en"), w))
        anchors = [w for w in EXPLICIT.get(tag, ()) if w in vals]
        result[tag] = list(dict.fromkeys(anchors + vals))[:180]
    return result


def _content_unique(words: tuple[str, ...]) -> bool:
    content = [w for w in words if w not in FUNCTIONS and len(w) > 2]
    return len(content) == len(set(content))


def _audit(text: str) -> dict:
    tape = normalize_letters(text)
    independent = "".join(re.findall(r"[a-z]", text.casefold()))
    return {
        "rendered": text,
        "letters": len(tape),
        "normalized_tape": tape,
        "independent_ascii_tape": independent,
        "exact": bool(tape) and tape == tape[::-1],
        "independent_exact": bool(independent) and independent == independent[::-1],
        "tapes_equal": tape == independent,
        "sha256": hashlib.sha256(tape.encode()).hexdigest(),
    }


def _lm_prior(words: tuple[str, ...]) -> float:
    # Diagnostic ranking only; it never certifies readability.
    return sum(zipf_frequency(w, "en") for w in words) - 0.8 * len(words)


def solve(tags: tuple[str, ...], by: dict[str, list[str]], budget: int = 1_200_000) -> tuple[list[dict], dict]:
    n = len(tags)
    # owner 0 = no debt; +1 = leftover left word; -1 = leftover right word.
    stack = [(0, n - 1, "", 0, (), ())]
    rows: list[dict] = []
    stats = Counter()
    right_by_first: dict[str, list[str]] = defaultdict(list)
    for w in by[tags[-1]]:
        right_by_first[w[-1]].append(w)
    while stack and stats["states"] < budget:
        i, j, residual, owner, left, right_rev = stack.pop()
        stats["states"] += 1
        if i > j:
            stats["terminal"] += 1
            # A debt may terminate at the character centre when the final
            # unmatched segment is itself a short palindrome (the common case
            # is one central letter).  Reject only a non-palindromic residual;
            # the previous implementation incorrectly discarded valid odd-
            # length tapes after all word slots had been consumed.
            if (residual and residual != residual[::-1]) or len(normalize_letters("".join(left + tuple(reversed(right_rev))))) < 39:
                stats["dead_terminal"] += 1
                continue
            words = left + tuple(reversed(right_rev))
            if not _content_unique(words):
                stats["repeated_content_reject"] += 1
                continue
            text = " ".join(words).capitalize() + "."
            a = _audit(text)
            checks = mechanical_admission_checks(text, min_letters=39, max_letters=240)
            row = {
                "rendered": text,
                "words": list(words),
                "audit": a,
                "mechanical_checks": checks,
                "mechanically_admitted": a["independent_exact"] and all(checks.values()),
                "lm_prior": _lm_prior(words),
                "template": list(tags),
                "reader_status": "not_run; programmatic diagnostics cannot certify readability",
            }
            rows.append(row)
            stats["exact_terminal"] += int(a["exact"])
            continue
        if owner == 0:
            lt, rt = tags[i], tags[j]
            groups: dict[str, list[str]] = defaultdict(list)
            for v in by[rt]:
                groups[v[-1]].append(v)
            for w in by[lt]:
                if w not in FUNCTIONS and w in left + right_rev:
                    continue
                for v in groups.get(w[0], ()):
                    if v not in FUNCTIONS and v in left + right_rev:
                        continue
                    e = v[::-1]
                    if w.startswith(e):
                        rem, new_owner = w[len(e):], 1 if len(w) > len(e) else 0
                    elif e.startswith(w):
                        rem, new_owner = e[len(w):], -1 if len(e) > len(w) else 0
                    else:
                        continue
                    stack.append((i + 1, j - 1, rem, new_owner, left + (w,), right_rev + (v,)))
        elif owner == 1:
            # Left word still has `residual`; consume it with the next right
            # word emitted from the outer edge.
            rt = tags[j]
            for v in by[rt]:
                if v not in FUNCTIONS and v in left + right_rev:
                    continue
                e = v[::-1]
                if residual.startswith(e):
                    rem, new_owner = residual[len(e):], 1 if len(residual) > len(e) else 0
                elif e.startswith(residual):
                    rem, new_owner = e[len(residual):], -1 if len(e) > len(residual) else 0
                else:
                    continue
                stack.append((i, j - 1, rem, new_owner, left, right_rev + (v,)))
        else:
            lt = tags[i]
            for w in by[lt]:
                if w not in FUNCTIONS and w in left + right_rev:
                    continue
                if residual.startswith(w):
                    rem, new_owner = residual[len(w):], -1 if len(residual) > len(w) else 0
                elif w.startswith(residual):
                    rem, new_owner = w[len(residual):], 1 if len(w) > len(residual) else 0
                else:
                    continue
                stack.append((i + 1, j, rem, new_owner, left + (w,), right_rev))
    return rows, dict(stats)


def run() -> dict:
    by = _brown_lexicon()
    all_rows: list[dict] = []
    stats = Counter()
    for name, tags in TEMPLATES.items():
        rows, st = solve(tags, by)
        for row in rows:
            row["template_name"] = name
        all_rows.extend(rows)
        stats.update({f"{name}.{k}": v for k, v in st.items()})
    # Deduplicate by tape while retaining the best diagnostic rendering.
    unique: dict[str, dict] = {}
    for row in all_rows:
        key = row["audit"]["normalized_tape"]
        if key not in unique or row["lm_prior"] > unique[key]["lm_prior"]:
            unique[key] = row
    rows = sorted(unique.values(), key=lambda r: (r["mechanically_admitted"], r["lm_prior"], r["audit"]["letters"]), reverse=True)
    admitted = [r for r in rows if r["mechanically_admitted"]]
    return {
        "experiment_id": EXPERIMENT_ID,
        "signature": SIGNATURE,
        "status": "completed_no_reader_promotion" if not admitted else "exact_hits_pending_blinded_readers",
        "config": {"templates": {k: list(v) for k, v in TEMPLATES.items()}, "inventory_sizes": {k: len(v) for k, v in by.items()}, "budget_per_template": 1_200_000, "debt_carrying_center_out": True, "catalogue_text_copied": False},
        "stats": {**stats, "unique_terminal_rows": len(rows), "mechanically_admitted": len(admitted), "reader_eligible": 0},
        "rendered_candidates": rows[:80],
        "exact_candidates": admitted,
        "next_repair": "Hold the highest-scoring seam fixed and replace only its typed lexical item with a held-out synonym, preserving debt state and independent audit.",
        "provenance": {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "source": "Brown POS counts plus explicit ordinary lexical anchors; no intact source sentences", "programmatic_readability_claim": False},
    }


if __name__ == "__main__":
    out = run()
    path = ROOT / "runs" / "pos-template-centerout-longform-repair-20260915.json"
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"status": out["status"], "stats": out["stats"], "path": str(path)}, indent=2))
