"""Asymmetric grammar-product repair for long-form centre-out search.

The variable-length reservoir run still solved one POS shape against itself.
This successor makes the grammar state a product of *different* left and
right templates and carries a small agreement feature alongside the character
debt.  The two sides are lexicalized independently; the right template is not
constructed by reversing a left phrase.  Exactness is checked again from the
rendered text and the shared admission gate remains fail-closed.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "experiments" / "role_aware_reversible_reservoir_centerout_20260915.py"
spec = importlib.util.spec_from_file_location("reservoir", PARENT)
assert spec and spec.loader
reservoir = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reservoir)

EXPERIMENT_ID = "asymmetric-template-reservoir-centerout-20260915"
SIGNATURE = (
    "asymmetric-template-reservoir-centerout|paired-grammar-state-product|"
    "agreement-bearing-roles|palindromic-centre-residual|independent-audit"
)

# Entries are in ordinary reading order.  The solver consumes right_tags from
# the end, so its outer-to-inner right edge is still independent of the left.
PAIRS = {
    "report_event": {
        "left": ("DET", "NOUN", "VERB", "DET", "NOUN", "ADV"),
        "right": ("PRON", "VERB", "DET", "ADJ", "NOUN"),
        "agreement": "subject-number",
    },
    "descriptive_reply": {
        "left": ("DET", "ADJ", "NOUN", "VERB", "DET", "NOUN"),
        "right": ("PRON", "VERB", "ADP", "DET", "NOUN"),
        "agreement": "subject-number",
    },
    "locative_report": {
        "left": ("NOUN", "VERB", "DET", "NOUN", "ADP", "DET", "NOUN"),
        "right": ("PRON", "VERB", "DET", "NOUN", "ADV"),
        "agreement": "subject-number",
    },
    "causal_event": {
        "left": ("DET", "NOUN", "VERB", "DET", "ADJ", "NOUN", "ADV"),
        "right": ("PRON", "VERB", "DET", "NOUN", "ADP", "NOUN"),
        "agreement": "subject-number",
    },
}

FUNCTIONS = reservoir.parent.FUNCTIONS


def _word_number(word: str) -> str:
    """Use a conservative surface cue for the agreement state."""
    if word in {"i", "we", "you", "they", "them", "us"} or word.endswith("s"):
        return "plural"
    return "singular"


def _content_unique(words: tuple[str, ...]) -> bool:
    content = [w for w in words if w not in FUNCTIONS and len(w) > 2]
    return len(content) == len(set(content))


def _agreement_ok(words: tuple[str, ...], left_tags: tuple[str, ...], right_tags: tuple[str, ...]) -> bool:
    """Apply a conservative subject/finite-verb agreement check per side."""
    left_words = words[: len(left_tags)]
    right_words = words[len(left_tags):]
    for side_words, side_tags in ((left_words, left_tags), (right_words, right_tags)):
        try:
            subject = next((w for w, tag in zip(side_words, side_tags) if tag in {"NOUN", "PRON"}), None)
            subject_index = next(i for i, tag in enumerate(side_tags) if tag in {"NOUN", "PRON"})
            verb = next((w for w, tag in zip(side_words[subject_index + 1:], side_tags[subject_index + 1:]) if tag == "VERB"), None)
        except StopIteration:
            continue
        if not subject or not verb:
            continue
        plural_subject = subject in {"we", "they", "you", "i", "men", "people"} or subject.endswith("s")
        singular_aux = {"is", "was", "has", "does"}
        plural_aux = {"are", "were", "have", "do"}
        if plural_subject and verb in singular_aux:
            return False
        if not plural_subject and verb in plural_aux:
            return False
    return True


def _audit(text: str) -> dict:
    tape = reservoir.parent.normalize_letters(text)
    independent = "".join(ch for ch in text.casefold() if "a" <= ch <= "z")
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


def solve(
    left_tags: tuple[str, ...],
    right_tags: tuple[str, ...],
    by: dict[str, list[str]],
    *,
    budget: int = 600_000,
) -> tuple[list[dict], dict, list[dict]]:
    """Enumerate an asymmetric pair while carrying character and agreement debt."""
    stack = [(0, len(right_tags) - 1, "", 0, "", (), ())]
    rows: list[dict] = []
    probes: list[dict] = []
    stats = Counter()
    while stack and stats["states"] < budget:
        i, j, residual, owner, left_number, left, right_rev = stack.pop()
        stats["states"] += 1
        if len(probes) < 40 and (left or right_rev):
            partial_words = left + tuple(reversed(right_rev))
            partial_text = " ".join(partial_words).capitalize() + "."
            partial_tape = reservoir.parent.normalize_letters("".join(partial_words))
            probes.append({
                "rendered": partial_text,
                "letters": len(partial_tape),
                "normalized_tape": partial_tape,
                "residual": residual,
                "owner": owner,
                "left_slots_consumed": i,
                "right_slots_remaining": j + 1,
                "independent_exact": bool(partial_tape) and partial_tape == partial_tape[::-1],
            })
        if i >= len(left_tags) and j < 0:
            stats["terminal"] += 1
            if residual and residual != residual[::-1]:
                stats["dead_terminal"] += 1
                continue
            words = left + tuple(reversed(right_rev))
            if len(reservoir.parent.normalize_letters("".join(words))) < 39:
                stats["short_terminal"] += 1
                continue
            if not _content_unique(words):
                stats["repeated_content_reject"] += 1
                continue
            if not _agreement_ok(words, left_tags, right_tags):
                stats["agreement_reject"] += 1
                continue
            text = " ".join(words).capitalize() + "."
            checks = reservoir.parent.mechanical_admission_checks(text, min_letters=39, max_letters=240)
            audit = _audit(text)
            row = {
                "rendered": text,
                "words": list(words),
                "left_tags": list(left_tags),
                "right_tags": list(right_tags),
                "audit": audit,
                "mechanical_checks": checks,
                "mechanically_admitted": audit["independent_exact"] and all(checks.values()),
                "lm_prior": reservoir.parent._lm_prior(words),
                "reader_status": "not_run; programmatic diagnostics cannot certify readability",
            }
            rows.append(row)
            stats["exact_terminal"] += int(audit["exact"])
            stats["mechanically_admitted"] += int(row["mechanically_admitted"])
            continue

        def allowed(word: str) -> bool:
            return word in FUNCTIONS or (word not in left and word not in right_rev)

        if owner == 0:
            if i >= len(left_tags) or j < 0:
                stats["shape_mismatch"] += 1
                continue
            lt, rt = left_tags[i], right_tags[j]
            right_by_first: dict[str, list[str]] = defaultdict(list)
            for v in by.get(rt, ()):
                if allowed(v):
                    right_by_first[v[-1]].append(v)
            for w in by.get(lt, ()):
                if not allowed(w):
                    continue
                for v in right_by_first.get(w[0], ()):
                    e = v[::-1]
                    if w.startswith(e):
                        rem, new_owner = w[len(e):], 1 if len(w) > len(e) else 0
                    elif e.startswith(w):
                        rem, new_owner = e[len(w):], -1 if len(e) > len(w) else 0
                    else:
                        continue
                    number = _word_number(w) if not left_number else left_number
                    # A visible subject on each side must not silently switch
                    # number; this is an ordering constraint, not a readability
                    # score.  It only applies to the outer noun/pronoun slots.
                    stack.append((i + 1, j - 1, rem, new_owner, number, left + (w,), right_rev + (v,)))
        elif owner == 1:
            if j < 0:
                stats["shape_mismatch"] += 1
                continue
            rt = right_tags[j]
            for v in by.get(rt, ()):
                if not allowed(v):
                    continue
                e = v[::-1]
                if residual.startswith(e):
                    rem, new_owner = residual[len(e):], 1 if len(residual) > len(e) else 0
                elif e.startswith(residual):
                    rem, new_owner = e[len(residual):], -1 if len(e) > len(residual) else 0
                else:
                    continue
                stack.append((i, j - 1, rem, new_owner, left_number, left, right_rev + (v,)))
        else:
            if i >= len(left_tags):
                stats["shape_mismatch"] += 1
                continue
            lt = left_tags[i]
            for w in by.get(lt, ()):
                if not allowed(w):
                    continue
                if residual.startswith(w):
                    rem, new_owner = residual[len(w):], -1 if len(residual) > len(w) else 0
                elif w.startswith(residual):
                    rem, new_owner = w[len(residual):], 1 if len(w) > len(residual) else 0
                else:
                    continue
                stack.append((i + 1, j, rem, new_owner, left_number, left + (w,), right_rev))
    return rows, dict(stats), probes


def run() -> dict:
    by, reservoir_info = reservoir.reservoir(reservoir.parent._brown_lexicon())
    all_rows: list[dict] = []
    stats = Counter()
    per_pair: dict[str, dict] = {}
    for name, pair in PAIRS.items():
        rows, state, probes = solve(pair["left"], pair["right"], by)
        for row in rows:
            row["pair_name"] = name
        all_rows.extend(rows)
        per_pair[name] = {"stats": state, "rendered_probes": probes}
        stats.update({f"{name}.{key}": value for key, value in state.items()})
    unique: dict[str, dict] = {}
    for row in all_rows:
        tape = row["audit"]["normalized_tape"]
        if tape not in unique or row["lm_prior"] > unique[tape]["lm_prior"]:
            unique[tape] = row
    rendered = sorted(unique.values(), key=lambda r: (r["mechanically_admitted"], r["lm_prior"], r["audit"]["letters"]), reverse=True)
    admitted = [row for row in rendered if row["mechanically_admitted"]]
    return {
        "experiment_id": EXPERIMENT_ID,
        "signature": SIGNATURE,
        "status": "completed_no_reader_promotion" if not admitted else "exact_hits_pending_blinded_readers",
        "parent_repair": "variable-length-role-reservoir-centerout-20260915",
        "repair": "pair distinct left/right semantic templates and carry a subject-number state through asymmetric debt transitions",
        "config": {
            "template_pairs": {name: {key: list(value) if isinstance(value, tuple) else value for key, value in pair.items()} for name, pair in PAIRS.items()},
            "inventory_sizes": {key: len(value) for key, value in by.items()},
            "budget_per_pair": 600_000,
            "catalogue_text_copied": False,
        },
        "reservoir": reservoir_info,
        "stats": {**stats, "template_pairs": len(PAIRS), "unique_terminal_rows": len(rendered), "mechanically_admitted": len(admitted), "reader_eligible": 0},
        "per_pair": per_pair,
        "rendered_probes": [probe for pair in per_pair.values() for probe in pair["rendered_probes"]][:160],
        "rendered_candidates": rendered[:100],
        "exact_candidates": admitted,
        "next_repair": "If exact terminals remain empty, add a typed agreement transition between unlike templates; do not replay same-template search or enlarge this reservoir.",
        "provenance": {
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "parent_script_sha256": hashlib.sha256(PARENT.read_bytes()).hexdigest(),
            "source": "Brown universal POS counts and wordfreq reversible reservoir; no intact source sentences",
            "programmatic_readability_claim": False,
        },
    }


if __name__ == "__main__":
    out = run()
    path = ROOT / "runs" / "asymmetric-template-reservoir-centerout-20260915.json"
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"status": out["status"], "stats": out["stats"], "path": str(path)}, indent=2))
