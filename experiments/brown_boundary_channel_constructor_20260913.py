"""Diagnostic corpus-compiled boundary channels, not a prose constructor.

Brown supplies token/POS/frequency options only.  The constructor never copies
a Brown sentence and never treats attestation as evidence of readability.  A
state is extended by one lexical item on each channel; the currently exposed
letter overlap is checked before that state enters the beam. POS tags do not
prove agreement, valency, semantics, or a complete sentence, so this module is
permanently diagnostic-only and may not admit any candidate.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

MIN_LETTERS, MAX_LETTERS = 100, 180
ROLE_TAGS = {
    "det": ("AT", "DT", "DTI", "DTS", "DTX"),
    "adj": ("JJ", "JJR", "JJS"),
    "noun": ("NN", "NNS", "NP", "NPS"),
    "verb": ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ"),
    "adv": ("RB", "RBR", "RBS"),
    "prep": ("IN",),
}


@dataclass(frozen=True)
class Lexeme:
    word: str
    role: str
    count: int
    source: str = "nltk.corpus.brown"


def brown_lexicon(limit_per_role: int = 80) -> dict[str, tuple[Lexeme, ...]]:
    """Compile clean frequent lexical options from the installed Brown corpus."""
    from nltk.corpus import brown
    counts: dict[str, Counter[str]] = {role: Counter() for role in ROLE_TAGS}
    for raw, tag in brown.tagged_words():
        word = raw.casefold()
        if not word.isascii() or not word.isalpha() or len(word) < 1:
            continue
        for role, tags in ROLE_TAGS.items():
            if tag in tags:
                counts[role][word] += 1
    return {
        role: tuple(Lexeme(w, role, n) for w, n in counts[role].most_common(limit_per_role))
        for role in ROLE_TAGS
    }


def overlap_ok(left: str, right: str) -> tuple[bool, int]:
    """Check every currently exposed letter pair, allowing shifted spaces."""
    a, b = normalize_letters(left), normalize_letters(right)[::-1]
    width = min(len(a), len(b))
    for i in range(width):
        if a[i] != b[i]:
            return False, i
    return True, width


def emit_pair(left: str, right: str) -> dict[str, object]:
    ok, matched = overlap_ok(left, right)
    return {"accepted": ok, "matched_letters": matched, "left": left, "right": right}


def render(left: tuple[str, ...], right: tuple[str, ...]) -> str:
    text = " ".join((*left, *reversed(right))).strip()
    return text[:1].upper() + text[1:] + "."


# POS-shaped diagnostic channels only. These are deliberately *not* called
# clauses: Brown tags cannot establish that independently substituted words
# form a grammatical or semantically coherent sentence.
LEFT_ROLES = ("det", "adj", "noun", "verb", "det", "noun", "prep", "det", "noun", "verb", "det", "noun")
RIGHT_ROLES = ("det", "adj", "noun", "verb", "det", "noun", "prep", "det", "noun", "verb", "det", "noun")


def search(*, beam: int = 120, limit_per_role: int = 80) -> dict[str, object]:
    vocab = brown_lexicon(limit_per_role)
    frontier = [((), (), "", "", [])]
    rejected: list[dict[str, object]] = []
    states = 0
    for depth, (lrole, rrole) in enumerate(zip(LEFT_ROLES, RIGHT_ROLES)):
        next_frontier = []
        # Prefer high-attestation choices, but retain independent channels.
        for left, right, ltext, rtext, provenance in frontier:
            for l in vocab[lrole]:
                for r in vocab[rrole]:
                    states += 1
                    nl, nr = (*left, l.word), (*right, r.word)
                    check = emit_pair(" ".join(nl), " ".join(nr))
                    prov = [*provenance, {"left": l.__dict__, "right": r.__dict__, "depth": depth}]
                    if check["accepted"]:
                        next_frontier.append((nl, nr, " ".join(nl), " ".join(nr), prov))
                    elif len(rejected) < 160:
                        # A rendered POS-shaped partial diagnostic, not a
                        # complete sentence and never a candidate.
                        tail_l = tuple(x.word for x in (vocab[x][0] for x in LEFT_ROLES[depth + 1:]))
                        tail_r = tuple(x.word for x in (vocab[x][0] for x in RIGHT_ROLES[depth + 1:]))
                        rejected.append({"kind": "boundary_emission_rejection", "depth": depth,
                            "rendered": render(nl + tail_l, nr + tail_r),
                            "emission": check, "provenance": prov})
        next_frontier.sort(key=lambda x: sum(p["left"]["count"] + p["right"]["count"] for p in x[4]), reverse=True)
        frontier = next_frontier[:beam]
        if not frontier:
            break
    closures = []
    for left, right, _, _, provenance in frontier:
        text = render(left, right)
        checks = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
        closures.append({"rendered": text, "left_clause": " ".join(left), "right_clause": " ".join(reversed(right)),
                         "letters": len(normalize_letters(text)), "checks": checks, "provenance": provenance,
                         "independent_exact": normalize_letters(text) == normalize_letters(text)[::-1]})
    # No independent feature-grammar witness exists, so a POS-shaped closure
    # must fail closed even if it happens to satisfy the letter checks.
    admitted: list[dict[str, object]] = []
    return {"status": "diagnostic_only_missing_feature_grammar_witness",
            "method": "brown_compiled_boundary_channel", "corpus": "nltk.corpus.brown",
            "parameters": {"beam": beam, "limit_per_role": limit_per_role, "min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS},
            "role_skeleton": {"left": LEFT_ROLES, "right": RIGHT_ROLES},
            "stats": {"states_emitted": states, "surviving_frontier": len(frontier), "partial_rejections": len(rejected), "admitted": len(admitted)},
            "closures": closures, "rejections": rejected, "admitted": admitted,
            "candidate_use": "forbidden: POS channels are not grammatical witnesses",
            "next_repair_operator": "Compile boundary options only as lexical domains for the connected feature grammar; require its independent agreement, valency, and semantic parse before any exact closure can be considered.",
            "reader_facing_next_test": "None. This diagnostic has no grammatical witness and is ineligible for any reader package."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beam", type=int, default=120)
    parser.add_argument("--limit-per-role", type=int, default=80)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "brown-boundary-channel-constructor-2026-09-13" / "result.json")
    args = parser.parse_args()
    result = search(beam=args.beam, limit_per_role=args.limit_per_role)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))


if __name__ == "__main__":
    main()
