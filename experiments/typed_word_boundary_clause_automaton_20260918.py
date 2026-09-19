"""Word-token clause automaton with residuals crossing phrase boundaries.

The reverse-slot experiment still paired whole phrases.  This lane keeps the
same independently typed subject/verb/object clauses but expands one lexical
token at a time.  If a word on one edge leaves a residual, the other edge may
choose its next grammatical token; this is the cross-boundary behavior that
the 38-letter seed uses (``Diana`` -> ``inspire`` -> ``men`` -> ``some``).

No finished tape is reversed to create an output.  The two clauses are chosen
independently from typed inventories, matched through a live residual, and
then checked by an independent normalized-tape/hash audit and the shared
mechanical gate.  Readability is never certified programmatically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

LEFT_PHASES = ("SUBJ_DET", "SUBJ_NOUN", "VERB", "OBJ_DET", "OBJ_NOUN", "ADJ_PREP", "ADJ_NOUN")
RIGHT_BUILD_PHASES = ("ADJ_NOUN", "ADJ_PREP", "OBJ_NOUN", "OBJ_DET", "VERB", "SUBJ_NOUN", "SUBJ_DET")

# Empty determiner is an explicit grammatical option, not a character filler:
# proper names and bare plural/mass nouns may be objects or subjects.
SUBJ_DETS = (("", None), ("a", "SG"), ("an", "SG"), ("the", None),
             ("our", "PL"), ("some", "PL"), ("one", "SG"),
             ("my", None), ("no", None))
OBJ_DETS = (("", None), ("a", "SG"), ("an", "SG"), ("the", None),
            ("our", "PL"), ("some", "PL"), ("one", "SG"),
            ("my", None), ("nine", "PL"))

# Inventory order is deliberate and stable: ordinary scene words precede the
# diagnostic proper names and semordnilap probes.  It is a proposal order only.
SUBJECTS = (
    ("aide", "SG"), ("editor", "SG"), ("writer", "SG"), ("reader", "SG"),
    ("poet", "SG"), ("scribe", "SG"), ("teacher", "SG"), ("doctor", "SG"),
    ("baker", "SG"), ("farmer", "SG"), ("guard", "SG"), ("pilot", "SG"),
    ("artist", "SG"), ("author", "SG"), ("nurse", "SG"), ("child", "SG"),
    ("men", "PL"), ("women", "PL"), ("editors", "PL"), ("writers", "PL"),
    ("readers", "PL"), ("poets", "PL"), ("teachers", "PL"), ("doctors", "PL"),
    ("bakers", "PL"), ("farmers", "PL"), ("guards", "PL"), ("pilots", "PL"),
    ("artists", "PL"), ("authors", "PL"), ("children", "PL"),
    ("diana", "SG"), ("leon", "SG"), ("noel", "SG"), ("nora", "SG"),
    ("eva", "SG"), ("ada", "SG"), ("anna", "SG"), ("ava", "SG"),
    ("otto", "SG"), ("bob", "SG"), ("ray", "SG"), ("sam", "SG"),
    ("max", "SG"), ("iris", "SG"), ("ian", "SG"), ("lee", "SG"),
)
OBJECTS = (
    ("memos", "PL"), ("memo", "SG"), ("notes", "PL"), ("note", "SG"),
    ("maps", "PL"), ("map", "SG"), ("letters", "PL"), ("letter", "SG"),
    ("stories", "PL"), ("story", "SG"), ("signals", "PL"), ("signal", "SG"),
    ("parcels", "PL"), ("parcel", "SG"), ("poems", "PL"), ("poem", "SG"),
    ("prose", "MASS"), ("reports", "PL"), ("report", "SG"), ("atlas", "SG"),
    ("drafts", "PL"), ("draft", "SG"), ("diaries", "PL"), ("diary", "SG"),
    ("books", "PL"), ("book", "SG"), ("pages", "PL"), ("page", "SG"),
    ("essays", "PL"), ("essay", "SG"), ("diana", "SG"), ("leon", "SG"),
    ("noel", "SG"), ("arena", "SG"), ("saga", "SG"),
    ("eva", "SG"), ("ada", "SG"), ("anna", "SG"), ("ava", "SG"),
    ("otto", "SG"), ("bob", "SG"), ("ray", "SG"), ("sam", "SG"),
    ("max", "SG"), ("iris", "SG"), ("ian", "SG"), ("lee", "SG"),
)
VERBS = (
    ("rips", "SG"), ("reads", "SG"), ("writes", "SG"), ("edits", "SG"),
    ("marks", "SG"), ("draws", "SG"), ("sends", "SG"), ("helps", "SG"),
    ("guides", "SG"), ("carries", "SG"), ("opens", "SG"), ("hears", "SG"),
    ("notes", "SG"), ("keeps", "SG"), ("holds", "SG"), ("makes", "SG"),
    ("takes", "SG"), ("meets", "SG"), ("calls", "SG"), ("names", "SG"),
    ("shows", "SG"), ("finds", "SG"), ("likes", "SG"), ("needs", "SG"),
    ("uses", "SG"), ("offers", "SG"), ("moves", "SG"), ("leads", "SG"),
    ("asks", "SG"), ("knows", "SG"), ("traces", "SG"), ("copies", "SG"),
    ("inspires", "SG"), ("carries", "SG"),
    ("inspire", "PL"), ("read", "PL"), ("write", "PL"), ("edit", "PL"),
    ("mark", "PL"), ("draw", "PL"), ("send", "PL"), ("help", "PL"),
    ("guide", "PL"), ("carry", "PL"), ("open", "PL"), ("hear", "PL"),
    ("note", "PL"), ("keep", "PL"), ("hold", "PL"), ("make", "PL"),
    ("take", "PL"), ("meet", "PL"), ("call", "PL"), ("name", "PL"),
    ("show", "PL"), ("find", "PL"), ("like", "PL"), ("need", "PL"),
    ("use", "PL"), ("offer", "PL"), ("move", "PL"), ("lead", "PL"),
    ("ask", "PL"), ("know", "PL"), ("trace", "PL"), ("copy", "PL"),
    ("saw", "ANY"), ("was", "ANY"), ("met", "ANY"), ("ate", "ANY"),
    ("did", "ANY"), ("ran", "ANY"), ("sat", "ANY"), ("led", "ANY"),
    ("said", "ANY"), ("sent", "ANY"), ("read", "ANY"),
)

ADJ_PREPS = ("at", "by", "in", "near", "with", "for", "on")
ADJ_NOUNS = ("dawn", "home", "noon", "town", "sea", "night", "arena", "river", "garden", "paper", "stone", "shore", "room", "school", "road", "station")

SEED_TAPE = "anaideripsninememossomemeninspirediana"
SEED_WORDS = frozenset(("an", "aide", "rips", "nine", "memos", "some", "men", "inspire", "diana"))


def tape(text: str) -> str:
    return normalize_letters(text)


def _audit(text: str) -> dict:
    t = tape(text)
    r = t[::-1]
    forward = hashlib.sha256(t.encode()).hexdigest()
    reverse = hashlib.sha256(r.encode()).hexdigest()
    return {
        "letters": len(t),
        "two_pointer_exact": t == r,
        "mismatch_count": sum(a != b for a, b in zip(t, r)),
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "sha_equal_under_reversal": forward == reverse,
    }


def _choices(phase: str, features: dict):
    if phase == "SUBJ_DET":
        for word, number in SUBJ_DETS:
            if number is not None and features.get("subj_num") not in (None, number):
                continue
            yield word, {"subj_num": number} if number is not None else {}
    elif phase == "SUBJ_NOUN":
        for word, number in SUBJECTS:
            if features.get("subj_num") not in (None, number):
                continue
            if features.get("verb_num") not in (None, "ANY", number):
                continue
            yield word, {"subj_num": number}
    elif phase == "VERB":
        for word, number in VERBS:
            if number != "ANY" and features.get("subj_num") not in (None, number):
                continue
            yield word, {"verb_num": number}
    elif phase == "OBJ_DET":
        for word, number in OBJ_DETS:
            if number is not None and features.get("obj_num") not in (None, number):
                continue
            yield word, {"obj_num": number} if number is not None else {}
    elif phase == "OBJ_NOUN":
        for word, number in OBJECTS:
            if features.get("obj_num") not in (None, number):
                continue
            yield word, {"obj_num": number}
    elif phase == "ADJ_PREP":
        present = features.get("adj_present")
        if present is False:
            yield "", {"adj_present": False}
        elif present is True:
            for word in ADJ_PREPS:
                yield word, {"adj_present": True}
        else:
            yield "", {"adj_present": False}
            for word in ADJ_PREPS:
                yield word, {"adj_present": True}
    elif phase == "ADJ_NOUN":
        present = features.get("adj_present")
        if present is False:
            yield "", {"adj_present": False}
        elif present is True:
            for word in ADJ_NOUNS:
                yield word, {"adj_present": True}
        else:
            yield "", {"adj_present": False}
            for word in ADJ_NOUNS:
                yield word, {"adj_present": True}
    else:
        raise ValueError(phase)


def _consume(side: str, residual: str, left_piece: str, right_piece: str):
    left = residual + tape(left_piece) if side == "L" else tape(left_piece)
    right = residual + tape(right_piece)[::-1] if side == "R" else tape(right_piece)[::-1]
    k = min(len(left), len(right))
    if left[:k] != right[:k]:
        return None
    if len(left) > len(right):
        return "L", left[k:]
    if len(right) > len(left):
        return "R", right[k:]
    return "", ""


def _render(left: tuple[str, ...], right_build: tuple[str, ...]) -> str:
    words = tuple(w for w in left if w) + tuple(w for w in reversed(right_build) if w)
    return " ".join(words) + "."


def run(*, max_states: int = 500_000, witnesses_per_state: int = 4) -> dict:
    # Each bucket stores a few independent lexical witnesses for one live
    # residual/type state; this avoids a duplicate Cartesian sweep while not
    # allowing the first lexical witness to erase every readable alternative.
    initial = (0, 0, "", "", (), ())
    buckets = {initial: [((), ())]}
    counts = [1]
    pruned = Counter()
    for _round in range(20):
        if not buckets:
            break
        next_buckets = {}
        expanded_any = False
        for (li, ri, side, residual, lf_sig, rf_sig), witnesses in buckets.items():
            lf_base = dict(lf_sig)
            rf_base = dict(rf_sig)
            if li >= len(LEFT_PHASES) and ri >= len(RIGHT_BUILD_PHASES):
                # Terminal states are retained in a special bucket; no more
                # lexical choices are needed.
                key = (li, ri, side, residual, tuple(sorted(lf_base.items())), tuple(sorted(rf_base.items())))
                next_buckets.setdefault(key, []).extend(witnesses)
                continue
            expanded_any = True
            for left, right, nli, nri, ns, nd, nlf, nrf in _expansions(
                li, ri, side, residual, lf_base, rf_base, witnesses
            ):
                key = (nli, nri, ns, nd, tuple(sorted(nlf.items())), tuple(sorted(nrf.items())))
                # Keep the live lexical witnesses, but deduplicate exact word
                # paths and cap each residual/type bucket deterministically.
                arr = next_buckets.setdefault(key, [])
                lwords, rwords = left, right
                if (lwords, rwords) not in arr:
                    arr.append((lwords, rwords))
                if len(arr) > witnesses_per_state:
                    del arr[witnesses_per_state:]
                if sum(len(v) for v in next_buckets.values()) >= max_states:
                    pruned["state_budget"] += 1
                    break
            if sum(len(v) for v in next_buckets.values()) >= max_states:
                break
        buckets = next_buckets
        counts.append(sum(len(v) for v in buckets.values()))
        # Every transition advances at least one phase; the maximum is finite.
        if counts[-1] == 0 or not expanded_any:
            break

    rows = []
    for (li, ri, side, residual, lf_sig, rf_sig), witnesses in buckets.items():
        if li != len(LEFT_PHASES) or ri != len(RIGHT_BUILD_PHASES) or side or residual:
            continue
        for left, right_build in witnesses:
            rendered = _render(left, right_build)
            t = tape(rendered)
            checks = mechanical_admission_checks(rendered, min_letters=30, max_letters=220)
            words = frozenset(w.casefold() for w in left + right_build if w)
            seed_control = t == SEED_TAPE or words == SEED_WORDS
            rows.append({
                "rendered": rendered,
                "left_clause": list(left),
                "right_clause": list(reversed(right_build)),
                "letters": len(t),
                "normalized_tape": t,
                "audit": _audit(rendered),
                "mechanical_checks": checks,
                "independent_exact": bool(t) and t == t[::-1],
                "seed_control": seed_control,
                "mechanically_admitted_before_seed_exclusion": bool(t) and t == t[::-1] and all(checks.values()),
                "mechanically_admitted": bool(t) and t == t[::-1] and all(checks.values()) and not seed_control,
                "reader_status": "not_run; programmatic checks do not certify readability",
            })
    rows.sort(key=lambda row: (-row["mechanically_admitted"], -row["letters"], row["rendered"]))
    admitted = [row for row in rows if row["mechanically_admitted"]]
    seed_controls = [row for row in rows if row["seed_control"] and row["independent_exact"]]
    return {
        "status": "typed_word_boundary_clause_complete",
        "experiment_id": "typed-word-boundary-clause-automaton-20260918",
        "signature": "typed-word-boundary|cross-phrase-residual|independent-role-clauses|agreement-carry",
        "config": {"left_phases": LEFT_PHASES, "right_build_phases": RIGHT_BUILD_PHASES,
                    "max_states": max_states, "witnesses_per_state": witnesses_per_state},
        "stats": {"state_counts": counts, "terminal_exact": len(rows),
                   "mechanically_admitted": len(admitted), "seed_control_exact": len(seed_controls),
                   "new_exact": len([row for row in rows if row["independent_exact"] and not row["seed_control"]]),
                   "reader_eligible": 0,
                   "pruned": dict(pruned)},
        "rendered_candidates_and_probes": rows,
        "admitted": admitted,
        "provenance": {"source": "hand-authored typed lexical inventory; clauses selected independently",
                        "finished_tape_reversed": False, "catalogue_text_imported": False,
                        "seed_scaffold_in_output": bool(seed_controls),
                        "promoted_seed_scaffold_in_output": False,
                        "seed_control_policy": "known 38-letter seed and word-set rearrangements are retained as smoke controls but excluded from promotion",
                        "independent_validator": "normalized two-pointer tape plus forward/reverse SHA-256",
                        "human_readability_certified": False},
        "next_repair": "add optional typed adjunct tokens after the cross-boundary SVO seam and rank complete-clause witnesses for manual review",
        "reader_gate": "closed until a candidate passes manual intact-prose review and randomized blinded controls",
    }


def _expansions(li, ri, side, residual, lf, rf, witnesses):
    """Yield transitions; the helper keeps lexical paths independent."""
    for left_words, right_words in witnesses:
        if side == "":
            if li >= len(LEFT_PHASES) and ri < len(RIGHT_BUILD_PHASES):
                for rp, rfeat in _choices(RIGHT_BUILD_PHASES[ri], rf):
                    got = _consume("", "", "", rp)
                    if got is not None:
                        ns, nd = got
                        yield (left_words, right_words + ((rp,) if rp else ()), li, ri + 1,
                               ns, nd, dict(lf), {**rf, **rfeat})
                continue
            if ri >= len(RIGHT_BUILD_PHASES) and li < len(LEFT_PHASES):
                for lp, lfeat in _choices(LEFT_PHASES[li], lf):
                    got = _consume("", "", lp, "")
                    if got is not None:
                        ns, nd = got
                        yield (left_words + ((lp,) if lp else ()), right_words, li + 1, ri,
                               ns, nd, {**lf, **lfeat}, dict(rf))
                continue
            for lp, lfeat in _choices(LEFT_PHASES[li], lf):
                for rp, rfeat in _choices(RIGHT_BUILD_PHASES[ri], rf):
                    got = _consume("", "", lp, rp)
                    if got is None:
                        continue
                    ns, nd = got
                    nlf, nrf = {**lf, **lfeat}, {**rf, **rfeat}
                    yield (left_words + ((lp,) if lp else ()),
                           right_words + ((rp,) if rp else ()), li + 1, ri + 1,
                           ns, nd, nlf, nrf)
        elif side == "L":
            if ri >= len(RIGHT_BUILD_PHASES):
                continue
            for rp, rfeat in _choices(RIGHT_BUILD_PHASES[ri], rf):
                got = _consume("L", residual, "", rp)
                if got is None:
                    continue
                ns, nd = got
                yield (left_words,
                       right_words + ((rp,) if rp else ()), li, ri + 1,
                       ns, nd, dict(lf), {**rf, **rfeat})
        else:
            if li >= len(LEFT_PHASES):
                continue
            for lp, lfeat in _choices(LEFT_PHASES[li], lf):
                got = _consume("R", residual, lp, "")
                if got is None:
                    continue
                ns, nd = got
                yield (left_words + ((lp,) if lp else ()),
                       right_words, li + 1, ri, ns, nd,
                       {**lf, **lfeat}, dict(rf))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-states", type=int, default=500_000)
    ap.add_argument("--witnesses-per-state", type=int, default=4)
    args = ap.parse_args()
    if args.out.exists():
        ap.error(f"refusing to overwrite existing output: {args.out}")
    result = run(max_states=args.max_states, witnesses_per_state=args.witnesses_per_state)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], indent=2))


if __name__ == "__main__":
    main()
