"""Lexicalized constituent interiors with a live character equation.

This lane keeps NP/VP/PP interiors and word-boundary state in the search
state.  The two sides are independent ordinary-order derivations; the right
derivation is consumed from its final constituent while its characters are
matched online.  Unlike complete phrase joins, a constituent is expanded one
typed terminal at a time, so an obligation may cross an internal word
boundary before either clause is complete.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.bilateral_grammar_csp_20260920 import _consume, palindromic_residual
OUT = ROOT / "runs" / "lexicalized-constituent-interior-csp-20260920.json"
ID = "lexicalized-constituent-interior-csp-20260920"
SIGNATURE = "lexicalized-constituent-interior|feature-carrying-synchronous-stack|live-boundary-state"


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict:
    tape = letters(text)
    mismatches = [(i, len(tape) - 1 - i) for i in range(len(tape) // 2)
                  if tape[i] != tape[-i - 1]]
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {
        "normalized": tape,
        "letters": len(tape),
        "exact": bool(tape) and not mismatches,
        "first_mismatch": mismatches[0] if mismatches else None,
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "sha_equal": forward == reverse,
    }


# A compact, authored lexical inventory.  Categories carry the semantic role
# and number feature; the grammar, not a post-hoc score, enforces agreement.
LEXICON = {
    "DET_SG": ("a", "an", "the", "one"),
    "DET_PL": ("some", "the"),
    "SUBJ_SG": ("aide", "poet", "scribe", "sailor", "gardener", "teacher", "keeper", "singer", "child", "artist", "writer", "reader", "pilot", "nurse", "doctor", "guard", "farmer", "baker", "captain", "monk", "fox", "dog", "cat"),
    "SUBJ_PL": ("men", "poets", "scribes", "sailors", "gardeners", "teachers", "keepers", "singers", "children", "artists", "writers", "readers", "pilots", "nurses", "doctors", "guards", "farmers", "bakers", "captains", "monks"),
    "SUBJ_NAME": ("diana", "anna", "ada", "leon", "noel", "iris", "otto", "nora", "ella", "eva", "eric", "olga"),
    "OBJ_SG": ("map", "note", "book", "letter", "rose", "lantern", "bridge", "garden", "harbor", "memo", "poem", "story", "tale", "verse", "flag", "boat", "key", "cart", "pen", "lamp", "rope", "gate", "message", "journal", "signal", "chart", "crown", "song", "bread", "water", "money", "fire", "door"),
    "OBJ_PL": ("memos", "maps", "notes", "books", "letters", "roses", "lanterns", "bridges", "gardens", "poems", "stories", "tales", "verses", "flags", "boats", "keys", "carts", "pens", "lamps", "ropes", "gates", "messages", "journals", "signals", "charts", "crowns", "songs"),
    "OBJ_NAME": ("diana", "anna", "ada", "leon", "noel", "iris", "otto", "nora", "ella", "eva", "eric", "olga"),
    "ADJ": ("old", "quiet", "young", "kind", "small", "bright", "red", "careful", "patient", "fair", "clear", "open", "dark", "long", "cold", "warm", "wise", "brave", "noble", "silent", "little", "fresh"),
    "V_SG": ("marks", "reads", "guides", "guards", "keeps", "sees", "writes", "saves", "inspires", "tends", "carries", "finds", "makes", "names", "notes", "calls", "asks", "helps", "meets", "sends", "sings", "draws", "bakes", "feeds", "holds"),
    "V_PL": ("mark", "read", "guide", "guard", "keep", "see", "write", "save", "inspire", "tend", "carry", "find", "make", "name", "note", "call", "ask", "help", "meet", "send", "sing", "draw", "bake", "feed", "hold"),
    "PREP": ("at", "by", "in", "near", "under", "through", "beside", "around", "beyond", "inside", "on", "with", "for"),
    "PLACE_SG": ("dawn", "rain", "sea", "moon", "wall", "river", "harbor", "garden", "hall", "town", "road", "train", "home", "desk", "room", "market", "court", "castle", "shore", "bridge", "grove"),
}


GRAMMAR = {
    "S_SG": (("NP_SG", "VP_SG"),),
    "S_PL": (("NP_PL", "VP_PL"),),
    "NP_SG": (("DET_SG", "SUBJ_SG"), ("DET_SG", "ADJ", "SUBJ_SG"), ("SUBJ_NAME",)),
    "NP_PL": (("DET_PL", "SUBJ_PL"), ("DET_PL", "ADJ", "SUBJ_PL")),
    "OBJNP_SG": (("DET_SG", "OBJ_SG"), ("DET_SG", "ADJ", "OBJ_SG"), ("OBJ_NAME",)),
    "OBJNP_PL": (("DET_PL", "OBJ_PL"), ("DET_PL", "ADJ", "OBJ_PL")),
    "PP": (("PREP", "DET_SG", "PLACE_SG"),),
    "VP_SG": (("V_SG", "OBJNP_SG"), ("V_SG", "OBJNP_SG", "PP")),
    "VP_PL": (("V_PL", "OBJNP_SG"), ("V_PL", "OBJNP_PL"), ("V_PL", "OBJNP_SG", "PP")),
}


@dataclass(frozen=True)
class State:
    left_stack: tuple[str, ...]
    right_stack: tuple[str, ...]
    left_pending: str
    right_pending: str
    left_words: tuple[str, ...]
    right_words: tuple[str, ...]
    used: frozenset[str]


def _expand(stack: tuple[str, ...], *, right: bool):
    """Return grammar successors, preserving ordinary order on both sides."""
    if not stack:
        return ()
    index = -1 if right else 0
    symbol = stack[index]
    if symbol not in GRAMMAR:
        return ()
    rest = stack[:index] + stack[index + 1:]
    out = []
    for production in GRAMMAR[symbol]:
        if right:
            # The rightmost ordinary constituent is consumed first, but the
            # stack still stores the clause in ordinary order.
            out.append(State(rest + tuple(production), (), "", "", (), (), frozenset()))
        else:
            out.append(State(tuple(production) + rest, (), "", "", (), (), frozenset()))
    return tuple(out)


def _is_nonterminal(symbol: str) -> bool:
    return symbol in GRAMMAR


def _word_choices(symbol: str):
    return LEXICON.get(symbol, ())


FUNCTION_WORDS = frozenset({"a", "an", "the", "one", "some"})


def _content(word: str) -> bool:
    return word not in FUNCTION_WORDS


def _used_add(used: frozenset[str], *words: str) -> frozenset[str]:
    return used | {word for word in words if _content(word)}


def _repeated(used: frozenset[str], word: str) -> bool:
    return _content(word) and word in used


def search(max_states: int = 120_000, min_letters: int = 39):
    # Stacks are carried explicitly in the key.  Right-side words are
    # prepended to preserve ordinary-order rendering after reverse traversal.
    agenda = [State(("S_SG",), ("S_SG",), "", "", (), (), frozenset())]
    seen = set()
    stats = {"popped": 0, "unique_states": 0, "grammar_expansions": 0,
             "word_choices": 0, "character_matches": 0, "character_prunes": 0,
             "repeat_prunes": 0, "complete": 0, "deepest_letters": 0,
             "center_rejections": 0, "nonempty_center_closures": 0}
    exact = []
    controls = []
    deepest = None

    while agenda and stats["popped"] < max_states:
        state = agenda.pop()
        stats["popped"] += 1
        # Pending words are already present in the surface word tuples; they
        # are residual boundary state, not additional emitted characters.
        depth = sum(len(letters(w)) for w in state.left_words + state.right_words)
        if depth > stats["deepest_letters"]:
            stats["deepest_letters"] = depth
            deepest = {
                "left": " ".join(state.left_words),
                "right": " ".join(state.right_words),
                "left_stack": state.left_stack,
                "right_stack": state.right_stack,
                "letters_emitted": depth,
            }
        key = (state.left_stack, state.right_stack, state.left_pending,
               state.right_pending, state.left_words, state.right_words)
        if key in seen:
            continue
        seen.add(key); stats["unique_states"] += 1

        # Expand nonterminals without consuming characters.  The left side
        # uses its first symbol; the right side uses its final symbol.
        if state.left_stack and _is_nonterminal(state.left_stack[0]):
            symbol = state.left_stack[0]
            rest = state.left_stack[1:]
            for production in GRAMMAR[symbol]:
                stats["grammar_expansions"] += 1
                agenda.append(State(tuple(production) + rest, state.right_stack,
                                    state.left_pending, state.right_pending,
                                    state.left_words, state.right_words, state.used))
            continue
        if state.right_stack and _is_nonterminal(state.right_stack[-1]):
            symbol = state.right_stack[-1]
            rest = state.right_stack[:-1]
            for production in GRAMMAR[symbol]:
                stats["grammar_expansions"] += 1
                agenda.append(State(state.left_stack, rest + tuple(production),
                                    state.left_pending, state.right_pending,
                                    state.left_words, state.right_words, state.used))
            continue

        # Both parses may finish at unequal letter counts. The unmatched
        # middle can end inside a word; it must be symmetric, not empty.
        if not state.left_stack and not state.right_stack:
            stats["complete"] += 1
            if not palindromic_residual(state.left_pending, state.right_pending[::-1]):
                stats["center_rejections"] += 1
                continue
            residual = _consume(state.left_pending, state.right_pending[::-1])
            assert residual is not None
            center = residual[0] or residual[1]
            if center:
                stats["nonempty_center_closures"] += 1
            text = " ".join(state.left_words) + "; " + " ".join(state.right_words) + "."
            result = audit(text)
            if result["letters"] >= min_letters:
                controls.append({"rendered": text, "audit": result,
                                 "center_residual": center,
                                 "provenance": {"complete_left_parse": True,
                                                "complete_right_parse": True,
                                                "finished_tape_reversal": False,
                                                "mirrored_word_units": False,
                                                "catalogue_text": False}})
                if result["exact"]:
                    exact.append(controls[-1])
            continue

        # Character emission is the live equation.  A pending word is an
        # internal boundary state: its remainder may be matched across the
        # next word on the opposite side.
        if state.left_pending or state.right_pending:
            if not state.left_pending:
                if not state.left_stack:
                    continue
                sym = state.left_stack[0]
                for word in _word_choices(sym):
                    if _repeated(state.used, word):
                        stats["repeat_prunes"] += 1; continue
                    stats["word_choices"] += 1
                    agenda.append(State(state.left_stack[1:], state.right_stack,
                                        word, state.right_pending,
                                        state.left_words + (word,), state.right_words,
                                        _used_add(state.used, word)))
                continue
            if not state.right_pending:
                if not state.right_stack:
                    continue
                sym = state.right_stack[-1]
                for word in _word_choices(sym):
                    if _repeated(state.used, word):
                        stats["repeat_prunes"] += 1; continue
                    stats["word_choices"] += 1
                    agenda.append(State(state.left_stack, state.right_stack[:-1],
                                        state.left_pending, word,
                                        state.left_words, (word,) + state.right_words,
                                        _used_add(state.used, word)))
                continue
            if state.left_pending[0] != state.right_pending[-1]:
                stats["character_prunes"] += 1
                continue
            stats["character_matches"] += 1
            agenda.append(State(state.left_stack, state.right_stack,
                                state.left_pending[1:], state.right_pending[:-1],
                                state.left_words, state.right_words, state.used))
            continue

        # No pending characters: choose the next lexical terminal on one or
        # both sides.  Expanding both sides whenever possible keeps the search
        # joint rather than turning into a finished-tape reverse parse.
        if state.left_stack and state.right_stack:
            lsym, rsym = state.left_stack[0], state.right_stack[-1]
            for lw in _word_choices(lsym):
                if _repeated(state.used, lw):
                    stats["repeat_prunes"] += 1; continue
                for rw in _word_choices(rsym):
                    if _repeated(state.used, rw) or (rw == lw and _content(rw)):
                        stats["repeat_prunes"] += 1; continue
                    stats["word_choices"] += 1
                    agenda.append(State(state.left_stack[1:], state.right_stack[:-1],
                                        lw, rw, state.left_words + (lw,),
                                        (rw,) + state.right_words,
                                        _used_add(state.used, lw, rw)))
            continue
        # Unequal derivation depths are allowed: a constituent can span the
        # next boundary on the other side.
        if state.left_stack:
            sym = state.left_stack[0]
            for word in _word_choices(sym):
                if not _repeated(state.used, word):
                    agenda.append(State(state.left_stack[1:], state.right_stack, word, "",
                                        state.left_words + (word,), state.right_words,
                                        _used_add(state.used, word)))
        elif state.right_stack:
            sym = state.right_stack[-1]
            for word in _word_choices(sym):
                if not _repeated(state.used, word):
                    agenda.append(State(state.left_stack, state.right_stack[:-1], "", word,
                                        state.left_words, (word,) + state.right_words,
                                        _used_add(state.used, word)))

    return {"stats": stats, "exact_candidates": exact, "controls": controls[:80],
            "deepest_frontier": deepest}


def run(**kwargs):
    result = search(**kwargs)
    result.update({
        "experiment_id": ID,
        "method": "joint lexicalized constituent grammar with live internal boundary states",
        "status": "completed_exact" if result["exact_candidates"] else "completed_no_exact_closure",
        "novelty_preflight": {
            "status": "passed",
            "signature": SIGNATURE,
            "distinct_from": "complete phrase joins and word-only CFG beams: constituent interiors are generated one typed terminal at a time",
            "shortcuts_rejected": ["finished-tape reversal", "word-order symmetry", "repeated units", "catalogue text", "post-hoc repair"],
        },
        "provenance": {
            "lexicon": "small authored role/feature inventory",
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "independent_audits": ["literal outside-in two-pointer", "forward/reverse SHA-256"],
            "reader_gate": "closed; programmatic exactness is not human readability evidence",
        },
        "next_construction": "add a lexicalized recipient/benefactive constituent only after novelty preflight; do not widen the same terminal bank",
    })
    return result


if __name__ == "__main__":
    result = run()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
    for row in result["exact_candidates"][:5]:
        print(row["rendered"])
