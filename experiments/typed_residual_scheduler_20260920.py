"""Typed clause grammar with one-sided live residual scheduling.

The previous bilateral grammar had to choose a lexical word on both sides at
each step.  This lane keeps both grammar frontiers typed, but lets the side
with available character debt advance alone.  Grammar and valency are hard
constraints; frequency only orders the bounded beam.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from bilateral_grammar_csp_20260920 import _consume
from forward_lexicalized_grammar_20260920 import Word, admission_ok, independent_audit, letters

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/typed-residual-scheduler-20260920.json"

GRAMMAR = {
    "CLAUSE": (("SUBJ", "V", "OBJ"), ("SUBJ", "V", "PP"), ("SUBJ", "ADV", "V", "OBJ")),
    "SUBJ": (("PROPN",), ("PRON",), ("N",), ("DET", "N"), ("DET", "ADJ", "N")),
    "OBJ": (("N",), ("PROPN",), ("DET", "N"), ("DET", "ADJ", "N"), ("NUM", "N")),
    "PP": (("PREP", "DET", "N"), ("PREP", "DET", "ADJ", "N")),
}


@dataclass(frozen=True)
class State:
    left_symbols: tuple[str, ...]
    right_symbols: tuple[str, ...]
    left: tuple[str, ...]
    right_rev: tuple[str, ...]
    left_residual: str
    right_residual: str
    score: float


def load_lexicon(path=ROOT / "data/brown_pcfg_bank_20260920.json", limit=70, include_relative=False):
    bank = json.loads(Path(path).read_text())["lexicon"]
    mapping = {"DET": "DET", "NOUN": "N", "VERB": "V", "ADJ": "ADJ",
               "PREP": "PREP", "PRON": "PRON"}
    words = []
    for source, pos in mapping.items():
        for row in bank[source][:limit]:
            text = row["word"].casefold()
            if re.fullmatch(r"[a-z]+", text):
                words.append(Word(text, pos))
    for text in "diana leon noel elba anna adam eve oscar ada otto".split():
        words.append(Word(text, "PROPN", entity="person"))
    for text in "one two three four five six seven eight nine ten eleven twelve".split():
        words.append(Word(text, "NUM"))
    # Keep the anchor vocabulary available without importing its sentence.
    words.extend([Word("aide", "N"), Word("rips", "V"), Word("memos", "N"),
                  Word("men", "N"), Word("inspire", "V")])
    if include_relative:
        words.extend(Word(text, "RELPRON") for text in ("who", "that", "which"))
    unique = {(w.text, w.pos): w for w in words}
    return tuple(unique.values())


def _nested_span(words: tuple[str, ...]) -> bool:
    for i in range(len(words)):
        for j in range(i + 2, len(words) + 1):
            if i == 0 and j == len(words):
                continue
            tape = letters(" ".join(words[i:j]))
            if len(tape) > 1 and tape == tape[::-1]:
                return True
    return False


def search(lexicon, *, grammar=None, max_words=14, max_nodes=250_000, beam_width=20_000):
    grammar = grammar or GRAMMAR
    by_pos = {}
    for word in lexicon:
        by_pos.setdefault(word.pos, []).append(word)
    right_by_exposed = {}
    for word in lexicon:
        right_by_exposed.setdefault(letters(word.text)[-1], []).append(word)
    # A simple common-word ordering is a search tie-breaker, never a readable
    # output certificate. Brown scores are not carried into the artifact.
    common = {"the": 8.0, "a": 7.8, "an": 7.0, "some": 6.3, "men": 5.4,
              "inspire": 4.2, "rips": 4.0, "aide": 4.0, "memos": 4.0}
    for pos, rows in by_pos.items():
        rows.sort(key=lambda w: (-common.get(w.text, 2.0), w.text))
    found = {}
    beam = [State(("CLAUSE",), ("CLAUSE",), (), (), "", "", 0.0)]
    stats = {"states": 0, "char_prunes": 0, "repeat_prunes": 0, "beam_prunes": 0,
             "grammar_expansions": 0, "complete": 0}

    for _ in range(max_words * 3):
        if not beam:
            break
        children = []
        for state in beam:
            if stats["states"] >= max_nodes:
                break
            stats["states"] += 1
            if not state.left_symbols and not state.right_symbols:
                stats["complete"] += 1
                if state.left_residual or state.right_residual:
                    continue
                words = state.left + tuple(reversed(state.right_rev))
                if len(words) >= 4 and admission_ok(list(words)) and not _nested_span(words):
                    text = " ".join(words)
                    audit = independent_audit(text)
                    if audit["exact"] and audit["letters"] >= 39:
                        found[text] = {"length": audit["letters"], "rendered": text + ".",
                                       "words": words, "audit": audit,
                                       "provenance": {"typed_grammar": True,
                                                      "one_sided_residual_scheduler": True,
                                                      "nested_span_rejected": True,
                                                      "finished_tape_reversal": False,
                                                      "post_hoc_repair": False,
                                                      "word_order_mirroring": False,
                                                      "catalogue_text": False}}
                continue
            if len(state.left) + len(state.right_rev) >= max_words:
                continue
            # Expand grammar symbols before emitting characters.
            if state.left_symbols and state.left_symbols[0] in grammar:
                for production in grammar[state.left_symbols[0]]:
                    stats["grammar_expansions"] += 1
                    children.append(State(tuple(production) + state.left_symbols[1:],
                                          state.right_symbols, state.left, state.right_rev,
                                          state.left_residual, state.right_residual, state.score))
                continue
            if state.right_symbols and state.right_symbols[-1] in grammar:
                for production in grammar[state.right_symbols[-1]]:
                    stats["grammar_expansions"] += 1
                    children.append(State(state.left_symbols, state.right_symbols[:-1] + tuple(production),
                                          state.left, state.right_rev, state.left_residual,
                                          state.right_residual, state.score))
                continue
            left_choices = by_pos.get(state.left_symbols[0], ()) if state.left_symbols else ()
            right_choices = by_pos.get(state.right_symbols[-1], ()) if state.right_symbols else ()
            if state.left_residual and not state.right_residual:
                pairs = ((None, w) for w in right_choices)
            elif state.right_residual and not state.left_residual:
                pairs = ((w, None) for w in left_choices)
            else:
                pairs = ((a, b) for a in left_choices
                         for b in right_by_exposed.get(letters(a.text)[0], ())
                         if b.pos == state.right_symbols[-1])
            for left_word, right_word in pairs:
                chosen = state.left + state.right_rev
                if ((left_word and left_word.text in chosen) or
                        (right_word and right_word.text in chosen)):
                    stats["repeat_prunes"] += 1
                    continue
                ls = state.left_residual + (letters(left_word.text) if left_word else "")
                rs = state.right_residual + (letters(right_word.text)[::-1] if right_word else "")
                residual = _consume(ls, rs)
                if residual is None:
                    stats["char_prunes"] += 1
                    continue
                left_symbols = state.left_symbols[1:] if left_word else state.left_symbols
                right_symbols = state.right_symbols[:-1] if right_word else state.right_symbols
                children.append(State(left_symbols, right_symbols,
                                      state.left + ((left_word.text,) if left_word else ()),
                                      state.right_rev + ((right_word.text,) if right_word else ()),
                                      residual[0], residual[1],
                                      state.score + (common.get(left_word.text, 2.0) if left_word else 0.0) +
                                      (common.get(right_word.text, 2.0) if right_word else 0.0)))
        if stats["states"] >= max_nodes:
            break
        children.sort(key=lambda s: (s.score + 0.03 * sum(map(len, s.left + s.right_rev)),
                                     len(s.left) + len(s.right_rev)), reverse=True)
        if len(children) > beam_width:
            stats["beam_prunes"] += len(children) - beam_width
            children = children[:beam_width]
        beam = children
    stats["status"] = "timeout" if stats["states"] >= max_nodes else ("SAT" if found else "UNSAT")
    stats["exact"] = len(found)
    return {"paths": sorted(found.values(), key=lambda r: (-r["length"], r["rendered"])), "stats": stats}


def run(*, lexicon_limit=70, max_words=14, max_nodes=250_000, beam_width=20_000):
    result = search(load_lexicon(limit=lexicon_limit), max_words=max_words,
                    max_nodes=max_nodes, beam_width=beam_width)
    result["experiment_id"] = "typed-residual-scheduler-20260920"
    result["provenance"] = {
        "method": "typed SVO/PP grammar with one-sided live residual scheduling",
        "lexicon_limit": lexicon_limit, "max_words": max_words,
        "max_nodes": max_nodes, "beam_width": beam_width,
        "one_sided_residual_scheduler": True, "candidate_reranking": False,
        "reader_gate": "closed unless exact row survives independent audit and blinded intact-versus-shuffled reading",
        "next_construction": "if exact remains empty, add a typed relative complement rather than widening this beam",
    }
    return result


if __name__ == "__main__":
    result = run()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"]))
    for row in result["paths"][:20]:
        print(row["rendered"])
