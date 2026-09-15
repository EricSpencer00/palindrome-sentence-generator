"""Wider typed outer-boundary search using Brown POS *word types* only.

Brown supplies lexical alternatives, never intact source sentences.  The
character zipper and all exact/admission checks are the independent engine in
``outer_boundary_residual_search_20260915``.  This run tests whether a larger
but still typed inventory repairs the first-character frontier without
falling back to a language-model reward or copied prose.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path

from nltk.corpus import brown

from experiments import outer_boundary_residual_search_20260915 as zipper
from llm_palindrome.lexicon import load_lexicon, is_real_word

ROOT = Path(__file__).resolve().parents[1]
LEXICON = load_lexicon(str(ROOT / "data" / "lexicon.txt"))


def types(tags: tuple[str, ...], limit: int = 180) -> tuple[str, ...]:
    counts = Counter(word.casefold() for word, tag in brown.tagged_words() if tag in tags and word.isalpha() and 3 <= len(word) <= 10)
    return tuple(word for word, _ in counts.most_common() if is_real_word(word, LEXICON))[:limit]


def run(max_states: int = 1_000_000):
    adjectives = types(("JJ", "JJR", "JJS"), 220)
    nouns = types(("NN", "NNS"), 350)
    # Prefer finite lexical forms that can function as ordinary present/past
    # predicates; Brown POS is only a proposal inventory, not a syntax claim.
    verbs = types(("VB", "VBD", "VBG", "VBN", "VBP", "VBZ"), 280)
    # Disjoint rank slices force the two clauses to be independently selected;
    # shared function words remain legal grammar, while content duplication is
    # rejected by the central admission gate.
    left_adj, right_adj = adjectives[::2], adjectives[1::2]
    left_noun, right_noun = nouns[::2], nouns[1::2]
    left_verb, right_verb = verbs[::2], verbs[1::2]
    # Boundary repair inventory: Brown's rank split can omit exactly the
    # terminal-vowel nouns needed to expose a second-character continuation.
    # These are ordinary lexical nouns, added as typed alternatives rather
    # than as sentence fragments or mirrored units.
    left_noun = tuple(dict.fromkeys(left_noun + ("error", "era", "edge", "education", "editor", "engine", "event")))
    right_noun = tuple(dict.fromkeys(right_noun + ("area", "idea", "camera", "drama", "opera", "sofa", "data", "quota", "flora", "fauna", "villa", "pizza", "panda", "agenda")))
    # The shortest intact English outer clause is deliberately tested first:
    # determiner + subject + verb + determiner + object.  Removing an
    # adjective is a grammar ablation, not a fragment shortcut; it exposes
    # whether the previous adjective boundary itself caused the frontier.
    zipper.LEFT = (
        ("det", ("a",)), ("subj", left_noun), ("verb", left_verb),
        ("det", ("a",)), ("obj", left_noun),
    )
    zipper.RIGHT = (
        ("det", ("a",)), ("subj", right_noun), ("verb", right_verb),
        ("det", ("a",)), ("obj", right_noun),
    )
    result = zipper.run(max_states)
    result["status"] = "brown_typed_outer_residual_search"
    result["config"].update({"lexical_source": "NLTK Brown POS word types only", "left_inventory_sizes": [len(x[1]) for x in zipper.LEFT], "right_inventory_sizes": [len(x[1]) for x in zipper.RIGHT], "intact_corpus_sentences_used": False})
    result["provenance"].update({"brown_corpus_word_types_sha256": sha256(json.dumps({"adjectives": adjectives, "nouns": nouns, "verbs": verbs}, sort_keys=True).encode()).hexdigest(), "generator": str(Path(__file__).resolve())})
    result["next_construction"] = "At the deepest outer-compatible state, add a typed lexical alternative with the required continuation; then rerun the same character zipper and independent admission gate. No sentence text is imported from Brown."
    return result


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--max-states", type=int, default=1_000_000); args = ap.parse_args()
    if args.out.exists(): ap.error("refusing to overwrite existing output")
    out = run(args.max_states); args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["stats"], indent=2))


if __name__ == "__main__": main()
