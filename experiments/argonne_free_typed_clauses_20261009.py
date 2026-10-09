"""Root-free typed-clause trial: removes the frozen center, not meaning review."""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.argonne_typed_overhang_growth_20261009 import (
    CENTERS, Grammar, exact_audit, predicate_reuse_audit, scene_clauses, tape,
)
from llm_palindrome.bigram import BigramModel
from llm_palindrome.search import WordTries, beam_search


def connected(facts):
    if not facts:
        return False
    reached = {facts[0][0]}
    while True:
        expanded = reached | {entity for s, _, o in facts if s in reached or o in reached for entity in (s, o)}
        if expanded == reached:
            return all(s in reached and o in reached for s, _, o in facts)
        reached = expanded


def fact_uses(clause):
    return tuple((clause.mode, fact) for fact in clause.facts)


class ClauseFrontier:
    """NFA keeps both continuing a clause and resetting at its end."""
    def __init__(self, grammar):
        self.grammar = grammar

    @lru_cache(maxsize=20000)
    def states(self, words):
        if not words:
            return frozenset({(0, frozenset())})
        out = set()
        for node, facts in self.states(words[:-1]):
            for child, completed in self.grammar.advances(node, words[-1]):
                added = fact_uses(self.grammar.clauses[completed]) if completed is not None else ()
                if len(set(added)) != len(added) or facts.intersection(added):
                    continue
                out.add((child, facts.union(added)))
        return frozenset(out)


class JoinScorer:
    def __init__(self, model):
        self.model = model

    def word_delta(self, left, right, placement, word, growth):
        if placement == "L":
            return self.model.forward(left[-2] if len(left) > 1 else None, word)
        return self.model.backward(word, right[1] if len(right) > 1 else None)


def run_job(clauses, model, *, min_letters, seconds=120, seed=0, beam_width=384,
            max_letters=2500, max_steps=1500, max_overhang=96):
    start = time.monotonic()
    deadline = start + seconds
    forward, backward = Grammar(clauses), Grammar(clauses, reverse=True)
    left_frontier, right_frontier = ClauseFrontier(forward), ClauseFrontier(backward)
    counts, seen, samples = Counter(), {}, []
    rng = random.Random(seed)
    deepest, longest = None, None
    reference_tapes = {tape(text) for text in CENTERS.values()}

    def closure(left, right):
        nonlocal longest
        words = left + right
        norm = "".join(words)
        if not norm or norm != norm[::-1] or len(norm) < min_letters:
            return False
        if words in seen:
            return seen[words]
        seen[words] = False
        counts["unique_exact_closures_checked"] += 1
        parsed = forward.parse(words)
        if parsed is None:
            counts["closure_incomplete_grammar"] += 1
            return False
        used = [clauses[c] for c in parsed]
        facts = [fact for clause in used for fact in clause.facts]
        uses = [use for clause in used for use in fact_uses(clause)]
        if len(set(uses)) != len(uses):
            counts["closure_repeated_fact"] += 1
            return False
        if not connected(facts):
            counts["closure_disconnected_fact_graph"] += 1
            return False
        text = " ".join(c.render() for c in used)
        audit = exact_audit(text)
        assert audit["exact"] and tape(text) == norm
        seen[words] = True
        if norm in admitted:
            return True
        phrase_audit = predicate_reuse_audit(used)
        for phrase in phrase_audit:
            for use in phrase["uses"]:
                use["clause_mode"] = used[use["clause_index"]].mode
            phrase["licensed_distinct_use"] = len({(u["clause_mode"], tuple(tuple(f) for f in u["canonical_fact_bindings"])) for u in phrase["uses"]}) == len(phrase["uses"])
        record = {"text": text, "letters": len(norm), "audit": audit,
                  "normalized_sha256": hashlib.sha256(norm.encode()).hexdigest(),
                  "clause_derivations": [c.__dict__ for c in used], "canonical_facts": facts,
                  "canonical_fact_uses": uses,
                  "whole_text_grammar_licensed": True, "connected_fact_graph": True,
                  "predicate_phrase_reuse": phrase_audit,
                  "reuse_policy": "Same clause mode plus fact banned; question-to-statement use allowed and reported. Distinct roles do not establish coherent context or truth.",
                  "inherited_reference_match": norm in reference_tapes,
                  "novelty": "reference match" if norm in reference_tapes else "not established",
                  "meaning_evidence": "not_evaluated"}
        admitted.add(norm)
        counts["unique_admitted_exact_closures"] += 1
        if longest is None or len(norm) > longest["letters"]:
            longest = record
        n = counts["unique_admitted_exact_closures"]
        if len(samples) < 100:
            samples.append(record)
        else:
            position = rng.randrange(n)
            if position < 100:
                samples[position] = record
        return True

    admitted = set()

    def allow_state(left, right):
        nonlocal deepest
        counts["candidate_states"] += 1
        letters = sum(map(len, left + right))
        if letters > max_letters:
            counts["length_rejections"] += 1
            return False
        lf, rf = left_frontier.states(left), right_frontier.states(right[::-1])
        if not lf or not rf:
            counts["prefix_or_suffix_grammar_rejections"] += 1
            return False
        if not any(not a.intersection(b) for _, a in lf for _, b in rf):
            counts["cross_frontier_completed_fact_rejections"] += 1
            return False
        counts["grammar_supported_states"] += 1
        if deepest is None or letters > deepest["letters"]:
            deepest = {"left": list(left), "right": list(right), "letters": letters,
                       "left_nfa_states": len(lf), "right_nfa_states": len(rf),
                       "status": "unfinished_frontiers; not an admitted output"}
        # Audit child closures before engine beam/step/deadline truncation.
        closure(left, right)
        return True

    vocabulary = sorted({w for c in clauses for w in c.words})
    engine_selected = beam_search(WordTries(vocabulary), JoinScorer(model), min_letters=min_letters,
                                  beam_width=beam_width, max_steps=max_steps, candidate_limit=len(vocabulary),
                                  seed=seed, allow_state=allow_state, allow_closed=closure,
                                  deadline=deadline, max_word_uses=None,
                                  max_overhang=max_overhang, closed_stop_factor=None)
    return {"min_letters": min_letters, "max_letters": max_letters, "seed": seed,
            "max_overhang": max_overhang, "closed_stop_factor": None, "max_word_uses": None,
            "seconds_budget": seconds, "elapsed_seconds": time.monotonic() - start,
            "status": "deadline" if time.monotonic() >= deadline else "engine_returned",
            "counters": dict(counts), "deepest_prefix": deepest, "longest_exact": longest,
            "engine_selected_words": engine_selected, "candidates": samples,
            "sample_policy": "all admitted until100, then deterministic reservoir; all unique admissions counted",
            "meaning_evidence": "not_evaluated"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--bigram-file", type=Path, default=ROOT / "data/count_2w.txt")
    ap.add_argument("--beam-width", type=int, default=384)
    ap.add_argument("--max-letters", type=int, default=2500)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--max-overhang", type=int, default=96)
    args = ap.parse_args()
    if not 0 < args.seconds <= 300 or min(args.beam_width, args.max_steps, args.max_overhang) < 1 or args.max_letters < 320:
        raise ValueError("require seconds in (0,300], positive beam/steps/overhang and max-letters>=320")
    if args.out.exists():
        raise FileExistsError(args.out)
    clauses = scene_clauses()
    model = BigramModel.from_file(str(args.bigram_file), vocab={w for c in clauses for w in c.words})
    paths = [Path(__file__), ROOT / "experiments/argonne_typed_overhang_growth_20261009.py",
             ROOT / "llm_palindrome/search.py", args.bigram_file]
    result = {"experiment": "argonne_free_typed_clauses_20261009", "status": "running", "planned_jobs": 4,
              "config": {k: v for k, v in vars(args).items() if k not in ("out", "bigram_file")},
              "engine_policy": {"closed_stop_factor": None, "max_word_uses": None},
              "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              "grammar_sha256": hashlib.sha256(json.dumps([c.__dict__ for c in clauses], sort_keys=True).encode()).hexdigest(),
              "grammar_clauses": len(clauses), "results": [],
              "limits": ["No frozen seed; existing reference tapes are never claimed novel.",
                         "Whole grammar, unique mode/fact uses and a connected entity graph do not establish meaning.",
                         "Same question or assertion may not repeat; question-to-statement role change is allowed and audited, not meaning-certified.",
                         "Local bigram ranks explore; no global word reuse limit; no model meaning score.",
                         "Overhang cap is explicit and configurable; incumbent length early stopping is disabled.",
                         "Bounded search failure is not an UNSAT proof."]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    def save():
        tmp = args.out.with_suffix(args.out.suffix + ".tmp")
        tmp.write_text(json.dumps(result, indent=2) + "\n")
        tmp.replace(args.out)
    save()
    for minimum in (30, 80, 160, 320):
        try:
            row = run_job(clauses, model, min_letters=minimum, seconds=args.seconds, seed=args.seed,
                          beam_width=args.beam_width, max_letters=args.max_letters, max_steps=args.max_steps,
                          max_overhang=args.max_overhang)
        except Exception as exc:
            row = {"min_letters": minimum, "status": "execution_error", "error": f"{type(exc).__name__}: {exc}", "candidates": []}
        result["results"].append(row)
        save()
        print(json.dumps({"min_letters": minimum, "status": row["status"], "counters": row.get("counters")}), flush=True)
    result["status"] = "completed_diagnostic"
    save()


if __name__ == "__main__":
    main()
