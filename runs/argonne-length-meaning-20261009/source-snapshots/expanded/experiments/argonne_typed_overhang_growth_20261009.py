"""Paired lexical versus contextual-grammar repayment around exact short seeds.

The new operator carries a clause-language state on each outer frontier. A word
must repay the live character debt AND leave a grammatical continuation on the
opposite frontier. Subject/object types, agreement, and fact identities come
from an explicit document-office scene. Entity reuse is allowed. This grammar
licenses possible meanings; it does not establish human readability or truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.bigram import BigramModel
from llm_palindrome.centerout import consume_suffix
from llm_palindrome.search import WordTries, consume, unit_letters

CENTERS = {
    "38-control": "An aide rips nine memos; some men inspire Diana.",
    "54-np-candidate": "An aide rips nine memo-hero memos. Some more home men inspire Diana.",
}
SEED_FACTS = (("aide", "rip", "nine_memos"), ("men", "inspire", "diana"))


def tape(text):
    return re.sub("[^a-z]", "", text.lower())


@dataclass(frozen=True)
class Clause:
    text: str
    facts: tuple[tuple[str, str, str], ...]
    frame: str
    mode: str = "statement"

    @property
    def words(self):
        return tuple(self.text.lower().split())

    def render(self):
        return self.text[:1].upper() + self.text[1:] + ("?" if self.mode == "question" else ".")


def scene_clauses() -> list[Clause]:
    """Finite typed productions with surface variants sharing fact identities.

    No palindrome or reverse-string lookup shapes this inventory. Passive and
    question alternations change boundary options without changing the event's
    participants. The 54-letter seed is retained as an unvalidated control.
    """
    people = [
        ("Diana", "diana", False), ("an aide", "aide", False),
        ("some men", "men", True), ("the men", "men", True),
        ("an editor", "editor", False), ("a reader", "reader", False),
        ("Ada", "ada", False), ("Eve", "eve", False),
        ("Otto", "otto", False), ("Iris", "iris", False),
    ]
    documents = [
        ("a memo", "memo", False), ("nine memos", "nine_memos", True),
        ("some notes", "notes", True), ("a letter", "letter", False),
        ("a report", "report", False), ("a poem", "poem", False),
        ("the news", "news", False), ("a notice", "notice", False),
    ]
    verbs = [
        ("read", "reads", "read"), ("write", "writes", "written"),
        ("edit", "edits", "edited"), ("copy", "copies", "copied"),
        ("file", "files", "filed"), ("review", "reviews", "reviewed"),
        ("quote", "quotes", "quoted"), ("rip", "rips", "ripped"),
    ]
    clauses = []
    for subject, sid, plural in people:
        for obj, oid, obj_plural in documents:
            for base, singular, participle in verbs:
                facts = ((sid, base, oid),)
                clauses.extend([
                    Clause(f"{subject} {base if plural else singular} {obj}", facts, "person_action_document"),
                    Clause(f"{obj} {'are' if obj_plural else 'is'} {participle} by {subject}", facts, "document_passive_person"),
                    Clause(f"{'do' if plural else 'does'} {subject} {base} {obj}", facts, "question_person_document", "question"),
                ])
        for other, oid, _ in people:
            if sid != oid:
                for base, singular in (("inspire", "inspires"), ("help", "helps"), ("meet", "meets")):
                    clauses.append(Clause(f"{subject} {base if plural else singular} {other}",
                                          ((sid, base, oid),), "person_action_person"))
    # Two linked events with one accessible referent; this changes clause
    # topology, rather than repeating the old single finite SVO shell.
    for subject, sid, plural in people:
        for first, fid, _ in documents[:4]:
            for second, oid, _ in documents[4:]:
                clauses.append(Clause(
                    f"{subject} who {'read' if plural else 'reads'} {first} {'review' if plural else 'reviews'} {second}",
                    ((sid, "read", fid), (sid, "review", oid)), "relative_shared_agent"))
    # Documents may name/credit people, but never perform the person-only acts.
    for subject, sid, plural in documents:
        for obj, oid, _ in people:
            for base, singular in (("name", "names"), ("credit", "credits"), ("describe", "describes")):
                clauses.append(Clause(f"{subject} {base if plural else singular} {obj}",
                                      ((sid, base, oid),), "document_reference_person"))
    return list({(c.words, c.facts, c.mode): c for c in clauses}.values())


class Grammar:
    def __init__(self, clauses: list[Clause], reverse: bool = False):
        self.clauses = clauses
        self.reverse = reverse
        self.edges: list[dict[str, int]] = [{}]
        self.ends: list[list[int]] = [[]]
        for cid, clause in enumerate(clauses):
            node = 0
            words = clause.words[::-1] if reverse else clause.words
            for word in words:
                if word not in self.edges[node]:
                    self.edges[node][word] = len(self.edges)
                    self.edges.append({})
                    self.ends.append([])
                node = self.edges[node][word]
            self.ends[node].append(cid)

    def advances(self, node: int, word: str):
        if node < 0:
            return ()
        out = []
        if word in self.edges[node]:
            out.append((self.edges[node][word], None))
        if self.ends[node] and word in self.edges[0]:
            out.extend((self.edges[0][word], cid) for cid in self.ends[node])
        return tuple(out)

    def possible_words(self, node: int):
        if node < 0:
            return ()
        return tuple(sorted(set(self.edges[node]) | (set(self.edges[0]) if self.ends[node] else set())))

    def complete(self, node):
        return node == 0 or (node > 0 and bool(self.ends[node]))

    def parse(self, words: tuple[str, ...]) -> tuple[int, ...] | None:
        """Parse final ordinary reading order using complete clause boundaries."""
        if self.reverse:
            raise ValueError("parse requires the forward grammar")
        @lru_cache(None)
        def visit(start):
            if start == len(words):
                return ()
            node = 0
            for end in range(start, len(words)):
                node = self.edges[node].get(words[end], -1)
                if node < 0:
                    return None
                for cid in self.ends[node]:
                    rest = visit(end + 1)
                    if rest is not None:
                        return (cid,) + rest
            return None
        return visit(0)


class ContextSupport:
    """Sound bounded grammar/debt rejection: horizon exhaustion stays unknown."""
    def __init__(self, forward: Grammar, backward: Grammar, max_depth: int = 6):
        self.forward, self.backward, self.max_depth = forward, backward, max_depth

    @lru_cache(maxsize=200000)
    def check(self, side: str, node: int, debt: str, depth: int):
        if not debt:
            return "supported"
        if depth == 0:
            return "unknown"
        grammar = self.forward if side == "R" else self.backward
        unknown = False
        for word in grammar.possible_words(node):
            match = consume(tape(word), debt) if side == "R" else consume_suffix(tape(word), debt)
            if match is None:
                continue
            remainder, flipped = match
            if not remainder or flipped:
                return "supported"
            for child, _ in grammar.advances(node, word):
                verdict = self.check(side, child, remainder, depth - 1)
                if verdict == "supported":
                    return verdict
                unknown |= verdict == "unknown"
        return "unknown" if unknown else "impossible"


def novelty_cost(clause: Clause, prior_facts, prior_frames):
    """Repeated actors cost nothing; repeated propositions and frames do."""
    fact_counts = Counter(prior_facts)
    repeated = sum(fact_counts[fact] for fact in clause.facts)
    frame_repeats = Counter(prior_frames)[clause.frame]
    return 12.0 * repeated + 2.5 * frame_repeats


def repeats_fact(clause: Clause, prior_facts):
    counts = Counter((*prior_facts, *clause.facts))
    return any(counts[fact] > 1 for fact in clause.facts)


def predicate_reuse_audit(clauses):
    """Audit verb-anchored spans; noun phrases/function words are exempt."""
    forms = {"write": {"write", "writes", "written"}, "copy": {"copy", "copies", "copied"},
             "rip": {"rip", "rips", "ripped"}, "describe": {"describe", "describes", "described"}}
    spans = {}
    for ci, clause in enumerate(clauses):
        words = clause.words
        starts = []
        for i, word in enumerate(words):
            bindings = tuple(f for f in clause.facts if word in forms.get(f[1], {f[1], f[1] + "s", f[1] + "ed"}))
            if bindings:
                starts.append((i, bindings))
        for j, (start, bindings) in enumerate(starts):
            end = starts[j + 1][0] if j + 1 < len(starts) else len(words)
            span = words[start:end]
            if len(span) >= 2:
                spans.setdefault(span, []).append({"clause_index": ci, "canonical_fact_bindings": bindings})
    return [{"predicate_span": " ".join(span), "uses": uses,
             "licensed_distinct_use": len({tuple(u["canonical_fact_bindings"]) for u in uses}) == len(uses)}
            for span, uses in sorted(spans.items()) if len(uses) > 1]


@dataclass
class State:
    left: tuple[str, ...] = ()
    right: tuple[str, ...] = ()
    debt: str = ""
    owner: str = "R"
    left_node: int = 0
    right_node: int = 0
    facts: tuple = SEED_FACTS
    frames: tuple = ()
    score: float = 0.0
    letters: int = 0


def exact_audit(text):
    norm = tape(text)
    mismatch = next((i for i in range(len(norm) // 2) if norm[i] != norm[-1 - i]), None)
    return {"letters": len(norm), "exact": bool(norm) and mismatch is None,
            "first_mismatch": mismatch,
            "sha256_forward": hashlib.sha256(norm.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(norm[::-1].encode()).hexdigest()}


def describe(state: State, center: str, grammar: Grammar):
    lp, rp = grammar.parse(state.left), grammar.parse(state.right)
    complete = lp is not None and rp is not None
    seed_parse = grammar.parse(tuple(re.findall("[a-z]+", center.lower())))
    cids = (lp or ()) + (rp or ())
    clauses = [grammar.clauses[cid] for cid in cids]
    facts = [fact for clause in clauses for fact in clause.facts]
    # Seed facts are retained only as reuse references, not generated content.
    fact_counts = Counter(SEED_FACTS)
    repeated_facts = 0
    for fact in facts:
        repeated_facts += fact_counts[fact] > 0
        fact_counts[fact] += 1
    frames = [c.frame for c in clauses]
    full_clauses = [*(grammar.clauses[c] for c in (lp or ())),
                    *(grammar.clauses[c] for c in (seed_parse or ())),
                    *(grammar.clauses[c] for c in (rp or ()))]
    if complete:
        text = " ".join([*(grammar.clauses[c].render() for c in lp), center,
                         *(grammar.clauses[c].render() for c in rp)])
    else:
        text = " ".join([" ".join(state.left), center, " ".join(state.right)]).strip()
    audit = exact_audit(text)
    return {
        "text": text, "audit": audit, "letters": audit["letters"],
        "growth_letters": audit["letters"] - len(tape(center)),
        "complete_clause_extensions": complete,
        "seed_grammar_licensed": seed_parse is not None,
        "whole_text_grammar_licensed": complete and seed_parse is not None,
        "clause_derivations": [{"text": c.text, "facts": c.facts, "frame": c.frame, "mode": c.mode} for c in clauses],
        "added_distinct_facts": len(set(facts) - set(SEED_FACTS)),
        "repeated_fact_occurrences": repeated_facts,
        "repeated_frame_occurrences": len(frames) - len(set(frames)),
        "predicate_phrase_reuse": predicate_reuse_audit(full_clauses),
        "predicate_phrase_reuse_scope": "parsed extensions plus seed when grammar licensed; unparsed spans unassessed",
        "reuse_policy": "Natural entity reuse allowed; identical canonical facts banned in typed admission, including paraphrases. Repeated predicates require distinct fact bindings; noun and function phrases exempt. These checks do not establish whole-text meaning.",
        "claim_scope": "role-compatible scene descriptions; grammar and human meaning unvalidated",
    }


def search_job(center, arm, clauses, model, *, seconds=120, seed=0, min_letters=120,
               max_letters=1000, beam_width=96, max_steps=500):
    if not exact_audit(center)["exact"]:
        raise ValueError("center must be exact")
    if arm not in ("lexical_debt", "typed_context_debt"):
        raise ValueError("unknown arm")
    typed = arm == "typed_context_debt"
    started = time.monotonic()
    deadline = started + seconds
    forward, backward = Grammar(clauses), Grammar(clauses, reverse=True)
    support = ContextSupport(forward, backward)
    vocabulary = sorted({w for c in clauses for w in c.words})
    tries = WordTries(vocabulary)
    center_words = re.findall("[a-z]+", center.lower())
    rng = random.Random(seed)
    counters = Counter()
    first = State(letters=len(tape(center)))
    beam, selected, longest = [first], None, None
    examples = []
    closed_surfaces = set()
    rejected_surfaces = set()

    def record_closure(state):
        nonlocal selected, longest, examples
        if state.debt or state.letters <= first.letters:
            return
        counters["exact_closure_states_seen"] += 1
        signature = (state.left, state.right)
        if signature in rejected_surfaces:
            return False
        if signature in closed_surfaces:
            return
        closed_surfaces.add(signature)
        counters["exact_closures_seen"] += 1
        record = describe(state, center, forward)
        if not record["audit"]["exact"]:
            raise AssertionError("independent exact audit disagrees with live debt")
        grammar_ok = record["complete_clause_extensions"]
        counters["complete_clause_closures_seen"] += grammar_ok
        if typed and grammar_ok and record["repeated_fact_occurrences"]:
            counters["repeated_fact_closure_rejections"] += 1
            rejected_surfaces.add(signature)
            return False
        if longest is None or record["letters"] > longest["letters"]:
            longest = record
        if record["letters"] >= min_letters and (not typed or grammar_ok):
            counters["target_closures_seen"] += 1
            # Selection inspects both full extensions, not merely the newly
            # completed local clause. Record before beam/deadline truncation.
            key = (int(record["whole_text_grammar_licensed"]), int(grammar_ok), record["added_distinct_facts"],
                   -record["repeated_fact_occurrences"], -record["repeated_frame_occurrences"], record["letters"])
            if selected is None or key > selected[0]:
                selected = (key, record)
                examples.append(record)
                examples = examples[-8:]

    preflight = []
    for word in backward.possible_words(0):
        verdict = support.check("R", 0, tape(word)[::-1], support.max_depth)
        preflight.append({"left_opening_word": word, "right_context_repayment": verdict})
    for step in range(max_steps):
        if not beam or time.monotonic() >= deadline:
            break
        pool = []
        for state in beam:
            if time.monotonic() >= deadline:
                break
            counters["expanded_states"] += 1
            side = "L" if not state.debt else state.owner
            g = backward if side == "L" else forward
            node = state.left_node if side == "L" else state.right_node
            if not state.debt:
                words = vocabulary
            elif side == "R":
                words = tries.left_candidates(state.debt, len(vocabulary))
            else:
                words = tries.right_candidates(state.debt[::-1], len(vocabulary))
            local = []
            for word in words:
                if time.monotonic() >= deadline:
                    break
                counters["candidate_words"] += 1
                letters = tape(word)
                if state.letters + len(letters) > max_letters:
                    counters["length_rejections"] += 1
                    continue
                if not state.debt:
                    rem, owner = letters[::-1], "R"
                else:
                    match = consume(letters, state.debt) if side == "R" else consume_suffix(letters, state.debt)
                    if match is None:
                        counters["character_rejections"] += 1
                        continue
                    rest, flipped = match
                    rem = rest[::-1] if flipped else rest
                    owner = ("L" if side == "R" else "R") if flipped else side
                if len(rem) > 48:
                    counters["overhang_rejections"] += 1
                    continue
                advances = g.advances(node, word)
                if not advances:
                    counters["grammar_incompatible_words"] += 1
                    if typed:
                        continue
                    advances = ((-1, None),)
                for child_node, completed in advances:
                    ln = child_node if side == "L" else state.left_node
                    rn = child_node if side == "R" else state.right_node
                    if typed and rem:
                        verdict = support.check(owner, rn if owner == "R" else ln, rem, support.max_depth)
                        counters[f"context_repayment_{verdict}"] += 1
                        if verdict == "impossible":
                            continue
                    left = (word,) + state.left if side == "L" else state.left
                    right = state.right + (word,) if side == "R" else state.right
                    neighbor = (state.left[0] if state.left else center_words[0]) if side == "L" else (state.right[-1] if state.right else center_words[-1])
                    delta = model.forward(word, neighbor) if side == "L" else model.forward(neighbor, word)
                    # The shared lexical score only ranks local joins; it is
                    # never reported as meaning. No entity-repeat penalty.
                    facts, frames = state.facts, state.frames
                    if completed is not None:
                        clause = clauses[completed]
                        if typed and repeats_fact(clause, facts):
                            counters["repeated_fact_transition_rejections"] += 1
                            continue
                        delta -= novelty_cost(clause, facts, frames)
                        facts += clause.facts
                        frames += (clause.frame,)
                    score = state.score + delta
                    child = State(left, right, rem, owner, ln, rn, facts, frames, score, state.letters + len(letters))
                    # Normalize join cost by generated words; slight debt cost
                    # breaks otherwise equivalent unfinished language states.
                    priority = score / max(1, len(left) + len(right)) - .03 * len(rem)
                    if record_closure(child) is False:
                        continue
                    local.append((priority, rng.random(), child))
            local.sort(key=lambda x: (-x[0], x[1]))
            pool.extend(local[:12])
        # Distinct cursors/debts remain separate even with equal token tapes.
        pool.sort(key=lambda x: (-x[0], x[1]))
        seen, beam = set(), []
        for _, _, state in pool:
            key = (state.left, state.right, state.left_node, state.right_node)
            if key not in seen:
                seen.add(key)
                beam.append(state)
                if len(beam) >= beam_width:
                    break
    timed_out = time.monotonic() >= deadline
    return {
        "arm": arm, "seed": seed, "min_letters": min_letters, "max_letters": max_letters,
        "beam_width": beam_width, "max_steps": max_steps,
        "status": ("deadline" if timed_out else "search_exhausted" if not beam else "step_limit") + ("_with_output" if selected else "_no_output"),
        "seconds_budget": seconds, "elapsed_seconds": time.monotonic() - started,
        "selected": selected[1] if selected else None, "longest_exact": longest,
        "whole_text_grammar_selected": selected[1] if selected and selected[1]["whole_text_grammar_licensed"] else None,
        "selection_scope": "selected may validate only extensions; whole_text_grammar_selected additionally requires the fixed seed to parse; neither is reader-certified",
        "counters": dict(counters), "frontier_states": len(beam),
        "preflight": {"opening_words": len(preflight), "supported_or_unknown": sum(r["right_context_repayment"] != "impossible" for r in preflight), "rows": preflight},
        "progress_examples": examples,
        "grammar_inventory": {"clauses": len(clauses), "words": len(vocabulary),
                              "forward_states": len(forward.edges), "backward_states": len(backward.edges)},
        "human_meaning_evidence": "not_evaluated",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--min-letters", type=int, default=120)
    ap.add_argument("--max-letters", type=int, default=1000)
    ap.add_argument("--beam-width", type=int, default=96)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--bigram-file", type=Path, default=ROOT / "data/count_2w.txt")
    args = ap.parse_args()
    if args.seconds <= 0 or args.seconds > 300:
        raise ValueError("seconds must be in (0, 300] per paired job")
    if not 0 < args.min_letters <= args.max_letters or args.beam_width <= 0 or args.max_steps <= 0:
        raise ValueError("require 0 < min-letters <= max-letters and positive beam-width/max-steps")
    if args.out.exists():
        raise FileExistsError(args.out)
    clauses = scene_clauses()
    words = {w for c in clauses for w in c.words}
    words.update(w for center in CENTERS.values() for w in re.findall("[a-z]+", center.lower()))
    model = BigramModel.from_file(str(args.bigram_file), vocab=words)
    result = {
        "experiment": "argonne-typed-overhang-growth-20261009", "status": "running",
        "hypothesis": "typed clause context and opposite-language debt support can preserve recoverable event relations during exact extension",
        "planned_jobs": 4, "results": [],
        "config": {k: getattr(args, k) for k in ("seconds", "min_letters", "max_letters", "beam_width", "max_steps", "seed")},
        "seeds": CENTERS,
        "seed_54_status": "deliberate exact but grammar-unlicensed control; never promoted to whole-text grammar evidence",
        "limits": ["Both exact centers are reference seeds, not new output; the unusual 54-letter wording is not meaning-certified.",
                   "Typed lexical roles license possible events, not world truth, discourse quality, or reader acceptance.",
                   "Local join counts may overlap existing search inputs; this is not a held-out language evaluation.",
                   "A bounded beam failure is not an UNSAT proof. Only an impossible repayment is pruned; lookahead horizon remains unknown.",
                   "Natural entity reuse is allowed. The typed arm rejects identical canonical facts, including paraphrases; the lexical baseline retains its earlier penalty. Repeated frames carry a penalty.",
                   "The paired arms change grammar/context admission together; this does not isolate their individual causal effects."],
        "difference_from_prior": {"reviewed": ["experiments/working_overhang_coherent_20260921.py", "experiments/working_overhang_phrase_20260921.py", "experiments/grammar_first_centerout_20260921.py", "docs/research/2026-10-08-openai-math.md"],
                                  "new_operator": "two clause-language cursors plus bounded opposite-context repayment support; shared entities and canonical facts across active/passive/question/relative realizations",
                                  "avoids": "center-word bans, fixed endpoint-unsatisfiable shells, closed-clause mirror lookup, copied finished tapes, posthoc letter repair"},
        "provenance": {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "bigram_sha256": hashlib.sha256(args.bigram_file.read_bytes()).hexdigest(),
                       "grammar_sha256": hashlib.sha256(json.dumps([c.__dict__ for c in clauses], sort_keys=True).encode()).hexdigest()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    def save():
        tmp = args.out.with_suffix(args.out.suffix + ".tmp")
        tmp.write_text(json.dumps(result, indent=2) + "\n")
        tmp.replace(args.out)
    save()
    jobs = [(cid, arm) for cid in CENTERS for arm in ("lexical_debt", "typed_context_debt")]
    random.Random(args.seed).shuffle(jobs)
    for cid, arm in jobs:
        try:
            row = search_job(CENTERS[cid], arm, clauses, model, seconds=args.seconds, seed=args.seed,
                             min_letters=args.min_letters, max_letters=args.max_letters,
                             beam_width=args.beam_width, max_steps=args.max_steps)
        except Exception as exc:
            row = {"arm": arm, "seed": args.seed, "status": "execution_error", "error": f"{type(exc).__name__}: {exc}", "selected": None, "longest_exact": None, "whole_text_grammar_selected": None}
        row["center_id"] = cid
        result["results"].append(row)
        save()
        print(json.dumps({"center_id": cid, "arm": arm, "status": row["status"], "letters": (row.get("selected") or {}).get("letters")}), flush=True)
    result["status"] = "completed_diagnostic"
    result["summary"] = {"planned_jobs": 4, "recorded_jobs": len(result["results"]),
                         "jobs_with_selected_output": sum(r["selected"] is not None for r in result["results"]),
                         "jobs_with_whole_text_grammar_selected": sum(r["whole_text_grammar_selected"] is not None for r in result["results"]),
                         "human_meaning_evidence": "not_evaluated; no output promoted as readable",
                         "execution_errors": sum(r["status"] == "execution_error" for r in result["results"])}
    save()


if __name__ == "__main__":
    main()
