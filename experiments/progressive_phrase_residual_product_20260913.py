"""Progressive phrase-residual construction over typed intact clauses.

Typed initial and terminal phrases are indexed by three real characters.  A
joined derivation is then expanded inward one character pair at a time; the
state explicitly carries the live lexical residual on both sides and role
transitions occur only at completed terminals.  No finished sentence is
filtered for a short boundary and then emitted: the whole connected clause is
replayed and independently parsed at closure.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.whole_text_palindrome_product_20260913 import Grammar, compile_slots
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

MIN_LETTERS, MAX_LETTERS = 30, 160
WORD_RE = re.compile(r"[a-z]+")


@dataclass(frozen=True)
class Lexeme:
    form: str
    category: str
    semantic: str
    subject_types: tuple[str, ...] = ()
    object_types: tuple[str, ...] = ()


@dataclass(frozen=True)
class Frame:
    identifier: str
    kind: str
    subject: Lexeme | None
    subject_adj: Lexeme | None
    verb: Lexeme
    object: Lexeme
    object_adj: Lexeme
    place: Lexeme | None = None
    place_adj: Lexeme | None = None


def lex(form: str, category: str, semantic: str, *, subject_types: tuple[str, ...] = (), object_types: tuple[str, ...] = ()) -> Lexeme:
    return Lexeme(form, category, semantic, subject_types, object_types)


PERSONS = tuple(lex(word, "noun", "person") for word in ("artist", "captain", "editor", "teacher", "tourist"))
ARTIFACTS = tuple(lex(word, "noun", "artifact") for word in ("canvas", "letter", "model", "report"))
PLACES = tuple(lex(word, "noun", "place") for word in ("arena", "garden", "office", "studio"))
PERSON_ADJS = tuple(lex(word, "adjective", "person") for word in ("agile", "calm", "eager", "patient", "skilled"))
OBJECT_ADJS = tuple(lex(word, "adjective", "artifact") for word in ("brief", "clean", "detailed", "solid"))
PLACE_OBJECT_ADJS = tuple(lex(word, "adjective", "place") for word in ("quiet", "public", "remote", "small"))
QUIET = next(item for item in PLACE_OBJECT_ADJS if item.form == "quiet")
EAGER = next(item for item in PERSON_ADJS if item.form == "eager")
BRIEF = next(item for item in OBJECT_ADJS if item.form == "brief")
VISITED = lex("visited", "past_verb", "transitive", subject_types=("person",), object_types=("place",))
PAST_ACTIONS = (
    lex("built", "past_verb", "transitive", subject_types=("person",), object_types=("artifact",)),
    lex("drafted", "past_verb", "transitive", subject_types=("person",), object_types=("artifact",)),
    lex("painted", "past_verb", "transitive", subject_types=("person",), object_types=("artifact",)),
    lex("repaired", "past_verb", "transitive", subject_types=("person",), object_types=("artifact",)),
)
BASE_ACTIONS = tuple(lex(word, "base_verb", "transitive", subject_types=("person",), object_types=("artifact",))
                    for word in ("build", "draft", "paint", "repair"))
PLACE_ADJS = PLACE_OBJECT_ADJS


def article(adj: Lexeme) -> Lexeme:
    return lex("an" if adj.form[0] in "aeiou" else "a", "determiner", "function")


@dataclass(frozen=True)
class Derivation:
    frame: Frame
    roles: tuple[str, ...]
    choices: tuple[tuple[Lexeme, ...], ...]

    @property
    def identifier(self) -> str:
        return self.frame.identifier

    @property
    def slots(self) -> tuple[tuple[str, ...], ...]:
        return tuple(tuple(item.form for item in choices) for choices in self.choices)


def svo_frames() -> tuple[Frame, ...]:
    # The endpoint algebra sees only these broad role inventories.  No phrase
    # is selected for a matching signature in advance; all verb/object pairs
    # are generated from their frame-specific valency relation.
    location_verbs = tuple(lex(word, "past_verb", "transitive", subject_types=("person",), object_types=("place",))
                           for word in ("entered", "toured", "visited"))
    artifact_verbs = tuple(lex(word, "past_verb", "transitive", subject_types=("person",), object_types=("artifact",))
                           for word in ("built", "drafted", "painted", "repaired"))
    rows = []
    for person in PERSONS:
        for person_adj in PERSON_ADJS:
            for obj in PLACES + ARTIFACTS:
                object_adjs = PLACE_OBJECT_ADJS if obj.semantic == "place" else OBJECT_ADJS
                verbs = location_verbs if obj.semantic == "place" else artifact_verbs
                for object_adj in object_adjs:
                    for verb in verbs:
                        rows.append(Frame(f"svo-{person.form}-{person_adj.form}-{verb.form}-{obj.form}-{object_adj.form}",
                                          "svo_typed", person, person_adj, verb, obj, object_adj))
    return tuple(rows)


def imperative_frames() -> tuple[Frame, ...]:
    return tuple(Frame(f"imperative-{verb.form}-{obj.form}-{object_adj.form}-{place.form}-{place_adj.form}",
                       "imperative_locative", None, None, verb, obj, object_adj, place, place_adj)
                 for verb in BASE_ACTIONS for obj in ARTIFACTS
                 for object_adj in OBJECT_ADJS for place in PLACES
                 for place_adj in PLACE_ADJS)


def derivation_for(frame: Frame) -> Derivation:
    if frame.kind == "svo_typed":
        assert frame.subject and frame.subject_adj
        return Derivation(frame, ("subject_det", "subject_adj", "subject", "verb", "object_det", "object_adj", "object"), (
            (article(frame.subject_adj),), (frame.subject_adj,), (frame.subject,), (frame.verb,),
            (article(frame.object_adj),), (frame.object_adj,), (frame.object,),
        ))
    assert frame.place and frame.place_adj
    return Derivation(frame, ("verb", "object_det", "object_adj", "object", "prep", "place_det", "place_adj", "place"), (
        (frame.verb,), (article(frame.object_adj),), (frame.object_adj,), (frame.object,),
        (lex("in", "preposition", "function"),), (article(frame.place_adj),), (frame.place_adj,), (frame.place,),
    ))


DERIVATIONS = tuple(derivation_for(frame) for frame in svo_frames() + imperative_frames())


def grammar_for(derivation: Derivation) -> Grammar:
    return compile_slots(derivation.slots)


def letters(words: tuple[str, ...]) -> str:
    return "".join(words)


def endpoint_realizations(derivation: Derivation) -> tuple[dict, ...]:
    """Enumerate every legal lexical path, retaining its chosen words."""
    rows = []
    for words in product(*derivation.choices):
        words = tuple(item.form for item in words)
        if derivation.frame.kind == "svo_typed":
            opening, final = words[:3], words[4:]
            rows.extend((
                {"derivation": derivation.identifier, "kind": derivation.frame.kind, "side": "opening",
                 "words": opening, "letters": letters(opening), "role": derivation.roles[0]},
                {"derivation": derivation.identifier, "kind": derivation.frame.kind, "side": "terminal",
                 "words": final, "letters": letters(final), "role": derivation.roles[-1], "surface_words": words},
            ))
        else:
            opening, final = (words[0],), words[4:]
            rows.extend((
                {"derivation": derivation.identifier, "kind": derivation.frame.kind, "side": "opening",
                 "words": opening, "letters": letters(opening), "role": derivation.roles[0], "surface_words": words},
                {"derivation": derivation.identifier, "kind": derivation.frame.kind, "side": "terminal",
                 "words": final, "letters": letters(final), "role": derivation.roles[-1], "surface_words": words},
            ))
    return tuple(rows)


def endpoint_algebra(derivations: tuple[Derivation, ...] = DERIVATIONS) -> dict:
    opening_index, reverse_suffix_index = defaultdict(list), defaultdict(list)
    for d in derivations:
        for realization in endpoint_realizations(d):
            if realization["side"] == "opening":
                opening_index[(d.frame.kind, realization["letters"][:3])].append(realization)
            else:
                reverse_suffix_index[(d.frame.kind, realization["letters"][::-1][:3])].append(realization)
    joins = []
    eligible = set()
    for d in derivations:
        openings = [r for r in endpoint_realizations(d) if r["side"] == "opening"]
        for opening in openings:
            key = (d.frame.kind, opening["letters"][:3])
            for ending in reverse_suffix_index.get(key, ()):
                if ending["derivation"] != d.identifier: continue
                matched = 0
                for left, right in zip(opening["letters"], ending["letters"][::-1]):
                    if left != right: break
                    matched += 1
                if matched < 3: continue
                eligible.add(d.identifier)
                joins.append({"derivation": d.identifier, "kind": d.frame.kind,
                              "opening_words": list(opening["words"]), "terminal_words": list(ending["words"]),
                              "chosen_surface_words": list(opening.get("surface_words", ending.get("surface_words", ()))),
                              "opening_letters": opening["letters"], "reversed_terminal_letters": ending["letters"][::-1],
                              "matched_pairs": matched, "semantic_roles": [opening["role"], ending["role"]]})
    return {"minimum_matched_pairs": 3,
            "opening_index": {f"{k[0]}:{k[1]}": len(v) for k, v in opening_index.items()},
            "reverse_suffix_index": {f"{k[0]}:{k[1]}": len(v) for k, v in reverse_suffix_index.items()},
            "joins": joins, "eligible_derivation_ids": sorted(eligible),
            "join_scope": "within_derivation_only; no cross-derivation phrase composition"}


def _live_residual(prefix, suffix):
    left = []
    for edge in reversed(prefix):
        left.append(edge.char)
        if edge.completed_word is not None: break
    right = []
    for edge in reversed(suffix):
        right.append(edge.char)
        if edge.completed_word is not None: break
    return "".join(reversed(left)), "".join(right)


FUNCTION_WORDS = frozenset({"a", "an", "the", "in", "at", "to", "of", "and"})


def _slot_boundaries(grammar: Grammar) -> tuple[tuple[int, int], ...]:
    """Recover each slot's actual graph interval from terminal edges."""
    starts = [grammar.start]
    ends = []
    for slot in range(len(grammar.slots)):
        terminal_edges = [edge for edge in grammar.edges if edge.slot == slot]
        if not terminal_edges:
            raise AssertionError("compiled grammar slot has no terminal edge")
        end = terminal_edges[0].target
        ends.append(end)
        starts.append(end)
    return tuple(zip(starts[:-1], ends))


def online_span_loss(grammar: Grammar, left: int, right: int, *, initial: bool = False) -> dict | None:
    """Detect a forced proper palindromic island at the live graph frontier."""
    if initial:
        return None
    boundaries = _slot_boundaries(grammar)
    interior = [index for index, (start, end) in enumerate(boundaries)
                if start >= left and end <= right]
    if len(interior) >= 2:
        # Current branch grammars are per-derivation and therefore each slot
        # has one chosen word.  Verify that condition rather than silently
        # treating a choice menu as a forced island.
        words = [grammar.slots[index][0] for index in interior]
        if all(len(grammar.slots[index]) == 1 for index in interior):
            tape = "".join(words)
            if tape and tape == tape[::-1]:
                return {"reason": "forced_proper_palindromic_island", "slot_indices": interior,
                        "words": words, "normalized_span": tape, "word_count": len(words)}
    elif len(interior) == 1 and len(grammar.slots[interior[0]]) == 1:
        word = grammar.slots[interior[0]][0]
        if word == word[::-1] and word not in FUNCTION_WORDS:
            return {"reason": "palindromic_content_word_island", "slot_indices": interior,
                    "words": [word], "normalized_span": word, "word_count": 1}
    return None


def progressive_construct(grammar: Grammar, derivation: Derivation, max_states: int = 100_000) -> dict:
    outgoing, incoming = defaultdict(list), defaultdict(list)
    for edge in grammar.edges:
        outgoing[edge.source].append(edge); incoming[edge.target].append(edge)
    reachable_cache = {}
    def reachable(node):
        if node not in reachable_cache:
            reachable_cache[node] = {node}
            for edge in outgoing[node]: reachable_cache[node].update(reachable(edge.target))
        return reachable_cache[node]
    reachable(grammar.start)
    stack = [(grammar.start, grammar.end, (), (), ())]
    seen, records, ledgers = set(), [], []
    states = 0; deepest = {"length": 0}; online_span_pruned = 0; span_prune_events = []
    while stack and states < max_states:
        left, right, prefix, suffix, pairs = stack.pop()
        key = (left, right, tuple((e.source, e.target, e.char) for e in prefix), tuple((e.source, e.target, e.char) for e in suffix))
        if key in seen: continue
        seen.add(key); states += 1
        lr, rr = _live_residual(prefix, suffix)
        left_role = derivation.roles[0]
        if prefix:
            last = prefix[-1]
            if last.slot is not None:
                left_role = derivation.roles[min(last.slot + 1, len(derivation.roles)-1)]
            else:
                completed = [edge.slot for edge in prefix if edge.slot is not None]
                left_role = derivation.roles[min((completed[-1] + 1) if completed else 0, len(derivation.roles)-1)]
        right_role = derivation.roles[-1]
        for edge in reversed(suffix):
            if edge.slot is not None:
                right_role = derivation.roles[edge.slot]
                break
        ledger = {"length": len(pairs), "left_node": left, "right_node": right,
                  "left_residual": lr, "right_residual": rr,
                  "left_role": left_role, "right_role": right_role,
                  "pair_trace": [[i, c] for i, c in pairs]}
        ledgers.append(ledger)
        if ledger["length"] > deepest["length"]: deepest = ledger
        span_loss = online_span_loss(grammar, left, right, initial=(not prefix and not suffix))
        if span_loss is not None:
            online_span_pruned += 1
            if len(span_prune_events) < 100:
                span_prune_events.append({**ledger, "prune": span_loss})
            continue
        middles = [()] if left == right else []
        middles += [(edge,) for edge in outgoing[left] if edge.target == right]
        for middle in middles:
            path = prefix + middle + suffix
            cursor = grammar.start; words = []; valid = True
            for edge in path:
                if edge.source != cursor: valid = False; break
                cursor = edge.target
                if edge.completed_word is not None: words.append(edge.completed_word)
            if valid and cursor == grammar.end and len(words) == len(grammar.slots):
                tape = "".join(words)
                if tape == tape[::-1] and tuple(words) not in seen:
                    seen.add(tuple(words)); records.append({"words": words, "letters": len(tape),
                        "exact": True, "center_characters": len(middle), "midpoint_letter_offset": len(prefix),
                        "residual_ledger": ledger})
        for first in outgoing[left]:
            for last in incoming[right]:
                if first.char == last.char and last.source in reachable(first.target):
                    stack.append((first.target, last.source, prefix + (first,), (last,) + suffix, pairs + ((len(pairs)+1, first.char),)))
    return {"states": states, "pending_states": len(stack), "states_exhausted": not stack,
            "truncated": bool(stack), "deepest_matched_pairs": deepest["length"],
            "deepest_residual_ledger": deepest, "ledger_count": len(ledgers),
            "online_span_pruned": online_span_pruned, "span_prune_events": span_prune_events,
            "records": records}


def audit(derivation, grammar, words, rec, join):
    text = render(words); exact = {**exact_audit(text)}
    parsed = independent_parse(derivation.identifier, text); replay = replay_words(grammar, words)
    pair_replay = replay_pair_ledger(text, rec["residual_ledger"])
    central = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    codes = [k for k, v in central.items() if not v]
    if not exact["exact"]: codes.append("independent_exact_audit_failed")
    if not parsed["ok"]: codes.append("independent_complete_reparse_failed")
    if not replay["ok"]: codes.append("independent_terminal_path_replay_failed")
    if not pair_replay["ok"]: codes.append("independent_pair_ledger_replay_failed")
    return {"record_kind": "progressive_phrase_residual_closure", "derivation": derivation.identifier,
            "rendered": text, "endpoint_join": join, "independent_exact_audit": exact,
            "independent_parse": parsed, "independent_terminal_path_replay": replay,
            "independent_pair_ledger_replay": pair_replay,
            "residual_ledger": rec["residual_ledger"], "central_admission": central,
            "mechanically_admitted": not codes, "rejection_codes": codes,
            "reader_status": "human-unreviewed; programmatic checks do not certify readability"}


def exact_audit(text):
    tape = normalize_letters(text); mismatches = [(i, len(tape)-1-i) for i in range(len(tape)//2) if tape[i] != tape[-1-i]]
    return {"exact": bool(tape) and not mismatches, "letters": len(tape), "mismatches": mismatches,
            "normalized_sha256": sha256(tape.encode()).hexdigest()}


def replay_pair_ledger(text, ledger):
    """Independently replay the outside-in character ledger against the tape."""
    tape = normalize_letters(text)
    checks = []
    for emitted_index, char in ledger.get("pair_trace", ()):
        index = int(emitted_index) - 1
        if index < 0 or index >= len(tape) or index >= len(tape) - index - 1:
            return {"ok": False, "reason": "ledger_index_out_of_range", "emitted_index": emitted_index}
        checks.append({"emitted_index": emitted_index, "left": tape[index],
                      "right": tape[-index-1], "ledger_char": char,
                      "equal": tape[index] == tape[-index-1] == char})
    return {"ok": all(item["equal"] for item in checks), "pairs_checked": len(checks), "checks": checks}


def replay_words(grammar, words):
    cursor = grammar.start
    for slot, word in enumerate(words):
        if word not in grammar.slots[slot]: return {"ok": False, "reason": "word_not_in_slot"}
        for offset, char in enumerate(word):
            options = [e for e in grammar.edges if e.source == cursor and e.char == char]
            edge = next((e for e in options if offset < len(word)-1 or e.completed_word == word), None)
            if edge is None: return {"ok": False, "reason": "edge_mismatch"}
            cursor = edge.target
    return {"ok": cursor == grammar.end, "words": list(words)}


INDEPENDENT_SUBJECTS = frozenset({"artist", "captain", "editor", "teacher", "tourist"})
INDEPENDENT_ARTIFACTS = frozenset({"canvas", "letter", "model", "report"})
INDEPENDENT_PLACES = frozenset({"arena", "garden", "office", "studio", "theater"})
INDEPENDENT_PERSON_ADJECTIVES = frozenset({"able", "agile", "calm", "eager", "early", "patient", "skilled"})
INDEPENDENT_ARTIFACT_ADJECTIVES = frozenset({"brief", "clean", "detailed", "solid"})
INDEPENDENT_PLACE_ADJECTIVES = frozenset({"quiet", "public", "remote", "small"})
INDEPENDENT_VERB_FRAMES = {
    "built": {"artifact"}, "drafted": {"artifact"}, "painted": {"artifact"}, "repaired": {"artifact"},
    "entered": {"place"}, "toured": {"place"}, "visited": {"place"},
    "build": {"artifact"}, "draft": {"artifact"}, "paint": {"artifact"}, "repair": {"artifact"},
}


def independent_parse(derivation_id, text):
    tokens = tuple(WORD_RE.findall(text.lower()))
    kind = "svo_typed" if str(derivation_id).startswith("svo-") else "imperative_locative"
    if kind == "svo_typed" and len(tokens) != 7: return {"ok": False, "reason": "wrong_word_count"}
    if kind != "svo_typed" and len(tokens) != 8: return {"ok": False, "reason": "wrong_word_count"}
    if kind == "svo_typed":
        sd, sa, subj, verb, od, oa, obj = tokens
        agreement = sd in {"a", "an", "the"} and od in {"a", "an", "the"}
        phonology = (sd == "the" or sd == ("an" if sa[:1] in "aeiou" else "a"))
        phonology = phonology and (od == "the" or od == ("an" if oa[:1] in "aeiou" else "a"))
        subject_type = "person" if subj in INDEPENDENT_SUBJECTS else ""
        object_type = "place" if obj in INDEPENDENT_PLACES else "artifact" if obj in INDEPENDENT_ARTIFACTS else ""
        valid_object_adj = INDEPENDENT_PLACE_ADJECTIVES if object_type == "place" else INDEPENDENT_ARTIFACT_ADJECTIVES
        valency = (sa in INDEPENDENT_PERSON_ADJECTIVES and oa in valid_object_adj
                   and subject_type == "person" and object_type in INDEPENDENT_VERB_FRAMES.get(verb, set()))
        agreement = agreement and subj in INDEPENDENT_SUBJECTS and obj in (INDEPENDENT_PLACES | INDEPENDENT_ARTIFACTS)
    else:
        verb, od, oa, obj, prep, pd, pa, place = tokens
        agreement = od in {"a", "an", "the"} and pd in {"a", "an", "the"}
        phonology = (od == "the" or od == ("an" if oa[:1] in "aeiou" else "a"))
        phonology = phonology and (pd == "the" or pd == ("an" if pa[:1] in "aeiou" else "a"))
        valency = (prep == "in" and oa in INDEPENDENT_ARTIFACT_ADJECTIVES and obj in INDEPENDENT_ARTIFACTS
                   and pa in INDEPENDENT_PLACE_ADJECTIVES and place in INDEPENDENT_PLACES
                   and verb in {"build", "draft", "paint", "repair"})
    return {"ok": agreement and phonology and valency, "agreement_ok": agreement,
            "determiner_modifier_ok": phonology, "valency_ok": valency, "tokens": list(tokens),
            "semantic_frame": kind, "independent_inventory": True}


def render(words):
    text = " ".join(words); return text[:1].upper() + text[1:] + "."


def run(max_states=100_000):
    algebra = endpoint_algebra()
    joins_by = defaultdict(list)
    for row in algebra["joins"]: joins_by[row["derivation"]].append(row)
    runs, exact = [], []
    for d in DERIVATIONS:
        if d.identifier not in joins_by: continue
        grammar = grammar_for(d); kernel = progressive_construct(grammar, d, max_states)
        outputs = [audit(d, grammar, tuple(r["words"]), r, joins_by[d.identifier][0]) for r in kernel["records"]]
        exact.extend(outputs)
        runs.append({"derivation": d.identifier, "kind": d.frame.kind, "endpoint_joins": joins_by[d.identifier],
                     "slot_roles": list(d.roles), "kernel": {k: v for k, v in kernel.items() if k != "records"},
                     "complete_exact_records": outputs})
    admitted = [r for r in exact if r["mechanically_admitted"]]
    return {"status": "progressive_phrase_residual_whole_text_product",
            "config": {"min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS, "max_states_per_eligible_derivation": max_states,
                       "endpoint_index_minimum_pairs": 3, "progressive_residual_expansion": True,
                       "fine_grained_frame_specific_valency": True, "plans": ["SVO", "imperative-transitive-locative"],
                       "single_intact_clause": True, "even_and_odd_center_meets": True, "center_may_be_inside_word": True,
                       "independent_terminal_path_replay": True, "independent_complete_reparse": True, "corpus_generation": False,
                       "human_readability_required_after_admission": True},
            "authored_derivation_count": len(DERIVATIONS), "eligible_derivation_count": len(runs),
            "endpoint_algebra": algebra, "derivation_runs": runs, "exact_closures": exact, "admitted_closures": admitted,
            "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                           "kernel": "experiments/whole_text_palindrome_product_20260913.py",
                           "material": "authored frame-specific typed role inventory; no catalogue or fixed output",
                           "grammar_sha256": sha256(json.dumps([d.slots for d in DERIVATIONS], sort_keys=True).encode()).hexdigest(),
                           "known_tape_check": "central catalogue and endpoint-scaffold gates applied"},
            "reader_facing_next_test": "Only an admitted closure may enter randomized blinded intact-prose and shuffled-control reading; programmatic checks do not certify readability.",
            "scope": "Finite endpoint-indexed progressive residual products; zero reports actual residual depth and exhaustion versus truncation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--out", required=True, type=Path); parser.add_argument("--max-states", type=int, default=100_000); args = parser.parse_args()
    if args.out.exists(): parser.error(f"output already exists: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True); result = run(args.max_states); args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "authored": result["authored_derivation_count"], "eligible": result["eligible_derivation_count"], "exact": len(result["exact_closures"]), "admitted": len(result["admitted_closures"])}, indent=2))


if __name__ == "__main__": main()
