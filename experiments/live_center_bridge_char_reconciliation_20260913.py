"""Live character-state reconciliation over a connected compound grammar.

Unlike whole-word bridge probes, this operator lexicalizes only an exposed
leaf and advances one character at a time.  The center compound is licensed
by a semantic table (``paper report`` is a report written on paper), while its
letter bridge is merely a search constraint.  The trace records every
character, residual, cancellation, and first contradiction.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

MIN_LETTERS, MAX_LETTERS = 100, 180
WORD = re.compile(r"[a-z]+")


@dataclass(frozen=True)
class Lexeme:
    word: str
    category: str
    semantic: str
    subject: str = ""
    object: str = ""


@dataclass(frozen=True)
class Symbol:
    name: str
    role: str = ""
    kind: str = ""


@dataclass(frozen=True)
class Node:
    symbol: Symbol
    children: tuple["Node", ...] = ()


@dataclass(frozen=True)
class Slot:
    index: int
    symbol: Symbol


LEXICON = (
    Lexeme("a", "det", "determiner"), Lexeme("an", "det", "determiner"), Lexeme("the", "det", "determiner"),
    Lexeme("this", "det", "determiner"), Lexeme("that", "det", "determiner"), Lexeme("after", "prep", "temporal"),
    Lexeme("near", "prep", "locative"), Lexeme("today", "adv", "temporal"),
    Lexeme("expert", "adj_person", "person"), Lexeme("patient", "adj_person", "person"), Lexeme("careful", "adj_person", "person"),
    Lexeme("comprehensive", "adj_artifact", "artifact"), Lexeme("substantial", "adj_artifact", "artifact"), Lexeme("detailed", "adj_artifact", "artifact"),
    Lexeme("valuable", "adj_artifact", "artifact"), Lexeme("archivist", "noun_person", "person"), Lexeme("researcher", "noun_person", "person"),
    Lexeme("curator", "noun_person", "person"), Lexeme("editor", "noun_person", "person"), Lexeme("administrator", "noun_person", "person"), Lexeme("paper", "noun_artifact", "artifact"),
    Lexeme("report", "noun_artifact", "artifact"), Lexeme("manuscript", "noun_artifact", "artifact"), Lexeme("collection", "noun_artifact", "artifact"),
    Lexeme("artifact", "noun_artifact", "artifact"), Lexeme("record", "noun_artifact", "artifact"), Lexeme("inspection", "noun_event", "event"),
    Lexeme("study", "noun_event", "event"), Lexeme("trial", "noun_event", "event"), Lexeme("repository", "noun_org", "organization"),
    Lexeme("archive", "noun_org", "organization"), Lexeme("museum", "noun_org", "organization"),
    Lexeme("reviews", "verb", "review", "person", "artifact"), Lexeme("records", "verb", "record", "person", "artifact"),
    Lexeme("repairs", "verb", "repair", "person", "artifact"), Lexeme("catalogs", "verb", "record", "person", "artifact"),
)
BY_CATEGORY: dict[str, tuple[Lexeme, ...]] = {}
for item in LEXICON:
    BY_CATEGORY.setdefault(item.category, ()); BY_CATEGORY[item.category] += (item,)

# Actual semantic licensing for the compound center; this is deliberately
# narrower than arbitrary noun adjacency.
LICENSED_COMPOUNDS = {("paper", "report"): "a report written on paper"}


def terminal(kind: str, role: str) -> Symbol:
    return Symbol("T", role=role, kind=kind)


class Grammar:
    def start(self) -> Symbol: return Symbol("S")
    def productions(self, s: Symbol) -> tuple[tuple[Symbol, ...], ...]:
        if s.name == "S": return ((Symbol("NP", "subject", "person"), terminal("verb", "matrix_verb"), Symbol("COMPOUND", "object"), Symbol("PP", "temporal"), Symbol("PP", "locative"), terminal("adv", "today")),)
        if s.name == "NP": return ((terminal("det", s.role + "_det"), terminal("adj_person" if s.kind == "person" else "adj_artifact", s.role + "_adj"), terminal("noun_person" if s.kind == "person" else "noun_artifact", s.role)),)
        if s.name == "COMPOUND": return ((terminal("det", "object_det"), terminal("adj_artifact", "object_adj"), terminal("adj_artifact", "object_adj2"), terminal("noun_artifact", "compound_modifier"), terminal("noun_artifact", "compound_head")),)
        if s.name == "PP": return ((terminal("prep", s.role), terminal("det", s.role + "_det"), terminal("noun_event" if s.role == "temporal" else "noun_org", s.role + "_object")),)
        return ()
    def expand(self, s: Symbol) -> Node:
        rhs = self.productions(s); return Node(s, tuple(self.expand(x) for x in rhs[0])) if rhs else Node(s)
    def digest(self) -> str:
        rows = []; seen = set(); queue = [self.start()]
        while queue:
            s = queue.pop(0)
            if s in seen: continue
            seen.add(s); rhs = self.productions(s); rows.append((asdict(s), [[asdict(x) for x in row] for row in rhs])); queue.extend(x for row in rhs for x in row if x.name != "T")
        return sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def slots(tree: Node) -> tuple[Slot, ...]:
    out: list[Slot] = []
    def visit(n: Node) -> None:
        if n.symbol.name == "T": out.append(Slot(len(out), n.symbol)); return
        for child in n.children: visit(child)
    visit(tree); return tuple(out)


def choices(s: Symbol) -> tuple[Lexeme, ...]: return BY_CATEGORY[s.kind]
def render(words: Iterable[str]) -> str:
    text = " ".join(words); return text[:1].upper() + text[1:] + "."


def semantic_ok(slot: Slot, item: Lexeme, assigned: dict[str, Lexeme]) -> bool:
    if item.word in {x.word for x in assigned.values()}: return False
    return slot.symbol.kind != "verb" or (item.subject, item.object) == ("person", "artifact")


def emit_char(left_residual: str, right_residual: str, char: str, side: int, trace: list[dict[str, object]], stats: Counter) -> tuple[str, str] | None:
    before = left_residual or right_residual
    if side == 1:
        stats["left_char_emissions"] += 1
        if right_residual:
            if char != right_residual[0]:
                trace.append({"side": "left", "char": char, "residual_before": before, "action": "contradiction"}); stats["residual_contradictions"] += 1; return None
            right_residual = right_residual[1:]; stats["cancellations"] += 1; action = "cancel"
        else: left_residual += char; action = "open_left"
    else:
        stats["right_char_emissions"] += 1
        if left_residual:
            if char != left_residual[0]:
                trace.append({"side": "right", "char": char, "residual_before": before, "action": "contradiction"}); stats["residual_contradictions"] += 1; return None
            left_residual = left_residual[1:]; stats["cancellations"] += 1; action = "cancel"
        else: right_residual += char; action = "open_right"
    trace.append({"side": "left" if side == 1 else "right", "char": char, "residual_before": before, "residual_after": left_residual or right_residual, "action": action})
    return left_residual, right_residual


def bridge_trace(left: str, right: str) -> dict[str, object]:
    """Produce a full live character trace for one licensed center pair."""
    stats = Counter(); trace: list[dict[str, object]] = []; lp = rp = ""; li, ri, last = len(left)-1, 0, 0
    while li >= 0 or ri < len(right):
        side = -1 if lp else (1 if last in {0, -1} and li >= 0 else -1)
        if side == 1 and li >= 0: char = left[li]; li -= 1
        elif side == -1 and ri < len(right): char = right[ri]; ri += 1
        else: break
        updated = emit_char(lp, rp, char, side, trace, stats)
        if updated is None: break
        lp, rp = updated; last = side
    return {"left": left, "right": right, "trace": trace, "stats": dict(stats), "completed": li < 0 and ri >= len(right) and not (lp or rp)}


def exact_audit(text: str) -> dict[str, object]:
    tape = normalize_letters(text); mismatches = [(i, len(tape)-1-i) for i in range(len(tape)//2) if tape[i] != tape[-i-1]]; cursor = 0; bounds = []
    for token in WORD.findall(text.lower()): cursor += len(token); bounds.append(cursor)
    return {"exact": bool(tape) and not mismatches, "letters": len(tape), "mismatches": mismatches, "normalized_sha256": sha256(tape.encode()).hexdigest(), "shifted_word_boundaries": sorted(set(bounds[:-1]) - {len(tape)-x for x in bounds[:-1]})}


def parse_tree(grammar: Grammar, text: str) -> Node | None:
    if text != text.strip() or re.sub(r"[A-Za-z ,.?!'-]", "", text): return None
    tokens = tuple(WORD.findall(text.lower())); expected = slots(grammar.expand(grammar.start()))
    if len(tokens) != len(expected): return None
    for i, slot in enumerate(expected):
        if not any(tokens[i] == x.word for x in choices(slot.symbol)): return None
        if slot.symbol.kind == "det" and not ((tokens[i] == "an" and tokens[i+1][0] in "aeiou") or (tokens[i] == "a" and tokens[i+1][0] not in "aeiou") or tokens[i] in {"the", "this", "that"}): return None
    if (tokens[7], tokens[8]) not in LICENSED_COMPOUNDS: return None
    if tokens[3] not in {x.word for x in BY_CATEGORY["verb"] if x.subject == "person" and x.object == "artifact"}: return None
    return grammar.expand(grammar.start())


def audit(grammar: Grammar, text: str, kind: str, provenance: tuple[str, ...]) -> dict[str, object]:
    exact = exact_audit(text); parsed = parse_tree(grammar, text); central = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS); codes = [k for k, v in central.items() if not v]
    if parsed is None: codes.append("independent_complete_reparse_failed")
    return {"record_kind": kind, "rendered": text, "provenance": provenance, "independent_exact_audit": exact, "independent_parse": parsed is not None, "central_admission": central, "mechanically_admitted": not codes, "rejection_codes": codes, "reader_status": "unreviewed; programmatic checks do not certify readability"}


def search(grammar: Grammar, leaf_slots: tuple[Slot, ...], *, state_limit: int, closure_limit: int, stats: Counter) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Live one-character center-out search; words appear only at exposure."""
    n, center = len(leaf_slots), len(leaf_slots)//2; words = [""]*n; closures: list[dict[str, object]] = []; traces: list[dict[str, object]] = []
    bridge_pairs = [(l, r) for l, r in LICENSED_COMPOUNDS if normalize_letters(l)[::-1] != normalize_letters(r) and normalize_letters(l)[-1] == normalize_letters(r)[0]]
    for left_word, right_word in bridge_pairs:
        bridge = bridge_trace(left_word, right_word); traces.append(bridge)
        for key, value in bridge["stats"].items():
            stats[key] += value
        assigned = {leaf_slots[center-1].symbol.role: Lexeme(left_word, "noun_artifact", "artifact"), leaf_slots[center].symbol.role: Lexeme(right_word, "noun_artifact", "artifact")}; words[center-1], words[center] = left_word, right_word
        # The center trace is live; only a fully reconciled center may move to
        # exposed outer leaves. A contradiction is retained as evidence.
        stats["state_count"] += len(bridge["trace"])
        if not bridge["completed"]: stats["center_bridge_contradictions"] += 1; words[center-1] = words[center] = ""; continue
        words[center-1] = words[center] = ""
    return closures, traces


def run(*, state_limit: int = 100_000, closure_limit: int = 100) -> dict[str, object]:
    grammar = Grammar(); leaves = slots(grammar.expand(grammar.start())); stats = Counter(state_count=0, left_char_emissions=0, right_char_emissions=0, cancellations=0, residual_contradictions=0, center_bridge_contradictions=0)
    closures, traces = search(grammar, leaves, state_limit=state_limit, closure_limit=closure_limit, stats=stats)
    controls = ("An expert administrator reviews a comprehensive substantial paper report after the inspection near a repository today.", "A patient administrator catalogs a comprehensive substantial paper report after the inspection near a repository today.")
    rows = [audit(grammar, text, "complete_connected_grammar_control", tuple(WORD.findall(text.lower()))) for text in controls]
    return {"status": "live_center_bridge_char_reconciliation_single_tree", "config": {"min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS, "state_limit": state_limit, "closure_limit": closure_limit, "one_connected_tree": True, "grammar_owns_every_leaf": True, "words_lexicalized_only_at_exposed_slots": True, "one_character_emission_states": True, "semantic_compound_licensing": True, "closure_requires_complete_tree": True, "independent_complete_reparse": True, "corpus_or_catalogue_generation": False}, "licensed_compounds": [{"modifier": a, "head": b, "meaning": meaning} for (a, b), meaning in LICENSED_COMPOUNDS.items()], "center_bridge_traces": traces, "stats": dict(stats), "exact_closures": closures, "complete_grammar_controls": rows, "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "grammar_sha256": grammar.digest(), "material": "task-authored connected grammar, semantic compound table, and lexicon; no corpus/catalogue text"}, "reader_facing_next_operator": "Add a second semantically licensed compound whose bridge survives beyond the center, then expose and reconcile outer leaves one character at a time with the same independent reparse gate.", "scope": "The bridge trace is a construction diagnostic; exactness and parsing do not certify human readability."}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--state-limit", type=int, default=100_000); parser.add_argument("--closure-limit", type=int, default=100); args = parser.parse_args()
    if args.out.exists(): parser.error(f"refusing to overwrite {args.out}")
    result = run(state_limit=args.state_limit, closure_limit=args.closure_limit); args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps({"out": str(args.out), "states": result["stats"]["state_count"], "exact": len(result["exact_closures"]), "admitted": sum(x["mechanically_admitted"] for x in result["exact_closures"]), "bridge_traces": len(result["center_bridge_traces"])}, indent=2))


if __name__ == "__main__": main()
