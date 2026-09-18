"""Connected-tree discourse grammar: ``Did I examine ...? I did.``

This is a fresh route after retiring the fixed ``Was it ... met at I saw``
envelope.  The root is a grammatical two-sentence discourse, with typed
object NPs and object-gap relatives.  A single frontier and character
residual are searched; no halves, catalogue phrases, or palindrome seeds are
used.  Punctuation is restored only for display and is ignored by the exact
letter audit.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "experiments/cfg_feature_palindrome_intersection_20260913.py"
spec = importlib.util.spec_from_file_location("discourse_fit_base_20260913", BASE_PATH)
BASE = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = BASE
spec.loader.exec_module(BASE)
from llm_palindrome.admission import mechanical_admission_checks

MIN_LETTERS, MAX_LETTERS = 100, 180
FIXED_LABELS = {"fixed_did", "fixed_i_question", "fixed_examine", "fixed_i", "fixed_did_final", "relative_that", "det"}

NOUNS = (
    ("portrait", "object"), ("packet", "object"), ("artifact", "object"),
    ("cabinet", "object"), ("garment", "object"), ("helmet", "object"),
    # The prior full-search ledger exposed an object-noun boundary requiring
    # terminal ``e`` (residual ``e`` versus ``helmet``'s ``t``).  ``statue`` is
    # an authored, ordinary object noun added as that typed morphology repair.
    ("statue", "object"),
    ("teacher", "person"), ("guard", "person"), ("writer", "person"),
    ("singer", "person"), ("farmer", "person"), ("nurse", "person"),
)
ADJECTIVES = ("quiet", "careful", "patient", "brave", "young", "gentle", "steady", "honest", "alert", "warm")
OBJECT_VERBS = ("watched", "painted", "saved", "praised")
PERSON_VERBS = ("greeted", "trusted", "followed", "helped", "joined")
LEXICON = {word: typ for word, typ in NOUNS}
LEXICON.update({word: "adjective" for word in ADJECTIVES})
LEXICON.update({word: "verb" for word in OBJECT_VERBS + PERSON_VERBS})


class DiscourseGrammar:
    """Finite typed CFG for a complete two-sentence discourse."""

    def __init__(self, max_depth: int = 6):
        self.max_depth = max_depth

    def start(self) -> BASE.Symbol:
        return BASE.sym("D", depth=str(self.max_depth))

    def terminal(self, symbol: BASE.Symbol) -> str | None:
        return symbol.feature("form") if symbol.name == "T" else None

    def productions(self, lhs: BASE.Symbol) -> tuple[BASE.Production, ...]:
        f = dict(lhs.features)
        if lhs.name == "D":
            return (BASE.Production("D:did-i-examine-object-i-did", lhs, (
                BASE.sym("T", label="fixed_did", form="did"),
                BASE.sym("T", label="fixed_i_question", form="i"),
                BASE.sym("T", label="fixed_examine", form="examine"),
                BASE.sym("NP", role="object", type="object", number="sing", depth=f["depth"]),
                BASE.sym("T", label="fixed_i", form="i"),
                BASE.sym("T", label="fixed_did_final", form="did"),
            )),)
        if lhs.name == "NP":
            role, typ, number, depth = f["role"], f["type"], f["number"], int(f["depth"])
            det = "this" if typ == "object" and role == "object" else "a"
            rows: list[BASE.Production] = []
            for adjective in ADJECTIVES:
                for noun, noun_type in NOUNS:
                    if noun_type != typ:
                        continue
                    prefix = (BASE.sym("T", label="det", form=det),
                              BASE.sym("T", label="adj", form=adjective),
                              BASE.sym("T", label=f"noun_{role}", form=noun))
                    rows.append(BASE.Production(f"NP:{role}:{typ}:{depth}:{adjective}:{noun}:base", lhs, prefix))
                    if depth > 0:
                        rows.append(BASE.Production(f"NP:{role}:{typ}:{depth}:{adjective}:{noun}:relative", lhs,
                                                    prefix + (BASE.sym("REL", head_type=typ, depth=str(depth - 1)),)))
            return tuple(rows)
        if lhs.name == "REL":
            head_type, depth = f["head_type"], int(f["depth"])
            patient_verbs = OBJECT_VERBS if head_type == "object" else PERSON_VERBS
            return tuple(BASE.Production(f"REL:{head_type}:{depth}:{verb}", lhs, (
                BASE.sym("T", label="relative_that", form="that"),
                BASE.sym("NP", role="relative_agent", type="person", number="sing", depth=str(depth)),
                BASE.sym("V", patient_type=head_type, subject_number="sing", form=verb),
            )) for verb in patient_verbs)
        if lhs.name == "V":
            return (BASE.Production(f"V:{lhs.feature('form')}", lhs,
                                    (BASE.sym("T", label="relative_verb", form=lhs.feature("form")),)),)
        return ()

    def digest(self) -> str:
        seen, queue, rows = set(), deque([self.start()]), []
        while queue:
            symbol = queue.popleft()
            if symbol in seen:
                continue
            seen.add(symbol)
            productions = self.productions(symbol)
            rows.append((symbol.name, symbol.features,
                         [(p.identifier, [x.name + str(x.features) for x in p.rhs]) for p in productions]))
            queue.extend(x for p in productions for x in p.rhs if self.terminal(x) is None)
        return sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def parse_discourse(grammar: DiscourseGrammar, text: str) -> BASE.Tree | None:
    """Independent complete reparse accepting only the declared display punctuation."""
    if text != text.strip() or re.sub(r"[a-z ?.!]", "", text.lower()):
        return None
    tokens = tuple(re.findall(r"[a-z]+", text.lower()))
    memo: dict[tuple[BASE.Symbol, int], tuple[tuple[BASE.Tree, int], ...]] = {}

    def parse_all(symbol: BASE.Symbol, offset: int) -> tuple[tuple[BASE.Tree, int], ...]:
        key = (symbol, offset)
        if key in memo:
            return memo[key]
        terminal = grammar.terminal(symbol)
        if terminal is not None:
            result = ((BASE.Tree(symbol, f"lex:{terminal}", (), terminal), offset + 1),) if offset < len(tokens) and tokens[offset] == terminal else ()
            memo[key] = result
            return result
        answers: list[tuple[BASE.Tree, int]] = []
        for production in grammar.productions(symbol):
            partials: list[tuple[list[BASE.Tree], int]] = [([], offset)]
            for child in production.rhs:
                next_partials: list[tuple[list[BASE.Tree], int]] = []
                for children, cursor in partials:
                    for node, end in parse_all(child, cursor):
                        next_partials.append((children + [node], end))
                partials = next_partials
                if not partials:
                    break
            answers.extend((BASE.Tree(symbol, production.identifier, tuple(children)), cursor) for children, cursor in partials)
        memo[key] = tuple(answers)
        return memo[key]

    return next((tree for tree, end in parse_all(grammar.start(), 0) if end == len(tokens)), None)


def exact_audit(text: str) -> dict[str, object]:
    tape = "".join(c for c in text.lower() if "a" <= c <= "z")
    mismatches = [(i, len(tape) - i - 1) for i in range(len(tape) // 2) if tape[i] != tape[-i - 1]]
    return {"exact": bool(tape) and not mismatches, "letters": len(tape), "mismatches": mismatches,
            "normalized_sha256": sha256(tape.encode()).hexdigest()}


def semantic_witness(tree: BASE.Tree | None) -> dict[str, object]:
    roles, relatives, agreement, valency = [], 0, True, True
    def visit(node: BASE.Tree) -> None:
        nonlocal relatives, agreement, valency
        if node.symbol.name == "NP":
            roles.append(dict(node.symbol.features))
        if node.symbol.name == "REL":
            relatives += 1
            if len(node.children) != 3 or node.children[0].terminal != "that" or node.children[1].symbol.name != "NP" or node.children[2].symbol.name != "V":
                valency = False
            elif node.children[2].symbol.feature("patient_type") != node.symbol.feature("head_type"):
                valency = False
            elif node.children[1].symbol.feature("number") != node.children[2].symbol.feature("subject_number"):
                agreement = False
        for child in node.children:
            visit(child)
    if tree:
        visit(tree)
    return {"semantic_roles": roles, "relative_count": relatives, "agreement_ok": agreement,
            "valency_ok": valency, "complete_tree": bool(tree and tree.symbol.name == "D")}


def display(text: str) -> str:
    words = text.split()
    return " ".join(words[:-2]) + "? " + " ".join(words[-2:]) + "."


def audit(grammar: DiscourseGrammar, text: str, kind: str, trace: tuple[tuple[int, str], ...]) -> dict[str, object]:
    surface = display(text)
    tree = parse_discourse(grammar, surface); exact = exact_audit(surface); witness = semantic_witness(tree)
    central = mechanical_admission_checks(surface, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    codes = [key for key, value in central.items() if not value]
    if tree is None:
        codes.append("independent_complete_reparse_failed")
    if not witness["agreement_ok"]:
        codes.append("agreement_failure")
    if not witness["valency_ok"]:
        codes.append("valency_failure")
    return {"record_kind": kind, "rendered": surface, "normalized_source": text,
            "independent_exact_audit": exact, "independent_parse": tree is not None,
            "feature_witness": witness, "central_admission": central, "mechanically_admitted": not codes,
            "rejection_codes": codes, "shared_tree_trace": list(trace),
            "reader_status": "unreviewed; programmatic checks do not certify readability"}


def content_words(state: BASE.State) -> tuple[str, ...]:
    return tuple(leaf.word for leaf in BASE.ordered_leaves(state) if leaf.label not in FIXED_LABELS)


def build_probe_tree(grammar: DiscourseGrammar) -> BASE.State:
    """Build one shallow, fully expanded tree for an actual boundary replay."""
    state = BASE.State((0,), (BASE.Node(0, grammar.start()),), (), "", 0, 0, ())
    while True:
        nodes = BASE.node_map(state); unresolved = [i for i, ref in enumerate(state.frontier) if not nodes[ref].terminal]
        if not unresolved:
            return state
        index = unresolved[0]; symbol = nodes[state.frontier[index]].symbol; f = dict(symbol.features)
        if symbol.name == "D":
            ident = "D:did-i-examine-object-i-did"
        elif symbol.name == "NP":
            d, typ = int(f["depth"]), f["type"]
            noun = "portrait" if typ == "object" else "teacher"
            ident = f"NP:{f['role']}:{typ}:{d}:quiet:{noun}:{'relative' if d > 0 else 'base'}"
        elif symbol.name == "REL":
            ident = f"REL:{f['head_type']}:{f['depth']}:watched"
        elif symbol.name == "V":
            ident = f"V:{f['form']}"
        else:
            raise AssertionError(symbol.name)
        options = [x for x in BASE.expand(grammar, state, index) if x.trace[-1][1] == ident]
        if len(options) != 1:
            raise AssertionError(f"nonunique probe production {symbol.name} {ident}")
        state = options[0]


def boundary_preflight(grammar: DiscourseGrammar) -> dict[str, object]:
    """Replay the opening/terminal boundary through BASE.emit, without seam scripting."""
    state = build_probe_tree(DiscourseGrammar(1)); emitted, events = state, []
    failed = None
    for step in range(1, 41):
        side = 1 if step % 2 else -1
        nodes, leaves = BASE.node_map(emitted), BASE.leaf_map(emitted)
        active = [ref for ref in (emitted.frontier if side == 1 else reversed(emitted.frontier)) if nodes[ref].terminal]
        if not active:
            break
        leaf = leaves[active[0]]; char = leaf.word[leaf.left] if side == 1 else leaf.word[-1 - leaf.right]
        next_state = BASE.emit(emitted, side)
        if next_state is None:
            failed = {"step": step, "side": side, "slot": leaf.label, "word": leaf.word,
                      "character": char, "emitter_rejected": True}
            break
        emitted = next_state
        events.append({"step": step, "side": side, "slot": leaf.label, "word": leaf.word,
                       "character": char, "residual_after": emitted.residual})
    return {"rendered": BASE.render(state), "display": display(BASE.render(state)),
            "emitter_events": events, "failed_attempt": failed,
            "opening_terminal_events": len(events), "independent_parse": parse_discourse(grammar, display(BASE.render(state))) is not None,
            "exact_audit": exact_audit(display(BASE.render(state))), "diagnostic_only": True}


def expand_edge(grammar: DiscourseGrammar, state: BASE.State, index: int, side: int) -> tuple[BASE.State, ...]:
    used = set(content_words(state)); previous = BASE.leaf_map(state); output = []
    for candidate in BASE.expand(grammar, state, index):
        added = [leaf for leaf in candidate.leaves if leaf.identifier not in previous]
        if any(leaf.word in used or leaf.word == leaf.word[::-1] for leaf in added if leaf.label not in FIXED_LABELS):
            continue
        output.append(candidate)
    return tuple(output)


def state_snapshot(state: BASE.State, attempts: list[dict[str, object]]) -> dict[str, object]:
    """Serialize one live connected-tree state for an independent replay."""
    nodes = []
    for node in state.nodes:
        nodes.append({"identifier": node.identifier, "symbol": {"name": node.symbol.name, "features": list(node.symbol.features)},
                      "terminal": node.terminal, "production": node.production, "children": list(node.children)})
    leaves = [{"identifier": leaf.identifier, "node": leaf.node, "word": leaf.word, "label": leaf.label,
               "left": leaf.left, "right": leaf.right} for leaf in state.leaves]
    return {"frontier": list(state.frontier), "nodes": nodes, "leaves": leaves,
            "residual": state.residual, "owner": state.owner, "length": state.length,
            "trace": [list(item) for item in state.trace], "rendered_tree": BASE.render(state),
            "attempts": attempts}


def restore_snapshot(snapshot: dict[str, object]) -> BASE.State:
    nodes = tuple(BASE.Node(int(row["identifier"]), BASE.Symbol(str(row["symbol"]["name"]), tuple(tuple(x) for x in row["symbol"]["features"])),
                            str(row["terminal"]), str(row["production"]), tuple(int(x) for x in row["children"])) for row in snapshot["nodes"])
    leaves = tuple(BASE.Leaf(int(row["identifier"]), int(row["node"]), str(row["word"]), str(row["label"]), int(row["left"]), int(row["right"])) for row in snapshot["leaves"])
    return BASE.State(tuple(int(x) for x in snapshot["frontier"]), nodes, leaves, str(snapshot["residual"]),
                     int(snapshot["owner"]), int(snapshot["length"]), tuple(tuple(x) for x in snapshot["trace"]))


def actual_attempts(state: BASE.State) -> list[dict[str, object]]:
    """Record the exact next terminal emissions offered by this state."""
    nodes, leaves = BASE.node_map(state), BASE.leaf_map(state)
    sides = (-state.owner,) if state.residual else (1, -1)
    attempts = []
    for side in sides:
        active = [ref for ref in (state.frontier if side == 1 else reversed(state.frontier)) if nodes[ref].terminal]
        if not active:
            attempts.append({"side": side, "available": False})
            continue
        leaf = leaves[active[0]]; char = leaf.word[leaf.left] if side == 1 else leaf.word[-1 - leaf.right]
        attempts.append({"side": side, "available": True, "slot": leaf.label, "word": leaf.word,
                         "character": char, "accepted": BASE.emit(state, side) is not None})
    return attempts


def replay_ledger(snapshot: dict[str, object]) -> dict[str, object]:
    state = restore_snapshot(snapshot); attempts = actual_attempts(state)
    expected = snapshot["attempts"]
    return {"same_rendered_tree": BASE.render(state) == snapshot["rendered_tree"],
            "same_length": state.length == snapshot["length"], "same_residual": state.residual == snapshot["residual"],
            "same_attempts": attempts == expected, "replayed_attempts": attempts}


def solver(grammar: DiscourseGrammar, *, max_states: int = 100000) -> dict[str, object]:
    initial = BASE.State((0,), (BASE.Node(0, grammar.start()),), (), "", 0, 0, ())
    queue, seen, exact_rows, admitted = deque([initial]), set(), {}, {}
    stats = Counter(states=0, expansions=0, emissions=0, residual_contradictions=0, complete_trees=0, exact_closures=0)
    deepest: dict[str, object] | None = None; deepest_length = -1
    deepest_contradiction: dict[str, object] | None = None; deepest_contradiction_length = -1
    while queue and stats["states"] < max_states:
        state = queue.pop(); stats["states"] += 1
        key = (state.frontier, state.nodes, state.leaves, state.residual, state.owner, state.length)
        if key in seen:
            continue
        seen.add(key)
        attempts = actual_attempts(state)
        if state.length > deepest_length:
            deepest_length = state.length; deepest = state_snapshot(state, attempts)
        # Separate the deepest live state from the deepest state with an
        # actually rejected next emission.  The latter is the actionable seam
        # witness, and is replayed independently below.
        rejected = any(a.get("available") and not a.get("accepted") for a in attempts)
        if rejected and state.length > deepest_contradiction_length:
            deepest_contradiction_length = state.length
            deepest_contradiction = state_snapshot(state, attempts)
        if BASE.complete(state):
            stats["complete_trees"] += 1
            if MIN_LETTERS <= state.length <= MAX_LETTERS and state.residual == state.residual[::-1]:
                text = BASE.render(state); row = audit(grammar, text, "complete_discourse_closure", state.trace)
                if row["independent_exact_audit"]["exact"] and row["independent_parse"]:
                    stats["exact_closures"] += 1; exact_rows.setdefault(text, row)
                    if row["mechanically_admitted"]:
                        admitted.setdefault(text, row)
            continue
        nodes = BASE.node_map(state); unresolved = [i for i, ref in enumerate(state.frontier) if not nodes[ref].terminal]
        index = None
        if state.residual:
            side = -state.owner; edge = 0 if side == 1 else len(state.frontier) - 1
            if not nodes[state.frontier[edge]].terminal:
                index = edge
        elif unresolved:
            if not nodes[state.frontier[0]].terminal:
                index, side = 0, 1
            elif not nodes[state.frontier[-1]].terminal:
                index, side = len(state.frontier) - 1, -1
            else:
                index, side = unresolved[0], 1
        if index is not None:
            for next_state in expand_edge(grammar, state, index, side):
                queue.append(next_state); stats["expansions"] += 1
        left_ready = bool(nodes[state.frontier[0]].terminal); right_ready = bool(nodes[state.frontier[-1]].terminal)
        for side in ((-state.owner,) if state.residual else (1, -1)):
            if (side == 1 and not left_ready) or (side == -1 and not right_ready):
                continue
            next_state = BASE.emit(state, side)
            if next_state is not None:
                queue.append(next_state); stats["emissions"] += 1
            else:
                stats["residual_contradictions"] += 1
    ledger = {
        "deepest_state": deepest,
        "replay": replay_ledger(deepest) if deepest is not None else None,
        "deepest_contradiction_state": deepest_contradiction,
        "deepest_contradiction_replay": replay_ledger(deepest_contradiction) if deepest_contradiction is not None else None,
    }
    return {"stats": dict(stats), "states_exhausted": not queue and stats["states"] < max_states,
            "exact_closures": list(exact_rows.values()), "mechanically_admitted_closures": list(admitted.values()),
            "deepest_state_ledger": ledger}


def explicit_control(grammar: DiscourseGrammar) -> BASE.State:
    state = BASE.State((0,), (BASE.Node(0, grammar.start()),), (), "", 0, 0, ())
    adjectives = iter(("quiet", "careful", "patient", "brave", "young", "gentle", "steady"))
    people = iter(("teacher", "guard", "writer", "singer", "farmer", "nurse"))
    verbs = iter(("watched", "painted", "saved", "praised", "watched", "painted"))
    while True:
        nodes = BASE.node_map(state); unresolved = [i for i, ref in enumerate(state.frontier) if not nodes[ref].terminal]
        if not unresolved:
            break
        index = unresolved[0]; symbol = nodes[state.frontier[index]].symbol; f = dict(symbol.features)
        if symbol.name == "D": ident = "D:did-i-examine-object-i-did"
        elif symbol.name == "NP":
            depth = int(f["depth"]); ident = f"NP:{f['role']}:{f['type']}:{depth}:quiet:portrait:relative" if depth > 0 and f["type"] == "object" else f"NP:{f['role']}:{f['type']}:{depth}:base"
        elif symbol.name == "REL": ident = f"REL:{f['head_type']}:{f['depth']}:{next(verbs)}"
        elif symbol.name == "V": ident = f"V:{f['form']}"
        elif symbol.name == "T": ident = ""
        else:
            # Select the authored lexical terminal by its role and the
            # corresponding deterministic control stream.
            if symbol.name == "T": ident = ""
            else: raise AssertionError(symbol.name)
        if symbol.name == "NP" and ident.endswith(":base"):
            d, typ = int(f["depth"]), f["type"]; noun = next(people) if typ == "person" else "portrait"
            ident = f"NP:{f['role']}:{typ}:{d}:quiet:{noun}:base"
        options = [x for x in BASE.expand(grammar, state, index) if x.trace[-1][1] == ident]
        if symbol.name == "T":
            ident = next(p.identifier for p in grammar.productions(symbol) if grammar.terminal(p.rhs[0]) == grammar.terminal(symbol)) if False else ""
        if len(options) != 1:
            # Terminal symbols are already expanded by grammar productions of
            # their parent; reaching one here means the control path is wrong.
            raise AssertionError(f"nonunique control production {symbol.name} {ident}")
        state = options[0]
    return state


def run(*, max_states: int = 100000) -> dict[str, object]:
    grammar = DiscourseGrammar(6); result = solver(grammar, max_states=max_states)
    # Build a deterministic long tree through the same expansion API.  This
    # explicit control is diagnostic only and is independently reparsed.
    state = BASE.State((0,), (BASE.Node(0, grammar.start()),), (), "", 0, 0, ())
    choices = {"D": "D:did-i-examine-object-i-did"}
    control_adjectives = {6: "quiet", 5: "careful", 4: "patient", 3: "brave", 2: "young", 1: "gentle", 0: "steady"}
    control_people = {5: "teacher", 4: "guard", 3: "writer", 2: "singer", 1: "farmer", 0: "nurse"}
    control_object_verbs = {5: "watched"}
    control_person_verbs = {4: "greeted", 3: "trusted", 2: "followed", 1: "helped", 0: "joined"}
    while True:
        nodes = BASE.node_map(state); unresolved = [i for i, ref in enumerate(state.frontier) if not nodes[ref].terminal]
        if not unresolved: break
        index = unresolved[0]; symbol = nodes[state.frontier[index]].symbol; f = dict(symbol.features)
        if symbol.name == "D": ident = choices["D"]
        elif symbol.name == "NP":
            d, typ = int(f["depth"]), f["type"]
            noun = "portrait" if typ == "object" else control_people[d]
            ident = f"NP:{f['role']}:{typ}:{d}:{control_adjectives[d]}:{noun}:{'relative' if d > 0 else 'base'}"
        elif symbol.name == "REL":
            d = int(f["depth"])
            verb = control_object_verbs[d] if f["head_type"] == "object" else control_person_verbs[d]
            ident = f"REL:{f['head_type']}:{d}:{verb}"
        elif symbol.name == "V": ident = f"V:{f['form']}"
        else: raise AssertionError(symbol.name)
        options = [x for x in BASE.expand(grammar, state, index) if x.trace[-1][1] == ident]
        if len(options) != 1: raise AssertionError(f"nonunique control path {symbol.name} {ident}")
        state = options[0]
    control_text = BASE.render(state); control = audit(grammar, control_text, "complete_discourse_recursive_control", state.trace)
    preflight = boundary_preflight(grammar)
    control.update({"diagnostic_only": True, "grammar_tree_fully_expanded": True,
                    "provenance": {"construction": "explicit six-level object-relative discourse tree", "shared_tree": True},
                    "reader_status": "grammar control only; not a palindrome candidate or readability evidence"})
    return {"status": "discourse_fit_relative_intersection", "config": {
        "min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS, "max_depth": grammar.max_depth,
        "max_states": max_states, "single_shared_derivation_tree": True,
        "character_residual_during_derivation": True, "joint_discourse_envelope": "Did I examine ...? I did.",
        "typed_object_gap_relatives": True, "closure_requires_all_leaves_consumed": True,
        "independent_complete_reparse": True, "corpus_generation": False},
        "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                       "grammar_sha256": grammar.digest(), "material": "authored ordinary discourse/object lexicon; no catalogue or known palindrome material"},
        "complete_recursive_control": control, **result,
        "joint_boundary_preflight": preflight,
        "reader_facing_next_operator": "Replace the exact typed relation named by deepest_contradiction_state, then rerun the same full-tree emitter and replay ledger.",
        "scope": "Zero-closure bounded run is diagnostic; no reader-facing palindrome is claimed."}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--max-states", type=int, default=100000)
    args = parser.parse_args()
    if args.out.exists(): parser.error(f"refusing to overwrite {args.out}")
    result = run(max_states=args.max_states); args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "states": result["stats"]["states"], "exact_closures": len(result["exact_closures"])}, indent=2))


if __name__ == "__main__": main()
