"""One-tree crossed-boundary discourse grammar.

The authored sentence envelope is ``Now nurses record ... and the contest
remains unwon.``  Its opening ``now`` meets the suffix of longer ``unwon``;
there is no mirrored multiword endpoint.  Variable prose is a typed CFG tree,
and every character is emitted through the exact shared residual engine.
"""
from __future__ import annotations
import argparse, importlib.util, json, re, sys
from collections import Counter, deque
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "experiments/cfg_feature_palindrome_intersection_20260913.py"
spec = importlib.util.spec_from_file_location("now_unwon_base_20260913", BASE_PATH)
BASE = importlib.util.module_from_spec(spec); assert spec.loader is not None
sys.modules[spec.name] = BASE; spec.loader.exec_module(BASE)
from llm_palindrome.admission import mechanical_admission_checks

MIN_LETTERS, MAX_LETTERS = 100, 180
FIXED_LABELS = {"fixed_now", "fixed_nurses", "fixed_record", "fixed_and", "fixed_the",
                "fixed_contest", "fixed_remains", "fixed_unwon", "det", "relative_that"}
NOUNS = (("portrait", "object"), ("packet", "object"), ("artifact", "object"),
         ("cabinet", "object"), ("garment", "object"), ("helmet", "object"),
         ("statue", "object"), ("teacher", "person"), ("guard", "person"),
         ("writer", "person"), ("singer", "person"), ("farmer", "person"), ("nurse", "person"))
ADJECTIVES = ("quiet", "careful", "patient", "brave", "young", "gentle", "steady", "honest")
OBJECT_VERBS = ("watched", "painted", "saved", "praised")
PERSON_VERBS = ("greeted", "trusted", "followed", "helped", "joined")


class CrossBoundaryGrammar:
    def __init__(self, max_depth=6): self.max_depth = max_depth
    def start(self): return BASE.sym("D", depth=str(self.max_depth))
    def terminal(self, symbol): return symbol.feature("form") if symbol.name == "T" else None
    def productions(self, lhs):
        f = dict(lhs.features)
        if lhs.name == "D":
            return (BASE.Production("D:now-nurses-record-object-and-contest-remains-unwon", lhs, (
                BASE.sym("T", label="fixed_now", form="now"), BASE.sym("T", label="fixed_nurses", form="nurses"),
                BASE.sym("T", label="fixed_record", form="record"), BASE.sym("NP", role="object", type="object", number="sing", depth=f["depth"]),
                BASE.sym("T", label="fixed_and", form="and"), BASE.sym("T", label="fixed_the", form="the"),
                BASE.sym("T", label="fixed_contest", form="contest"), BASE.sym("T", label="fixed_remains", form="remains"),
                BASE.sym("T", label="fixed_unwon", form="unwon"))),)
        if lhs.name == "NP":
            role, typ, depth = f["role"], f["type"], int(f["depth"]); det = "a" if f["number"] == "sing" else "the"; rows = []
            for adj in ADJECTIVES:
                for noun, noun_type in NOUNS:
                    if noun_type != typ: continue
                    prefix = (BASE.sym("T", label="det", form=det), BASE.sym("T", label="adj", form=adj), BASE.sym("T", label=f"noun_{role}", form=noun))
                    rows.append(BASE.Production(f"NP:{role}:{typ}:{depth}:{adj}:{noun}:base", lhs, prefix))
                    if depth > 0: rows.append(BASE.Production(f"NP:{role}:{typ}:{depth}:{adj}:{noun}:relative", lhs, prefix + (BASE.sym("REL", head_type=typ, depth=str(depth - 1)),)))
            return tuple(rows)
        if lhs.name == "REL":
            head, depth = f["head_type"], int(f["depth"]); verbs = OBJECT_VERBS if head == "object" else PERSON_VERBS
            return tuple(BASE.Production(f"REL:{head}:{depth}:{verb}", lhs, (BASE.sym("T", label="relative_that", form="that"), BASE.sym("NP", role="relative_agent", type="person", number="sing", depth=str(depth)), BASE.sym("V", patient_type=head, subject_number="sing", form=verb))) for verb in verbs)
        if lhs.name == "V": return (BASE.Production(f"V:{lhs.feature('form')}", lhs, (BASE.sym("T", label="relative_verb", form=lhs.feature("form")),)),)
        return ()
    def digest(self):
        seen, queue, rows = set(), deque([self.start()]), []
        while queue:
            symbol = queue.popleft()
            if symbol in seen: continue
            seen.add(symbol); prods = self.productions(symbol)
            rows.append((symbol.name, symbol.features, [(p.identifier, [x.name + str(x.features) for x in p.rhs]) for p in prods]))
            queue.extend(x for p in prods for x in p.rhs if self.terminal(x) is None)
        return sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def parse_complete(grammar, text):
    if text != text.strip() or re.sub(r"[a-z ,;.!?]", "", text.lower()): return None
    tokens = tuple(re.findall(r"[a-z]+", text.lower())); memo = {}
    def parses(symbol, offset):
        key = (symbol, offset)
        if key in memo: return memo[key]
        term = grammar.terminal(symbol)
        if term is not None:
            ans = ((BASE.Tree(symbol, f"lex:{term}", (), term), offset + 1),) if offset < len(tokens) and tokens[offset] == term else ()
            memo[key] = ans; return ans
        answers = []
        for prod in grammar.productions(symbol):
            partials = [([], offset)]
            for child in prod.rhs:
                nxt = []
                for children, cursor in partials: nxt.extend((children + [tree], end) for tree, end in parses(child, cursor))
                partials = nxt
                if not partials: break
            answers.extend((BASE.Tree(symbol, prod.identifier, tuple(children)), end) for children, end in partials)
        memo[key] = tuple(answers); return memo[key]
    return next((tree for tree, end in parses(grammar.start(), 0) if end == len(tokens)), None)


def exact_audit(text):
    tape = "".join(c for c in text.lower() if "a" <= c <= "z")
    mismatches = [(i, len(tape) - i - 1) for i in range(len(tape) // 2) if tape[i] != tape[-i - 1]]
    return {"exact": bool(tape) and not mismatches, "letters": len(tape), "mismatches": mismatches, "normalized_sha256": sha256(tape.encode()).hexdigest()}


def semantic_witness(tree):
    roles, relatives, agreement, valency = [], 0, True, True
    def visit(node):
        nonlocal relatives, agreement, valency
        if node.symbol.name == "NP": roles.append(dict(node.symbol.features))
        if node.symbol.name == "REL":
            relatives += 1
            if len(node.children) != 3 or node.children[0].terminal != "that" or node.children[1].symbol.name != "NP" or node.children[2].symbol.name != "V": valency = False
            elif node.children[2].symbol.feature("patient_type") != node.symbol.feature("head_type"): valency = False
            elif node.children[1].symbol.feature("number") != node.children[2].symbol.feature("subject_number"): agreement = False
        for child in node.children: visit(child)
    if tree: visit(tree)
    root_ok = bool(tree and tree.children[1].terminal == "nurses" and tree.children[2].terminal == "record")
    return {"semantic_roles": roles, "relative_count": relatives, "agreement_ok": agreement and root_ok, "valency_ok": valency, "complete_tree": bool(tree and tree.symbol.name == "D")}


def audit(grammar, text, kind, trace):
    tree, exact, witness = parse_complete(grammar, text), exact_audit(text), semantic_witness(parse_complete(grammar, text))
    central = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS); codes = [k for k, v in central.items() if not v]
    if tree is None: codes.append("independent_complete_reparse_failed")
    if not witness["agreement_ok"]: codes.append("agreement_failure")
    if not witness["valency_ok"]: codes.append("valency_failure")
    return {"record_kind": kind, "rendered": text, "independent_exact_audit": exact, "independent_parse": tree is not None, "feature_witness": witness, "central_admission": central, "mechanically_admitted": not codes, "rejection_codes": codes, "shared_tree_trace": list(trace), "reader_status": "unreviewed; programmatic checks do not certify readability"}


def snapshot(state, attempts):
    return {"frontier": list(state.frontier), "nodes": [{"identifier": n.identifier, "symbol": {"name": n.symbol.name, "features": list(n.symbol.features)}, "terminal": n.terminal, "production": n.production, "children": list(n.children)} for n in state.nodes], "leaves": [{"identifier": x.identifier, "node": x.node, "word": x.word, "label": x.label, "left": x.left, "right": x.right} for x in state.leaves], "residual": state.residual, "owner": state.owner, "length": state.length, "trace": [list(x) for x in state.trace], "rendered_tree": BASE.render(state), "attempts": attempts}


def restore(row):
    nodes = tuple(BASE.Node(int(n["identifier"]), BASE.Symbol(n["symbol"]["name"], tuple(tuple(x) for x in n["symbol"]["features"])), n["terminal"], n["production"], tuple(n["children"])) for n in row["nodes"])
    leaves = tuple(BASE.Leaf(int(x["identifier"]), int(x["node"]), x["word"], x["label"], int(x["left"]), int(x["right"])) for x in row["leaves"])
    return BASE.State(tuple(row["frontier"]), nodes, leaves, row["residual"], int(row["owner"]), int(row["length"]), tuple(tuple(x) for x in row["trace"]))


def attempts(state):
    nodes, leaves = BASE.node_map(state), BASE.leaf_map(state); sides = (-state.owner,) if state.residual else (1, -1); out = []
    for side in sides:
        active = [r for r in (state.frontier if side == 1 else reversed(state.frontier)) if nodes[r].terminal]
        if not active: out.append({"side": side, "available": False}); continue
        leaf = leaves[active[0]]; char = leaf.word[leaf.left] if side == 1 else leaf.word[-1 - leaf.right]
        out.append({"side": side, "available": True, "slot": leaf.label, "word": leaf.word, "character": char, "accepted": BASE.emit(state, side) is not None})
    return out


def boundary_preflight():
    """Replay the crossed outer seam using the real emitter on one full tree."""
    grammar = CrossBoundaryGrammar(0); state = explicit_control(grammar); events = []; failed = None
    for step in range(1, 31):
        side = 1 if step % 2 else -1; nodes, leaves = BASE.node_map(state), BASE.leaf_map(state)
        active = [r for r in (state.frontier if side == 1 else reversed(state.frontier)) if nodes[r].terminal]
        if not active: break
        leaf = leaves[active[0]]; char = leaf.word[leaf.left] if side == 1 else leaf.word[-1 - leaf.right]
        nxt = BASE.emit(state, side)
        if nxt is None:
            failed = {"step": step, "side": side, "slot": leaf.label, "word": leaf.word, "character": char, "emitter_rejected": True}; break
        state = nxt; events.append({"step": step, "side": side, "slot": leaf.label, "word": leaf.word, "character": char, "residual_after": state.residual})
    return {"rendered": BASE.render(state), "emitter_events": events, "failed_attempt": failed,
            "independent_parse": parse_complete(grammar, BASE.render(state)) is not None, "diagnostic_only": True}


def replay(row):
    state = restore(row); actual = attempts(state)
    return {"same_rendered_tree": BASE.render(state) == row["rendered_tree"], "same_length": state.length == row["length"], "same_residual": state.residual == row["residual"], "same_attempts": actual == row["attempts"], "replayed_attempts": actual}


def expand_edge(grammar, state, index):
    used = {x.word for x in BASE.ordered_leaves(state) if x.label not in FIXED_LABELS}; prior = BASE.leaf_map(state); out = []
    for candidate in BASE.expand(grammar, state, index):
        added = [x for x in candidate.leaves if x.identifier not in prior]
        if any(x.word in used or x.word == x.word[::-1] for x in added if x.label not in FIXED_LABELS): continue
        out.append(candidate)
    return tuple(out)


def solver(grammar, max_states=100000):
    initial = BASE.State((0,), (BASE.Node(0, grammar.start()),), (), "", 0, 0, ()); queue = deque([initial]); seen = set(); exact_rows = {}; admitted = {}
    stats = Counter(states=0, expansions=0, emissions=0, residual_contradictions=0, complete_trees=0, exact_closures=0); deepest = None; deepest_len = -1; contradiction = None; contradiction_len = -1
    while queue and stats["states"] < max_states:
        state = queue.pop(); stats["states"] += 1; key = (state.frontier, state.nodes, state.leaves, state.residual, state.owner, state.length)
        if key in seen: continue
        seen.add(key); offered = attempts(state)
        if state.length > deepest_len: deepest_len = state.length; deepest = snapshot(state, offered)
        if any(a.get("available") and not a.get("accepted") for a in offered) and state.length > contradiction_len: contradiction_len = state.length; contradiction = snapshot(state, offered)
        if BASE.complete(state):
            stats["complete_trees"] += 1
            if MIN_LETTERS <= state.length <= MAX_LETTERS and state.residual == state.residual[::-1]:
                row = audit(grammar, BASE.render(state), "complete_cross_boundary_closure", state.trace)
                if row["independent_exact_audit"]["exact"] and row["independent_parse"]:
                    stats["exact_closures"] += 1; exact_rows.setdefault(row["rendered"], row)
                    if row["mechanically_admitted"]: admitted.setdefault(row["rendered"], row)
            continue
        nodes = BASE.node_map(state); unresolved = [i for i, r in enumerate(state.frontier) if not nodes[r].terminal]; index = None
        if state.residual:
            side = -state.owner; edge = 0 if side == 1 else len(state.frontier) - 1
            if not nodes[state.frontier[edge]].terminal: index = edge
        elif unresolved:
            if not nodes[state.frontier[0]].terminal: index, side = 0, 1
            elif not nodes[state.frontier[-1]].terminal: index, side = len(state.frontier) - 1, -1
            else: index, side = unresolved[0], 1
        if index is not None:
            for nxt in expand_edge(grammar, state, index): queue.append(nxt); stats["expansions"] += 1
        left_ready, right_ready = bool(nodes[state.frontier[0]].terminal), bool(nodes[state.frontier[-1]].terminal)
        for side in ((-state.owner,) if state.residual else (1, -1)):
            if (side == 1 and not left_ready) or (side == -1 and not right_ready): continue
            nxt = BASE.emit(state, side)
            if nxt is not None: queue.append(nxt); stats["emissions"] += 1
            else: stats["residual_contradictions"] += 1
    ledger = {"deepest_state": deepest, "replay": replay(deepest) if deepest else None, "deepest_contradiction_state": contradiction, "deepest_contradiction_replay": replay(contradiction) if contradiction else None}
    return {"stats": dict(stats), "states_exhausted": not queue and stats["states"] < max_states, "exact_closures": list(exact_rows.values()), "mechanically_admitted_closures": list(admitted.values()), "deepest_state_ledger": ledger}


def explicit_control(grammar):
    state = BASE.State((0,), (BASE.Node(0, grammar.start()),), (), "", 0, 0, ()); adjectives = {6:"quiet",5:"careful",4:"patient",3:"brave",2:"young",1:"gentle",0:"steady"}; people = {5:"teacher",4:"guard",3:"writer",2:"singer",1:"farmer",0:"nurse"}; verbs = {4:"greeted",3:"trusted",2:"followed",1:"helped",0:"joined"}
    while True:
        nodes = BASE.node_map(state); unresolved = [i for i, r in enumerate(state.frontier) if not nodes[r].terminal]
        if not unresolved: return state
        index = unresolved[0]; symbol = nodes[state.frontier[index]].symbol; f = dict(symbol.features)
        if symbol.name == "D": ident = "D:now-nurses-record-object-and-contest-remains-unwon"
        elif symbol.name == "NP":
            d, typ = int(f["depth"]), f["type"]; noun = "portrait" if typ == "object" else people[d]; ident = f"NP:{f['role']}:{typ}:{d}:{adjectives[d]}:{noun}:{'relative' if d > 0 else 'base'}"
        elif symbol.name == "REL":
            d = int(f["depth"]); verb = "watched" if f["head_type"] == "object" else verbs[d]; ident = f"REL:{f['head_type']}:{d}:{verb}"
        elif symbol.name == "V": ident = f"V:{f['form']}"
        else: raise AssertionError(symbol.name)
        options = [x for x in BASE.expand(grammar, state, index) if x.trace[-1][1] == ident]
        if len(options) != 1: raise AssertionError(f"nonunique control path {symbol.name} {ident}")
        state = options[0]


def run(max_states=100000):
    # Depth five supplies a 172-letter intact control inside the 180-letter
    # audit window while retaining five nested typed relatives.
    grammar = CrossBoundaryGrammar(5); result = solver(grammar, max_states); control_state = explicit_control(grammar)
    control = audit(grammar, BASE.render(control_state), "complete_cross_boundary_recursive_control", control_state.trace); control.update({"diagnostic_only": True, "grammar_tree_fully_expanded": True, "provenance": {"construction": "explicit five-level typed object-relative control", "shared_tree": True}, "reader_status": "grammar control only; not a palindrome candidate or readability evidence"})
    return {"status": "discourse_now_unwon_intersection", "config": {"min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS, "max_depth": grammar.max_depth, "max_states": max_states, "single_shared_derivation_tree": True, "character_residual_during_derivation": True, "outer_boundary_crossing": "now is matched inside longer unwon", "closure_requires_all_leaves_consumed": True, "independent_complete_reparse": True, "corpus_generation": False}, "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "grammar_sha256": grammar.digest(), "material": "authored ordinary coordinated discourse and typed relative lexicon; no catalogue or known palindrome material"}, "complete_recursive_control": control, **result, "joint_boundary_preflight": boundary_preflight(), "reader_facing_next_operator": "Use deepest_contradiction_state to replace the exact exposed relation or boundary word, then rerun this same-tree emitter.", "scope": "Zero-closure bounded run is diagnostic; no reader-facing palindrome is claimed."}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--max-states", type=int, default=100000); args = parser.parse_args()
    if args.out.exists(): parser.error(f"refusing to overwrite {args.out}")
    result = run(args.max_states); args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps({"out": str(args.out), "states": result["stats"]["states"], "exact_closures": len(result["exact_closures"])}, indent=2))


if __name__ == "__main__": main()
