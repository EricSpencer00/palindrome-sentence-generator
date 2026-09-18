"""Full connected-tree scheduler for the lexicon-backed ``we``/``anew`` seam.

The authored ordinary discourse is ``We assess ... and we begin anew.``  The
opening ``we`` is matched against the suffix of longer ``anew``; the answer is
not a token-order mirror.  This module supplies a fresh typed grammar while
reusing only the repository's audited single-tree character emitter.
"""
from __future__ import annotations
import argparse, importlib.util, json, re, sys
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = ROOT / "experiments/discourse_now_unwon_intersection_20260913.py"
spec = importlib.util.spec_from_file_location("we_anew_engine_20260913", ENGINE_PATH)
ENGINE = importlib.util.module_from_spec(spec); assert spec.loader is not None
sys.modules[spec.name] = ENGINE; spec.loader.exec_module(ENGINE)
BASE = ENGINE.BASE
ENGINE.FIXED_LABELS.update({"fixed_we", "fixed_assess", "fixed_begin", "fixed_anew"})
from llm_palindrome.admission import mechanical_admission_checks

MIN_LETTERS, MAX_LETTERS = 100, 180
NOUNS = (("portrait", "object"), ("packet", "object"), ("artifact", "object"), ("cabinet", "object"), ("garment", "object"), ("helmet", "object"), ("statue", "object"), ("teacher", "person"), ("guard", "person"), ("writer", "person"), ("singer", "person"), ("farmer", "person"), ("nurse", "person"))
ADJECTIVES = ("quiet", "careful", "patient", "brave", "young", "gentle", "steady", "honest")
OBJECT_VERBS = ("watched", "painted", "saved", "praised")
PERSON_VERBS = ("greeted", "trusted", "followed", "helped", "joined")


class WeAnewGrammar:
    def __init__(self, max_depth=6): self.max_depth = max_depth
    def start(self): return BASE.sym("D", depth=str(self.max_depth))
    def terminal(self, symbol): return symbol.feature("form") if symbol.name == "T" else None
    def productions(self, lhs):
        f = dict(lhs.features)
        if lhs.name == "D":
            return (BASE.Production("D:we-assess-object-and-we-begin-anew", lhs, (
                BASE.sym("T", label="fixed_we", form="we"), BASE.sym("T", label="fixed_assess", form="assess"),
                BASE.sym("NP", role="object", type="object", number="sing", depth=f["depth"]),
                BASE.sym("T", label="fixed_and", form="and"), BASE.sym("T", label="fixed_we", form="we"),
                BASE.sym("T", label="fixed_begin", form="begin"), BASE.sym("T", label="fixed_anew", form="anew"))),)
        if lhs.name == "NP":
            role, typ, depth = f["role"], f["type"], int(f["depth"]); det = "the" if role == "object" else "a"; rows = []
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
        from collections import deque
        seen, queue, rows = set(), deque([self.start()]), []
        while queue:
            symbol = queue.popleft()
            if symbol in seen: continue
            seen.add(symbol); prods = self.productions(symbol)
            rows.append((symbol.name, symbol.features, [(p.identifier, [x.name + str(x.features) for x in p.rhs]) for p in prods]))
            queue.extend(x for p in prods for x in p.rhs if self.terminal(x) is None)
        return sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def parse_complete(grammar, text):
    return ENGINE.parse_complete(grammar, text) if hasattr(ENGINE, "parse_complete") else None


def exact_audit(text): return ENGINE.exact_audit(text)


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
    root_ok = bool(tree and tree.children[0].terminal == "we" and tree.children[1].terminal == "assess" and tree.children[-2].terminal == "begin")
    return {"semantic_roles": roles, "relative_count": relatives, "agreement_ok": agreement and root_ok, "valency_ok": valency, "complete_tree": bool(tree and tree.symbol.name == "D")}


def audit(grammar, text, kind, trace):
    tree = parse_complete(grammar, text); exact = exact_audit(text); witness = semantic_witness(tree); central = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS); codes = [k for k, v in central.items() if not v]
    if tree is None: codes.append("independent_complete_reparse_failed")
    if not witness["agreement_ok"]: codes.append("agreement_failure")
    if not witness["valency_ok"]: codes.append("valency_failure")
    return {"record_kind": kind, "rendered": text, "independent_exact_audit": exact, "independent_parse": tree is not None, "feature_witness": witness, "central_admission": central, "mechanically_admitted": not codes, "rejection_codes": codes, "shared_tree_trace": list(trace), "reader_status": "unreviewed; programmatic checks do not certify readability"}


def explicit_control(grammar):
    state = BASE.State((0,), (BASE.Node(0, grammar.start()),), (), "", 0, 0, ()); adjectives = {6:"quiet",5:"careful",4:"patient",3:"brave",2:"young",1:"gentle",0:"steady"}; people = {5:"teacher",4:"guard",3:"writer",2:"singer",1:"farmer",0:"nurse"}; verbs = {4:"greeted",3:"trusted",2:"followed",1:"helped",0:"joined"}
    while True:
        nodes = BASE.node_map(state); unresolved = [i for i, r in enumerate(state.frontier) if not nodes[r].terminal]
        if not unresolved: return state
        index = unresolved[0]; symbol = nodes[state.frontier[index]].symbol; f = dict(symbol.features)
        if symbol.name == "D": ident = "D:we-assess-object-and-we-begin-anew"
        elif symbol.name == "NP":
            d, typ = int(f["depth"]), f["type"]; noun = "portrait" if typ == "object" else people[d]; ident = f"NP:{f['role']}:{typ}:{d}:{adjectives[d]}:{noun}:{'relative' if d > 0 else 'base'}"
        elif symbol.name == "REL":
            d = int(f["depth"]); verb = "watched" if f["head_type"] == "object" else verbs[d]; ident = f"REL:{f['head_type']}:{d}:{verb}"
        elif symbol.name == "V": ident = f"V:{f['form']}"
        else: raise AssertionError(symbol.name)
        options = [x for x in BASE.expand(grammar, state, index) if x.trace[-1][1] == ident]
        if len(options) != 1: raise AssertionError(f"nonunique control path {symbol.name} {ident}")
        state = options[0]


def boundary_preflight():
    grammar = WeAnewGrammar(0); state = explicit_control(grammar); events = []; failed = None
    for step in range(1, 31):
        side = 1 if step % 2 else -1; nodes, leaves = BASE.node_map(state), BASE.leaf_map(state); active = [r for r in (state.frontier if side == 1 else reversed(state.frontier)) if nodes[r].terminal]
        if not active: break
        leaf = BASE.leaf_map(state)[active[0]]; char = leaf.word[leaf.left] if side == 1 else leaf.word[-1 - leaf.right]; nxt = BASE.emit(state, side)
        if nxt is None: failed = {"step": step, "side": side, "slot": leaf.label, "word": leaf.word, "character": char, "emitter_rejected": True}; break
        state = nxt; events.append({"step": step, "side": side, "slot": leaf.label, "word": leaf.word, "character": char, "residual_after": state.residual})
    return {"rendered": BASE.render(state), "emitter_events": events, "failed_attempt": failed, "independent_parse": parse_complete(grammar, BASE.render(state)) is not None, "diagnostic_only": True}


def solver(grammar, max_states=100000):
    """Run the shared full-tree scheduler with this route's closure audit."""
    prior_audit = ENGINE.audit
    ENGINE.audit = audit
    try:
        return ENGINE.solver(grammar, max_states=max_states)
    finally:
        ENGINE.audit = prior_audit


def run(max_states=100000):
    grammar = WeAnewGrammar(6); result = solver(grammar, max_states=max_states)
    control_state = explicit_control(WeAnewGrammar(5)); control = audit(WeAnewGrammar(5), BASE.render(control_state), "complete_we_anew_recursive_control", control_state.trace); control.update({"diagnostic_only": True, "grammar_tree_fully_expanded": True, "provenance": {"construction": "explicit five-level typed object-relative control", "shared_tree": True}, "reader_status": "grammar control only; not a palindrome candidate or readability evidence"})
    return {"status": "discourse_we_anew_intersection", "config": {"min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS, "max_depth": grammar.max_depth, "max_states": max_states, "single_shared_derivation_tree": True, "character_residual_during_derivation": True, "outer_boundary_crossing": "we is matched inside longer lexicon-backed anew", "closure_requires_all_leaves_consumed": True, "independent_complete_reparse": True, "corpus_generation": False}, "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "grammar_sha256": grammar.digest(), "material": "authored ordinary coordinated discourse and typed relative lexicon; no catalogue or known palindrome material"}, "complete_recursive_control": control, **result, "joint_boundary_preflight": boundary_preflight(), "reader_facing_next_operator": "Use the replayed deepest contradiction to alter the exact exposed typed relation or boundary word, then rerun the full scheduler.", "scope": "Zero-closure bounded run is diagnostic; no reader-facing palindrome is claimed."}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--max-states", type=int, default=100000); args = parser.parse_args()
    if args.out.exists(): parser.error(f"refusing to overwrite {args.out}")
    result = run(args.max_states); args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps({"out": str(args.out), "states": result["stats"]["states"], "exact_closures": len(result["exact_closures"])}, indent=2))


if __name__ == "__main__": main()
