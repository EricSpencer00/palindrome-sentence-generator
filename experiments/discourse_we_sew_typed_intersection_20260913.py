"""Typed sewing-frame successor for the crossed ``we``/``sew`` route.

The loose-object predecessor is retained as a diagnostic only.  Here every
object NP is a garment/material and every object-gap relative uses an authored
sewing-compatible relation, so ``We start the new garment ... and sew`` is a
coherent ordinary construction at every depth.
"""
from __future__ import annotations
import argparse, importlib.util, json, re, sys
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = ROOT / "experiments/discourse_now_unwon_intersection_20260913.py"
spec = importlib.util.spec_from_file_location("typed_sew_engine_20260913", ENGINE_PATH)
ENGINE = importlib.util.module_from_spec(spec); assert spec.loader is not None
sys.modules[spec.name] = ENGINE; spec.loader.exec_module(ENGINE)
BASE = ENGINE.BASE; ENGINE.FIXED_LABELS.update({"fixed_we", "fixed_start", "fixed_sew"})
from llm_palindrome.admission import mechanical_admission_checks

MIN_LETTERS, MAX_LETTERS = 100, 180
OBJECT_NOUNS = ("garment", "fabric", "banner", "shirt", "dress", "cloth")
PERSON_NOUNS = ("tailor", "worker", "designer", "artist", "maker", "teacher", "guard")
OBJECT_ADJECTIVES = ("new", "plain", "clean", "bright", "small", "soft", "fine", "heavy")
PERSON_ADJECTIVES = ("careful", "patient", "brave", "young", "gentle", "steady", "honest", "alert")
OBJECT_VERBS = ("folded", "washed", "mended", "cut", "marked", "sewed")
PERSON_VERBS = ("helped", "guided", "trusted", "joined", "followed", "met")


class TypedSewGrammar:
    def __init__(self, max_depth=6): self.max_depth = max_depth
    def start(self): return BASE.sym("D", depth=str(self.max_depth))
    def terminal(self, symbol): return symbol.feature("form") if symbol.name == "T" else None
    def productions(self, lhs):
        f = dict(lhs.features)
        if lhs.name == "D":
            return (BASE.Production("D:we-start-garment-and-sew", lhs, (BASE.sym("T", label="fixed_we", form="we"), BASE.sym("T", label="fixed_start", form="start"), BASE.sym("NP", role="object", type="object", number="sing", depth=f["depth"]), BASE.sym("T", label="fixed_and", form="and"), BASE.sym("T", label="fixed_sew", form="sew"))),)
        if lhs.name == "NP":
            role, typ, depth = f["role"], f["type"], int(f["depth"]); nouns = OBJECT_NOUNS if typ == "object" else PERSON_NOUNS; adjs = OBJECT_ADJECTIVES if typ == "object" else PERSON_ADJECTIVES; det = "the" if typ == "object" else "a"; rows = []
            for adj in adjs:
                for noun in nouns:
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
            seen.add(symbol); prods = self.productions(symbol); rows.append((symbol.name, symbol.features, [(p.identifier, [x.name + str(x.features) for x in p.rhs]) for p in prods])); queue.extend(x for p in prods for x in p.rhs if self.terminal(x) is None)
        return sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def parse_complete(grammar, text):
    if text != text.strip() or re.sub(r"[a-z ,;.!?]", "", text.lower()): return None
    tokens = tuple(re.findall(r"[a-z]+", text.lower())); memo = {}
    def parses(symbol, offset):
        key = (symbol, offset)
        if key in memo: return memo[key]
        term = grammar.terminal(symbol)
        if term is not None:
            ans = ((BASE.Tree(symbol, f"lex:{term}", (), term), offset + 1),) if offset < len(tokens) and tokens[offset] == term else (); memo[key] = ans; return ans
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
    root_ok = bool(tree and tree.children[0].terminal == "we" and tree.children[1].terminal == "start" and tree.children[-1].terminal == "sew")
    return {"semantic_roles": roles, "relative_count": relatives, "agreement_ok": agreement and root_ok, "valency_ok": valency, "complete_tree": bool(tree and tree.symbol.name == "D")}


def audit(grammar, text, kind, trace):
    tree = parse_complete(grammar, text); tape = "".join(c for c in text.lower() if "a" <= c <= "z"); mismatches = [(i, len(tape) - i - 1) for i in range(len(tape) // 2) if tape[i] != tape[-i - 1]]; exact = {"exact": bool(tape) and not mismatches, "letters": len(tape), "mismatches": mismatches, "normalized_sha256": sha256(tape.encode()).hexdigest()}; witness = semantic_witness(tree); central = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS); codes = [k for k, v in central.items() if not v]
    if tree is None: codes.append("independent_complete_reparse_failed")
    if not witness["agreement_ok"]: codes.append("agreement_failure")
    if not witness["valency_ok"]: codes.append("valency_failure")
    return {"record_kind": kind, "rendered": text, "independent_exact_audit": exact, "independent_parse": tree is not None, "feature_witness": witness, "central_admission": central, "mechanically_admitted": not codes, "rejection_codes": codes, "shared_tree_trace": list(trace), "reader_status": "unreviewed; programmatic checks do not certify readability"}


def explicit_control(grammar):
    state = BASE.State((0,), (BASE.Node(0, grammar.start()),), (), "", 0, 0, ()); object_adjectives = {6:"new",5:"clean",4:"bright",3:"small",2:"soft",1:"fine",0:"heavy"}; person_adjectives = {5:"careful",4:"patient",3:"brave",2:"young",1:"gentle",0:"steady"}; people = {5:"tailor",4:"worker",3:"designer",2:"artist",1:"maker",0:"teacher"}; verbs = {4:"guided",3:"trusted",2:"joined",1:"followed",0:"met"}
    while True:
        nodes = BASE.node_map(state); unresolved = [i for i, r in enumerate(state.frontier) if not nodes[r].terminal]
        if not unresolved: return state
        index = unresolved[0]; symbol = nodes[state.frontier[index]].symbol; f = dict(symbol.features)
        if symbol.name == "D": ident = "D:we-start-garment-and-sew"
        elif symbol.name == "NP":
            d, typ = int(f["depth"]), f["type"]; noun = "garment" if typ == "object" else people[d]; adjective = object_adjectives[d] if typ == "object" else person_adjectives[d]; ident = f"NP:{f['role']}:{typ}:{d}:{adjective}:{noun}:{'relative' if d > 0 else 'base'}"
        elif symbol.name == "REL":
            d = int(f["depth"]); verb = "mended" if f["head_type"] == "object" else verbs[d]; ident = f"REL:{f['head_type']}:{d}:{verb}"
        elif symbol.name == "V": ident = f"V:{f['form']}"
        else: raise AssertionError(symbol.name)
        options = [x for x in BASE.expand(grammar, state, index) if x.trace[-1][1] == ident]
        if len(options) != 1: raise AssertionError(f"nonunique control path {symbol.name} {ident}")
        state = options[0]


def boundary_preflight():
    grammar = TypedSewGrammar(0); state = explicit_control(grammar); events = []; failed = None
    for step in range(1, 31):
        side = 1 if step % 2 else -1; nodes = BASE.node_map(state); active = [r for r in (state.frontier if side == 1 else reversed(state.frontier)) if nodes[r].terminal]
        if not active: break
        leaf = BASE.leaf_map(state)[active[0]]; char = leaf.word[leaf.left] if side == 1 else leaf.word[-1 - leaf.right]; nxt = BASE.emit(state, side)
        if nxt is None: failed = {"step": step, "side": side, "slot": leaf.label, "word": leaf.word, "character": char, "emitter_rejected": True}; break
        state = nxt; events.append({"step": step, "side": side, "slot": leaf.label, "word": leaf.word, "character": char, "residual_after": state.residual})
    return {"rendered": BASE.render(state), "emitter_events": events, "failed_attempt": failed, "independent_parse": parse_complete(grammar, BASE.render(state)) is not None, "diagnostic_only": True}


def solver(grammar, max_states=100000):
    prior = ENGINE.audit; ENGINE.audit = audit
    try: return ENGINE.solver(grammar, max_states=max_states)
    finally: ENGINE.audit = prior


def run(max_states=100000):
    grammar = TypedSewGrammar(6); result = solver(grammar, max_states); control_grammar = TypedSewGrammar(5); control_state = explicit_control(control_grammar); control = audit(control_grammar, BASE.render(control_state), "complete_typed_sew_recursive_control", control_state.trace); control.update({"diagnostic_only": True, "grammar_tree_fully_expanded": True, "provenance": {"construction": "explicit five-level sewing-licensed typed object-relative control", "shared_tree": True}, "reader_status": "grammar control only; not a palindrome candidate or readability evidence"})
    return {"status": "discourse_we_sew_typed_intersection", "config": {"min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS, "max_depth": grammar.max_depth, "max_states": max_states, "single_shared_derivation_tree": True, "character_residual_during_derivation": True, "outer_boundary_crossing": "we is matched through longer sew and into start", "typed_sewing_object_frame": True, "closure_requires_all_leaves_consumed": True, "independent_complete_reparse": True, "corpus_generation": False}, "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "grammar_sha256": grammar.digest(), "material": "authored sewing-licensed garment/material lexicon and typed relatives; no catalogue or known palindrome material"}, "complete_recursive_control": control, **result, "joint_boundary_preflight": boundary_preflight(), "reader_facing_next_operator": "Use the replayed deepest contradiction to alter the exact exposed sewing relation, then rerun the full scheduler.", "scope": "Zero-closure bounded run is diagnostic; no reader-facing palindrome is claimed."}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--max-states", type=int, default=100000); args = parser.parse_args()
    if args.out.exists(): parser.error(f"refusing to overwrite {args.out}")
    result = run(args.max_states); args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps({"out": str(args.out), "states": result["stats"]["states"], "exact_closures": len(result["exact_closures"])}, indent=2))


if __name__ == "__main__": main()
