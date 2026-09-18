"""Full connected-tree scheduler for the explicit see/bees imperative."""
from __future__ import annotations
import argparse, importlib.util, json, re, sys
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = ROOT / "experiments/discourse_now_unwon_intersection_20260913.py"
spec = importlib.util.spec_from_file_location("see_bees_engine_20260913", ENGINE_PATH)
ENGINE = importlib.util.module_from_spec(spec); assert spec.loader is not None
sys.modules[spec.name] = ENGINE; spec.loader.exec_module(ENGINE)
BASE = ENGINE.BASE; ENGINE.FIXED_LABELS.update({"fixed_see", "fixed_the", "fixed_keeper", "fixed_carry", "fixed_cross", "fixed_paint", "fixed_wash", "fixed_feed", "fixed_and", "fixed_clean", "fixed_narrow", "fixed_bright", "fixed_small", "fixed_hungry", "fixed_basket", "fixed_bridge", "fixed_sign", "fixed_dish", "fixed_bees"})
from llm_palindrome.admission import mechanical_admission_checks

MIN_LETTERS, MAX_LETTERS = 100, 180
FRAMES = (("basket", "clean"), ("bridge", "narrow"), ("sign", "bright"), ("dish", "small"))


class SeeBeesGrammar:
    def __init__(self, max_depth=0): self.max_depth = max_depth
    def start(self): return BASE.sym("D", depth=str(self.max_depth))
    def terminal(self, symbol): return symbol.feature("form") if symbol.name == "T" else None
    def productions(self, lhs):
        if lhs.name == "D":
            rhs = [BASE.sym("T", label="fixed_see", form="see"), BASE.sym("T", label="fixed_the", form="the"), BASE.sym("T", label="fixed_keeper", form="keeper"), BASE.sym("T", label="fixed_carry", form="carry"), BASE.sym("NP", role="object_1", frame="basket", type="object", number="sing", depth="0")]
            for i, (verb, frame) in enumerate((("cross", "bridge"), ("paint", "sign"), ("wash", "dish")), 2): rhs.extend((BASE.sym("T", label=f"fixed_and_{i}", form="and"), BASE.sym("T", label=f"fixed_{verb}", form=verb), BASE.sym("NP", role=f"object_{i}", frame=frame, type="object", number="sing", depth="0")))
            rhs.extend((BASE.sym("T", label="fixed_and_5", form="and"), BASE.sym("T", label="fixed_feed", form="feed"), BASE.sym("T", label="fixed_the", form="the"), BASE.sym("T", label="fixed_hungry", form="hungry"), BASE.sym("T", label="fixed_bees", form="bees")))
            return (BASE.Production("D:see-keeper-carry-basket-cross-bridge-paint-sign-wash-dish-feed-bees", lhs, tuple(rhs)),)
        if lhs.name == "NP":
            f = dict(lhs.features); noun, adj = next(x for x in FRAMES if x[0] == f["frame"])
            return (BASE.Production(f"NP:{f['role']}:{noun}:{adj}:base", lhs, (BASE.sym("T", label="fixed_the", form="the"), BASE.sym("T", label=f"fixed_{adj}", form=adj), BASE.sym("T", label=f"fixed_{noun}", form=noun))),)
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
    roles, frames = [], []
    def visit(node):
        if node.symbol.name == "NP": roles.append(dict(node.symbol.features)); frames.append(node.symbol.feature("frame"))
        for child in node.children: visit(child)
    if tree: visit(tree)
    return {"semantic_roles": roles, "event_frames": frames, "relative_count": 0, "agreement_ok": bool(tree and tree.children[0].terminal == "see" and tree.children[2].terminal == "keeper"), "valency_ok": frames == [x[0] for x in FRAMES], "complete_tree": bool(tree and tree.symbol.name == "D")}


def audit(grammar, text, kind, trace):
    tree = parse_complete(grammar, text); tape = "".join(c for c in text.lower() if "a" <= c <= "z"); mismatches = [(i, len(tape) - i - 1) for i in range(len(tape) // 2) if tape[i] != tape[-i - 1]]; exact = {"exact": bool(tape) and not mismatches, "letters": len(tape), "mismatches": mismatches, "normalized_sha256": sha256(tape.encode()).hexdigest()}; witness = semantic_witness(tree); central = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS); codes = [k for k, v in central.items() if not v]
    if tree is None: codes.append("independent_complete_reparse_failed")
    if not witness["agreement_ok"]: codes.append("agreement_failure")
    if not witness["valency_ok"]: codes.append("valency_failure")
    return {"record_kind": kind, "rendered": text, "independent_exact_audit": exact, "independent_parse": tree is not None, "feature_witness": witness, "central_admission": central, "mechanically_admitted": not codes, "rejection_codes": codes, "shared_tree_trace": list(trace), "reader_status": "unreviewed; programmatic checks do not certify readability"}


def explicit_control(grammar):
    state = BASE.State((0,), (BASE.Node(0, grammar.start()),), (), "", 0, 0, ())
    while True:
        nodes = BASE.node_map(state); unresolved = [i for i, r in enumerate(state.frontier) if not nodes[r].terminal]
        if not unresolved: return state
        index = unresolved[0]; symbol = nodes[state.frontier[index]].symbol; f = dict(symbol.features)
        if symbol.name == "D": ident = "D:see-keeper-carry-basket-cross-bridge-paint-sign-wash-dish-feed-bees"
        elif symbol.name == "NP":
            noun, adj = next(x for x in FRAMES if x[0] == f["frame"]); ident = f"NP:{f['role']}:{noun}:{adj}:base"
        else: raise AssertionError(symbol.name)
        options = [x for x in BASE.expand(grammar, state, index) if x.trace[-1][1] == ident]
        if len(options) != 1: raise AssertionError(f"nonunique control path {symbol.name} {ident}")
        state = options[0]


def boundary_preflight():
    grammar = SeeBeesGrammar(); state = explicit_control(grammar); events = []; failed = None
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
    grammar = SeeBeesGrammar(); result = solver(grammar, max_states); control_state = explicit_control(grammar); control = audit(grammar, BASE.render(control_state), "complete_explicit_imperative_control", control_state.trace); control.update({"diagnostic_only": True, "grammar_tree_fully_expanded": True, "provenance": {"construction": "five explicit ordinary imperative action/object clauses", "shared_tree": True}, "reader_status": "grammar control only; not a palindrome candidate or readability evidence"})
    return {"status": "imperative_see_bees_clauses_intersection", "config": {"min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS, "max_depth": 0, "max_states": max_states, "single_shared_derivation_tree": True, "character_residual_during_derivation": True, "outer_boundary_crossing": "see is matched inside longer lexicon-backed bees", "explicit_clause_count": 5, "recursive_padding": False, "closure_requires_all_leaves_consumed": True, "independent_complete_reparse": True, "corpus_generation": False}, "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "grammar_sha256": grammar.digest(), "material": "authored ordinary imperative action clauses and typed object frames; no catalogue or known palindrome material"}, "complete_recursive_control": control, **result, "joint_boundary_preflight": boundary_preflight(), "reader_facing_next_operator": "Use the replayed deepest contradiction to alter the exact exposed clause boundary, then rerun the full scheduler.", "scope": "Zero-closure bounded run is diagnostic; no reader-facing palindrome is claimed."}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--max-states", type=int, default=100000); args = parser.parse_args()
    if args.out.exists(): parser.error(f"refusing to overwrite {args.out}")
    result = run(args.max_states); args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps({"out": str(args.out), "states": result["stats"]["states"], "exact_closures": len(result["exact_closures"])}, indent=2))


if __name__ == "__main__": main()
