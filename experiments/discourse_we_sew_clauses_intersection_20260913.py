"""Full-tree scheduler for explicit sewing/work clauses.

This successor removes recursive relative padding.  Its single connected tree
contains five ordinary clauses with explicit object referents:
``We start the new garment and mend the clean cloth and mark the bright banner
and fold the soft fabric and wash the plain shirt and sew.``
"""
from __future__ import annotations
import argparse, importlib.util, json, re, sys
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = ROOT / "experiments/discourse_now_unwon_intersection_20260913.py"
spec = importlib.util.spec_from_file_location("clause_sew_engine_20260913", ENGINE_PATH)
ENGINE = importlib.util.module_from_spec(spec); assert spec.loader is not None
sys.modules[spec.name] = ENGINE; spec.loader.exec_module(ENGINE)
BASE = ENGINE.BASE; ENGINE.FIXED_LABELS.update({"fixed_we", "fixed_start", "fixed_mend", "fixed_mark", "fixed_fold", "fixed_wash", "fixed_sew", "fixed_and_2", "fixed_and_3", "fixed_and_4", "fixed_and_5", "fixed_and_6"})
from llm_palindrome.admission import mechanical_admission_checks

MIN_LETTERS, MAX_LETTERS = 100, 180
FRAMES = (("start", "garment"), ("mend", "cloth"), ("mark", "banner"), ("fold", "fabric"), ("wash", "shirt"))
ADJECTIVES = ("new", "clean", "bright", "soft", "plain", "fresh", "fine", "heavy")


class ClauseSewGrammar:
    def __init__(self, max_depth=0): self.max_depth = max_depth
    def start(self): return BASE.sym("D", depth=str(self.max_depth))
    def terminal(self, symbol): return symbol.feature("form") if symbol.name == "T" else None
    def productions(self, lhs):
        if lhs.name == "D":
            rhs = [BASE.sym("T", label="fixed_we", form="we")]
            for index, (verb, noun) in enumerate(FRAMES, 1):
                if index > 1: rhs.append(BASE.sym("T", label=f"fixed_and_{index}", form="and"))
                rhs.extend((BASE.sym("T", label=f"fixed_{verb}", form=verb), BASE.sym("NP", role=f"object_{index}", frame=verb, type="object", number="sing", depth="0")))
            rhs.extend((BASE.sym("T", label="fixed_and_6", form="and"), BASE.sym("T", label="fixed_sew", form="sew")))
            return (BASE.Production("D:we-five-explicit-work-clauses-and-sew", lhs, tuple(rhs)),)
        if lhs.name == "NP":
            f = dict(lhs.features); frame, noun = f["frame"], dict(FRAMES)[f["frame"]]; rows = []
            for adjective in ADJECTIVES:
                prefix = (BASE.sym("T", label="det", form="the"), BASE.sym("T", label="adj", form=adjective), BASE.sym("T", label=f"noun_{f['role']}", form=noun))
                rows.append(BASE.Production(f"NP:{f['role']}:{frame}:{adjective}:{noun}:base", lhs, prefix))
            return tuple(rows)
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
    roles, frames, agreement, valency = [], [], True, True
    def visit(node):
        nonlocal agreement, valency
        if node.symbol.name == "NP": roles.append(dict(node.symbol.features)); frames.append(node.symbol.feature("frame"))
        for child in node.children: visit(child)
    if tree: visit(tree)
    expected = [x[0] for x in FRAMES]
    root_ok = bool(tree and tree.children[0].terminal == "we" and tree.children[-1].terminal == "sew")
    if frames != expected: valency = False
    return {"semantic_roles": roles, "event_frames": frames, "relative_count": 0, "agreement_ok": agreement and root_ok, "valency_ok": valency, "complete_tree": bool(tree and tree.symbol.name == "D")}


def audit(grammar, text, kind, trace):
    tree = parse_complete(grammar, text); tape = "".join(c for c in text.lower() if "a" <= c <= "z"); mismatches = [(i, len(tape) - i - 1) for i in range(len(tape) // 2) if tape[i] != tape[-i - 1]]; exact = {"exact": bool(tape) and not mismatches, "letters": len(tape), "mismatches": mismatches, "normalized_sha256": sha256(tape.encode()).hexdigest()}; witness = semantic_witness(tree); central = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS); codes = [k for k, v in central.items() if not v]
    if tree is None: codes.append("independent_complete_reparse_failed")
    if not witness["agreement_ok"]: codes.append("agreement_failure")
    if not witness["valency_ok"]: codes.append("valency_failure")
    return {"record_kind": kind, "rendered": text, "independent_exact_audit": exact, "independent_parse": tree is not None, "feature_witness": witness, "central_admission": central, "mechanically_admitted": not codes, "rejection_codes": codes, "shared_tree_trace": list(trace), "reader_status": "unreviewed; programmatic checks do not certify readability"}


def explicit_control(grammar):
    state = BASE.State((0,), (BASE.Node(0, grammar.start()),), (), "", 0, 0, ()); choices = ("new", "clean", "bright", "soft", "plain")
    while True:
        nodes = BASE.node_map(state); unresolved = [i for i, r in enumerate(state.frontier) if not nodes[r].terminal]
        if not unresolved: return state
        index = unresolved[0]; symbol = nodes[state.frontier[index]].symbol; f = dict(symbol.features)
        if symbol.name == "D": ident = "D:we-five-explicit-work-clauses-and-sew"
        elif symbol.name == "NP":
            frame = f["frame"]; noun = dict(FRAMES)[frame]; adjective = choices[int(f["role"].split("_")[-1]) - 1]; ident = f"NP:{f['role']}:{frame}:{adjective}:{noun}:base"
        else: raise AssertionError(symbol.name)
        options = [x for x in BASE.expand(grammar, state, index) if x.trace[-1][1] == ident]
        if len(options) != 1: raise AssertionError(f"nonunique control path {symbol.name} {ident}")
        state = options[0]


def boundary_preflight():
    grammar = ClauseSewGrammar(); state = explicit_control(grammar); events = []; failed = None
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
    grammar = ClauseSewGrammar(); result = solver(grammar, max_states); control_state = explicit_control(grammar); control = audit(grammar, BASE.render(control_state), "complete_explicit_clause_control", control_state.trace); control.update({"diagnostic_only": True, "grammar_tree_fully_expanded": True, "provenance": {"construction": "five explicit sewing/work clauses with typed object frames", "shared_tree": True}, "reader_status": "grammar control only; not a palindrome candidate or readability evidence"})
    return {"status": "discourse_we_sew_clauses_intersection", "config": {"min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS, "max_depth": 0, "max_states": max_states, "single_shared_derivation_tree": True, "character_residual_during_derivation": True, "outer_boundary_crossing": "we is matched through longer sew and into start", "explicit_clause_count": 5, "recursive_padding": False, "closure_requires_all_leaves_consumed": True, "independent_complete_reparse": True, "corpus_generation": False}, "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "grammar_sha256": grammar.digest(), "material": "authored ordinary sewing/work clauses and explicit typed object frames; no catalogue or known palindrome material"}, "complete_recursive_control": control, **result, "joint_boundary_preflight": boundary_preflight(), "reader_facing_next_operator": "Use the replayed deepest contradiction to change the exact exposed clause boundary, then rerun the full scheduler.", "scope": "Zero-closure bounded run is diagnostic; no reader-facing palindrome is claimed."}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--max-states", type=int, default=100000); args = parser.parse_args()
    if args.out.exists(): parser.error(f"refusing to overwrite {args.out}")
    result = run(args.max_states); args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps({"out": str(args.out), "states": result["stats"]["states"], "exact_closures": len(result["exact_closures"])}, indent=2))


if __name__ == "__main__": main()
