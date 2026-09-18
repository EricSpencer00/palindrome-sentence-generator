"""Reconstructible evidence for the deepest actual causal-bridge search state.

This is an instrumentation repair, not a new generation success. Each search
transition records its grammar expansions and character emission. The deepest
state is replayed independently from the initial root; the next contradiction
is extracted from the replayed state rather than inferred from a counter.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"experiments/causal_subject_interior_bridge_20260913.py"
SPEC=importlib.util.spec_from_file_location("causal_capture_parent",SOURCE)
CAUSAL=importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name]=CAUSAL
SPEC.loader.exec_module(CAUSAL)
BASE,PAIR,PARENT=CAUSAL.BASE,CAUSAL.PAIR,CAUSAL.PARENT


def initial(grammar):
    return BASE.State((0,),(BASE.Node(0,grammar.start()),),(),"",0,0,())


def digest_state(state):
    return sha256(json.dumps(asdict(state),sort_keys=True).encode()).hexdigest()


def transition_actions(before,after):
    added=after.trace[len(before.trace):]
    if added:
        assert after.length==before.length
        return tuple({"action":"expand","node":node,"production":production} for node,production in added)
    assert after.length==before.length+1
    prior=BASE.leaf_map(before)
    changed=[leaf for leaf in after.leaves if prior[leaf.identifier]!=leaf]
    assert len(changed)==1
    leaf=changed[0]
    old=prior[leaf.identifier]
    side=1 if leaf.left==old.left+1 else -1
    char=old.word[old.left] if side==1 else old.word[-1-old.right]
    return ({"action":"emit","node":leaf.identifier,"side":side,"character":char,
             "word":leaf.word,"debt_before":before.residual,"debt_after":after.residual},)


def replay(grammar,actions):
    state=initial(grammar)
    emitted=0
    for action in actions:
        if action["action"]=="expand":
            edge=state.frontier.index(action["node"])
            if edge not in (0,len(state.frontier)-1):
                raise ValueError("ledger expands an unexposed interior node")
            candidates=BASE.expand(grammar,state,edge)
            matches=[candidate for candidate in candidates if candidate.trace[-1][1]==action["production"]]
            if len(matches)!=1:
                raise ValueError("ledger grammar production is unavailable or ambiguous")
            state=matches[0]
        elif action["action"]=="emit":
            side,edge=PARENT.active_edge(state)
            if side!=action["side"] or state.frontier[edge]!=action["node"]:
                raise ValueError("ledger character does not come from the required exposed edge")
            leaf=BASE.leaf_map(state)[action["node"]]
            char=leaf.word[leaf.left] if side==1 else leaf.word[-1-leaf.right]
            if char!=action["character"] or leaf.word!=action["word"] or state.residual!=action["debt_before"]:
                raise ValueError("ledger emission differs from its grammar leaf or character debt")
            next_state=BASE.emit(state,side)
            if next_state is None or next_state.residual!=action["debt_after"]:
                raise ValueError("ledger contains an unmatched character emission")
            state=next_state
            emitted+=1
        else:
            raise ValueError("unknown ledger action")
    assert emitted==state.length
    return state


def witness(grammar,state,ledger):
    restored=replay(grammar,ledger)
    if restored!=state:
        raise AssertionError("deepest-state ledger does not reconstruct the search state")
    nodes=BASE.node_map(restored)
    side,edge=PARENT.active_edge(restored)
    node=nodes[restored.frontier[edge]] if restored.frontier else None
    conflict=None
    if node and node.terminal and restored.residual:
        leaf=BASE.leaf_map(restored)[node.identifier]
        char=leaf.word[leaf.left] if side==1 else leaf.word[-1-leaf.right]
        last_emission=next((a for a in reversed(ledger) if a["action"]=="emit"),None)
        conflict={"expected_character":restored.residual[0],"opposing_character":char,
                  "debt_source_word":last_emission["word"] if last_emission else None,
                  "opposing_word":leaf.word,"opposing_side":side,
                  "immediate_emitter_rejects":BASE.emit(restored,side) is None}
    return {"diagnostic_only":True,"record_kind":"actual_search_frontier_replay",
            "emitted_letters":restored.length,"matched_pairs":restored.length//2,
            "ledger":list(ledger),"replayed_state":asdict(restored),
            "state_sha256":digest_state(restored),"replay_verified":True,"next_character_conflict":conflict,
            "assigned_words_in_tree_order":[{"word":leaf.word,"emitted_left":leaf.left,"emitted_right":leaf.right}
                                             for leaf in BASE.ordered_leaves(restored)],
            "candidate":False,"complete_tree":BASE.complete(restored)}


def solve(grammar,max_states=100000):
    root=initial(grammar)
    stack=[(root,())]
    best,best_ledger=root,()
    stats=Counter(states=0,complete_trees=0,deepest_emitted_letters=0)
    exact,admitted={},{}
    while stack and stats["states"]<max_states:
        state,ledger=stack.pop()
        stats["states"]+=1
        if state.length>best.length:
            best,best_ledger=state,ledger
        stats["deepest_emitted_letters"]=best.length
        if state.length>PARENT.MAX_LETTERS:
            continue
        if BASE.complete(state):
            stats["complete_trees"]+=1
            if state.residual==state.residual[::-1]:
                text=BASE.render(state)
                row=PARENT.audit(grammar,text,"complete_shared_tree_closure",state.trace)
                assert row["independent_exact_audit"]["exact"]
                exact.setdefault(text,row)
                if row["mechanically_admitted"]:
                    admitted.setdefault(text,row)
            continue
        children=PAIR.successors(grammar,state,stats)
        stack.extend((child,ledger+transition_actions(state,child)) for child in reversed(children))
    return {"stats":dict(stats),"states_exhausted":not stack,"exact_closures":list(exact.values()),
            "mechanically_admitted_closures":list(admitted.values()),
            "deepest_actual_search_witness":witness(grammar,best,best_ledger)}


def run(max_states=100000):
    grammar=CAUSAL.Grammar()
    return {"method":"causal_subject_bridge_with_replayable_actual_search_witness",
        "instrumentation_only":True,"config":{"max_states":max_states},
        "provenance":{"capture_sha256":sha256(Path(__file__).read_bytes()).hexdigest(),
                      "generator_sha256":sha256(SOURCE.read_bytes()).hexdigest(),
                      "grammar_sha256":grammar.digest(),"control_used_as_search_seed":False},
        **solve(grammar,max_states)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--max-states",default=100000,type=int)
    args=parser.parse_args()
    if args.out.exists():parser.error("refusing to overwrite an existing artifact")
    result=run(args.max_states)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"out":str(args.out),"stats":result["stats"],
        "replay_verified":result["deepest_actual_search_witness"]["replay_verified"],
        "conflict":result["deepest_actual_search_witness"]["next_character_conflict"]}))


if __name__=="__main__":main()
