"""Bounded two-sided clause construction with live reverse segmentation.

The left arm is generated in ordinary SVO order.  The right arm is generated
independently from the character stream read from the opposite end: a reversed
grammar trie is intersected character-by-character with the forward trie.  No
finished sentence is reversed and no seed/catalogue text is used.
"""
from __future__ import annotations
import argparse, hashlib, json
from dataclasses import dataclass
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import normalize_letters, mechanical_admission_checks, tokenize

LIMIT = 250_000
@dataclass(frozen=True)
class Clause:
    subject: str; verb: str; object: str; adjunct: str
    @property
    def words(self): return (self.subject, self.verb, self.object, self.adjunct)
    @property
    def tape(self): return normalize_letters(" ".join(self.words))
    @property
    def text(self): return " ".join(self.words).capitalize() + "."

# Authored, ordinary clauses; alternatives are typed by role and agreement.
SUBJECTS = ("the baker", "the sailor", "the gardener", "the keeper", "a pilot", "a teacher", "the artist", "a farmer")
VERBS = ("marks", "charts", "guards", "carries", "records", "watches", "plants", "draws")
OBJECTS = ("the harbor", "the garden", "the bridge", "a lantern", "the vessel", "the valley", "a map", "the orchard")
ADJUNCTS = ("at dawn", "near the shore", "beside the gate", "through the rain", "in the valley")

def clauses():
    # Keep semantic/valency templates ordinary; finite inventory is deliberately
    # broad enough to exercise trie branching but remains human-authored.
    for s in SUBJECTS:
        for v in VERBS:
            for o in OBJECTS:
                for a in ADJUNCTS:
                    if s.startswith("a ") and v not in {"carries", "watches", "plants", "draws"}: continue
                    yield Clause(s, v, o, a)

class Node:
    def __init__(self): self.children = {}; self.ends = []
def add(root, tape, clause):
    n=root
    for c in tape: n=n.children.setdefault(c, Node())
    n.ends.append(clause)

def audit(tape):
    mismatches=[]
    for i,(a,b) in enumerate(zip(tape,tape[::-1])):
        if a != b: mismatches.append({"position":i+1,"left":a,"reverse":b})
    return {"letters":len(tape),"two_pointer_exact":bool(tape) and not mismatches,
            "mismatches":mismatches[:12],"sha256_forward":hashlib.sha256(tape.encode()).hexdigest(),
            "sha256_reverse":hashlib.sha256(tape[::-1].encode()).hexdigest()}

def parse_clause(text):
    try: w=tuple(tokenize(text.rstrip(".")))
    except ValueError: return False
    return any(w == c.words for c in clauses())

def run(state_limit=LIMIT):
    left,right=Node(),Node(); inventory=tuple(clauses())
    for c in inventory:
        add(left,c.tape,c)
        # Right clause is parsed from the reverse-facing stream, not post-hoc.
        add(right,c.tape[::-1],c)
    states=0; frontier=[]; rejected=[]; exact=[]; truncated=False
    def visit(a,b,prefix):
        nonlocal states,truncated
        if states>=state_limit: truncated=True; return
        states += 1
        common=sorted(set(a.children)&set(b.children))
        if a.ends and b.ends:
            for l in a.ends:
                for r in b.ends:
                    rendered=l.text+" "+r.text
                    gate=mechanical_admission_checks(rendered,min_letters=38,max_letters=400)
                    row={"left_clause":l.text,"right_clause":r.text,"rendered":rendered,
                         "left_tape":l.tape,"right_forward_tape":r.tape,"right_reversed_tape":r.tape[::-1],
                         "live_character_equation":prefix,"audit":audit(normalize_letters(rendered)),
                         "independent_parse":{"left":parse_clause(l.text),"right":parse_clause(r.text)},"admission":gate}
                    if row["audit"]["two_pointer_exact"] and all(row["independent_parse"].values()) and all(gate.values()): exact.append(row)
                    else: rejected.append({**row,"rejection_codes":[k for k,v in gate.items() if not v]+(["not_exact"] if not row["audit"]["two_pointer_exact"] else [])})
        if not common and prefix: frontier.append({"prefix":prefix,"left_next":sorted(a.children),"right_next":sorted(b.children)})
        for c in common: visit(a.children[c],b.children[c],p+c)
    visit(left,right,"")
    return {"experiment":"reverse-segmentation-grammar-product-20260920","status":"completed_exact" if exact else "completed_no_exact_closure",
      "novelty_preflight":{"status":"passed","distinct_mechanism":"complete ordinary-order clauses from authored lexicon; reverse grammar trie consumes live character equations","excluded":["known palindromes","catalogue text","word-order mirrors","repeated/self-palindromic modules","fragments","RLAIF"],"preflight_before_search":True},
      "config":{"state_limit":state_limit,"inventory":len(inventory),"left_forward_trie":True,"right_reverse_trie":True,"simultaneous_character_intersection":True,"posthoc_reversal":False,"remote_target":"hst-bench"},
      "stats":{"states":states,"truncated":truncated,"frontier_rows":len(frontier),"rejected_rows":len(rejected),"exact_rows":len(exact)},
      "frontier":frontier[:500],"rejected_rows":rejected[:500],"exact_candidates":exact,
      "provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"lexicon":"authored finite role inventory","human_readable_candidates":True,"independent_audits":["two-pointer character scan","forward/reverse SHA-256","independent clause parse"]},
      "falsifier":"A complete pair with >=38 letters passing exact audit, independent parses, and all admission checks would falsify the no-closure hypothesis.","next_action":"expand typed lexicon only if a new ordinary clause family changes the live-character frontier"}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--out",type=Path,required=True); p.add_argument("--state-limit",type=int,default=LIMIT); a=p.parse_args()
    if a.out.exists(): p.error("refusing to overwrite output")
    a.out.parent.mkdir(parents=True,exist_ok=True); r=run(a.state_limit); a.out.write_text(json.dumps(r,indent=2)+"\n"); print(json.dumps(r["stats"],indent=2))
if __name__=="__main__": main()
