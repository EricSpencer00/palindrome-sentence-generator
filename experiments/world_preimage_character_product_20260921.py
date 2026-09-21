"""Live character intersection over executable scene states, not finished stories.

The reverse frontier traverses predecessor edges of the SAME forward scene
automaton. Thus it carries exact world-state preimages, not a reversed plan.
No complete surface is constructed until two character frontiers meet.
"""
from __future__ import annotations
import hashlib
import json
import re
from collections import defaultdict, deque
from pathlib import Path

ID = "world-preimage-character-product-20260921"
ACTORS = (("an aide", False), ("some men", True), ("Diana", False),
          ("the porter", False), ("a nurse", False))
# Bits: memos intact, gate open, parcel delivered. Preconditions are executable.
OPS = (("read", "reads", "nine memos", 1, 0, 0),
       ("rip", "rips", "nine memos", 1, 0, 1),
       ("save", "saves", "some memos", 1, 0, 0),
       ("open", "opens", "the gate", 0, 2, 0),
       ("close", "closes", "the gate", 2, 0, 2),
       ("deliver", "delivers", "a parcel", 2, 4, 0),
       ("inspire", "inspires", "Diana", 0, 0, 0))

def norm(s): return re.sub("[^a-z]", "", s.lower())

def audit(text):
    tape = norm(text)
    mismatch = next((i for i in range(len(tape)//2)
                     if tape[i] != tape[len(tape)-1-i]), None)
    words = re.findall("[a-z]+", text.lower())
    sentences = [norm(s) for s in text.split(".") if norm(s)]
    spans = []
    for i in range(len(words)):
        for j in range(i+2, len(words)+1):
            s = "".join(words[i:j])
            if len(s) < len(tape) and s == s[::-1]:
                spans.append(" ".join(words[i:j]))
    return {"letters": len(tape), "pointer_exact": bool(tape) and mismatch is None,
            "first_mismatch": mismatch,
            "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest(),
            "repeated_sentences": len(sentences) != len(set(sentences)),
            "repeated_words": sorted(w for w in set(words) if words.count(w)>1),
            "word_order_symmetry": words == words[::-1],
            "proper_self_palindromic_spans": spans,
            "self_palindromic_sentences": sum(s == s[::-1] for s in sentences)}

class Graph:
    def __init__(self, semantics=True, depth=3):
        self.nodes = {}; self.out = defaultdict(list); self.inc = defaultdict(list)
        self.edges = []; self.rejected = []; self.semantic_prunes = 0
        self.terminals = []; self.maxdepth = depth
        self.start = self.node(("subject", 0, 1))
        todo = deque([("subject", 0, 1)]); seen = set()
        while todo:
            state = todo.popleft()
            if state in seen: continue
            seen.add(state); src = self.node(state)
            phase, count, world, *rest = state
            if phase == "subject":
                if count >= 2: self.terminals.append(src)
                if count == depth: continue
                for actor, plural in ACTORS:
                    nxt = ("verb", count, world, plural)
                    self.emit(src, self.node(nxt), actor, actor+" ")
                    todo.append(nxt)
            elif phase == "verb":
                plural, = rest
                for index, (base, singular, obj, require, add, delete) in enumerate(OPS):
                    if semantics and world & require != require:
                        self.semantic_prunes += 1
                        if len(self.rejected)<8:
                            self.rejected.append({"partial_stage":"subject emitted; verb/object undecided",
                              "world":world,"rejected_action":base,"required_bits":require})
                        continue
                    nxt = ("object", count, world, index)
                    verb = base if plural else singular
                    self.emit(src, self.node(nxt), verb, verb+" ")
                    todo.append(nxt)
            else:
                index, = rest; base, singular, obj, require, add, delete = OPS[index]
                after = (world & ~delete) | add
                nxt = ("subject", count+1, after)
                self.emit(src, self.node(nxt), obj, obj+". ")
                todo.append(nxt)
        self.grammar_states = len(seen)

    def node(self, key):
        if key not in self.nodes: self.nodes[key] = len(self.nodes)
        return self.nodes[key]

    def emit(self, src, dst, letters, rendered):
        tape = norm(letters)
        for i, c in enumerate(tape):
            nxt = dst if i == len(tape)-1 else self.node(("char", len(self.edges)))
            e = (src, nxt, c, rendered if i == 0 else "")
            index = len(self.edges); self.edges.append(e)
            self.out[src].append(index); self.inc[nxt].append(index); src = nxt

    def render(self, ids):
        return "".join(self.edges[i][3] for i in ids).strip()

    def accepts(self, text):
        frontier = {self.start}
        for char in norm(text):
            frontier = {self.edges[e][1] for node in frontier for e in self.out[node]
                        if self.edges[e][2] == char}
        return bool(frontier.intersection(self.terminals))

    def solve(self, cap=200000):
        queue = deque((self.start, end, (), ()) for end in self.terminals)
        seen = set(); solutions = []; matched = 0; exhausted = True
        while queue:
            if len(seen)>=cap: exhausted=False; break
            left, right, lp, rp = queue.popleft()
            key = (left, right, len(lp))
            if key in seen: continue
            seen.add(key)
            if left == right:
                ids = lp+rp[::-1]
                solutions.append(self.render(ids))
            for li in self.out[left]:
                a,b,c,_ = self.edges[li]
                if b == right:
                    solutions.append(self.render(lp+(li,)+rp[::-1]))
                for ri in self.inc[right]:
                    x,y,d,_ = self.edges[ri]
                    if c == d:
                        matched += 1
                        queue.append((b,x,lp+(li,),rp+(ri,)))
        # One witness per merged state is enough for reachability, not enumeration.
        return {"status":"exhausted" if exhausted else "state_cap",
          "product_states":len(seen),"matched_character_transitions":matched,
          "pending_states":len(queue),"witnesses":sorted(set(solutions)),
          "enumeration":"one witness per merged character/world state; not all surfaces"}

def replay(actions):
    world=1; trace=[]
    for name in actions:
        op=next(x for x in OPS if x[0]==name)
        req,add,delete=op[3:]
        if world & req != req:
            return {"valid":False,"failed":name,"world":world,"required":req,"trace":trace}
        world=(world & ~delete)|add; trace.append({"action":name,"world_after":world})
    return {"valid":True,"trace":trace}

def run():
    modes={}; graphs={}
    for enabled in (True,False):
        g=Graph(enabled); graphs[enabled]=g; result=g.solve()
        result.update({"grammar_states":g.grammar_states,"character_nodes":len(g.nodes),
          "character_edges":len(g.edges),"semantic_prunes":g.semantic_prunes,
          "pre_render_rejection_witnesses":g.rejected,
          "rendered_outputs":[{"text":s,"audit":audit(s)} for s in result.pop("witnesses")]})
        modes["semantic" if enabled else "ablated"]=result
    controls=[("An aide opens the gate. The porter delivers a parcel.",["open","deliver"]),
      ("An aide rips nine memos. Some men read nine memos.",["rip","read"]),
      ("A nurse closes the gate. The porter delivers a parcel.",["close","deliver"])]
    exact=[r for r in modes["semantic"]["rendered_outputs"] if r["audit"]["letters"]>38]
    return {"experiment_id":ID,"method":"executable world-state predecessor relation intersected with shared character equality",
      "modes":modes,"fresh_exact_gt38":exact,
      "controls":[{"text":s,"world_replay":replay(p),"audit":audit(s),
        "semantic_graph_accepts":graphs[True].accepts(s),
        "ablated_graph_accepts":graphs[False].accepts(s)} for s,p in controls],
      "known_seed_calibration":{"text":"An aide rips nine memos; some men inspire Diana.",
        "audit":audit("An aide rips nine memos; some men inspire Diana."),"novel":False},
      "provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "vocabulary":"frozen authored actor/action domain; seed vocabulary deliberately overlaps for calibration",
        "catalogue_material":False,"completed_surface_pair_enumeration":False,
        "post_hoc_character_repair":False,"mirror_chunk_assembly":False,
        "reversal_usage":"path reconstruction and independent hash audit only",
        "semantic_limit":"resource consistency, not reader coherence; inspire has no causal precondition",
        "shortcut_limit":"all witnesses reported including ineligible repeats; none promoted without audits"},
      "falsifier":"Semantic ablation must restore read-after-rip and close-before-open branches before full surfaces exist; exact failure bounds this frozen domain only.",
      "reader_evidence":False,"north_star_met":False}

if __name__ == "__main__": print(json.dumps(run(),indent=2))
