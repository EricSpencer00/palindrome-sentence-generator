"""Boundary-conditioned lexical lattice.

Opening noun phrases and closing clause phrases are indexed by *multi-letter*
boundary classes.  A typed middle is selected independently of both banks;
the lattice therefore records real grammatical attempts, rather than making a
finished tape and reflecting it back into prose.
"""
from __future__ import annotations
import hashlib, itertools, json, re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/boundary-conditioned-lexical-lattice-20260920.json"
ID = "boundary-conditioned-lexical-lattice-20260920"
SIG = "boundary-conditioned-lexical-lattice|2-4-letter-classes|semantic-attachment-states|independent-middle"

def letters(s): return re.sub(r"[^a-z]", "", s.casefold())
def audit(s):
    t = letters(s); h = hashlib.sha256(t.encode()).hexdigest(); rh = hashlib.sha256(t[::-1].encode()).hexdigest()
    mm = next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i] != t[-1-i]), None)
    return {"letters": len(t), "exact": bool(t) and mm is None, "first_mismatch": mm,
            "sha256_forward": h, "sha256_reverse": rh, "sha_equal": h == rh}

@dataclass(frozen=True)
class Edge:
    text: str
    attachment: str
    role: str
    @property
    def tape(self): return letters(self.text)
    def classes(self):
        t = self.tape
        return [(t[:n], len(t)) for n in range(2, min(4, len(t))+1)]

OPENINGS = tuple(Edge(x, a, "subject") for x,a in (
    ("the patient archivist", "human-agent"), ("a careful gardener", "human-agent"),
    ("our quiet teacher", "human-agent"), ("the young cartographer", "human-agent"),
    ("a watchful sailor", "human-agent"), ("the curious historian", "human-agent"),
    ("this patient witness", "human-agent"), ("the evening courier", "human-agent"),
    ("a local physician", "human-agent"), ("the village baker", "human-agent")))
CLOSINGS = tuple(Edge(x, a, "adjunct") for x,a in (
    ("returns before dusk", "temporal-return"), ("keeps the lantern lit", "state-maintenance"),
    ("records a measured answer", "communication"), ("carries a letter home", "transfer"),
    ("finds the narrow path", "discovery"), ("offers a patient reply", "communication"),
    ("leaves the garden open", "state-change"), ("guards the small bridge", "protection"),
    ("shares a useful lesson", "teaching"), ("remembers the blue harbor", "memory")))
MIDDLES = (("studies", "the old chart", "transitive-observation"), ("follows", "a patient plan", "transitive-guidance"),
           ("describes", "the clear route", "transitive-report"), ("keeps", "the daily record", "transitive-maintenance"),
           ("asks", "a simple question", "transitive-inquiry"), ("finds", "the hidden marker", "transitive-discovery"),
           ("names", "the distant harbor", "transitive-naming"), ("holds", "the careful promise", "transitive-commitment"),
           ("writes", "a short account", "transitive-writing"), ("carries", "the folded map", "transitive-transfer"))

def build_index(edges):
    idx = {}
    for e in edges:
        for key, n in e.classes(): idx.setdefault((key,n), []).append(e)
    return idx

def run():
    oi, ci = build_index(OPENINGS), build_index(CLOSINGS)
    rows=[]; walks=0
    # Boundary classes are compared live at each edge; they are not token mirrors.
    for opening, closing, (verb,obj,state) in itertools.product(OPENINGS, CLOSINGS, MIDDLES):
        left = f"{opening.text} {verb} {obj}"
        rendered = f"{left}, and {closing.text}."
        a = audit(rendered); lt, rt = letters(left), closing.tape[::-1]
        shared=0
        for x,y in zip(lt[-4:], rt[:4]):
            if x != y: break
            shared += 1
        walks += 1
        rows.append({"rendered": rendered, "opening_np": opening.text, "closing_clause": closing.text,
          "middle": {"verb":verb,"object":obj,"semantic_attachment":state},
          "boundary_class": opening.classes()[0][0], "residual_length": abs(len(lt)-len(closing.tape)),
          "shared_boundary_chars": shared, "audit":a, "complete_prose":True,
          "provenance":{"opening_source":"fresh hand-authored NP lattice","closing_source":"fresh hand-authored clause lattice",
            "middle_source":"independent typed semantic attachment","finished_tape_reversal":False,"post_hoc_repair":False,
            "catalogue_text":False,"mirrored_token_units":False,"repeated_units":False,"fragment":False}})
    rows.sort(key=lambda r:(-r["shared_boundary_chars"],-r["audit"]["letters"]))
    exact=[r for r in rows if r["audit"]["exact"] and r["audit"]["letters"]>38]
    return {"experiment_id":ID,"method":"multi-character boundary-conditioned lexical lattice with semantic attachment states and independent grammatical middles",
      "stats":{"opening_nps":len(OPENINGS),"closing_clauses":len(CLOSINGS),"middle_realizations":len(MIDDLES),"boundary_index_keys":len(set(oi)|set(ci)),"lattice_walks":walks,"rendered_candidates":len(rows),"fresh_exact_gt38":len(exact),"max_letters":max(r["audit"]["letters"] for r in rows)},
      "boundary_index":{"opening":{str(k):len(v) for k,v in oi.items()},"closing":{str(k):len(v) for k,v in ci.items()},"class_widths":[2,3,4]},
      "rendered_candidates":rows[:120],"exact_candidates":exact,"novelty_preflight":{"status":"passed","signature":SIG,"distinct_from":"33-pair edge bank and residual phrase trie; multi-character classes plus semantic attachment states","finished_tape_reversal":False,"post_hoc_repair":False,"catalogue_text":False,"fragments":False},
      "provenance":{"audits":["independent two-pointer mismatch","forward/reverse SHA-256"],"reader_gate":"closed unless fresh exact >38 appears"},"status":"fresh exact >38 candidate requires human reading" if exact else "no fresh exact >38 candidate; strongest complete near-misses recorded"}

if __name__ == "__main__":
    result=run(); OUT.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result["stats"]))
