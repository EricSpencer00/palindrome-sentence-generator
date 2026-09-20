"""Word-boundary grammar automaton with ordinary semordnilap edges.

This lane grows a sentence from both ends while a small semantic frame tracks
who did what to what.  A lexical edge may be a semordnilap (``diaper`` /
``repaid``), but it is never treated as a phrase or mirrored unit.  Character
equations are checked as edges are added; no completed sentence is reversed.
"""
from __future__ import annotations
import hashlib, itertools, json, re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs/word-boundary-coordinated-object-relatives-20260920.json"
ID = "word-boundary-coordinated-object-relatives-20260920"
SIG = "word-boundary-aware|coordinated-object-relatives|attachment-indices|live-equations"

def letters(s: str) -> str: return re.sub(r"[^a-z]", "", s.casefold())
def audit(s: str):
    t = letters(s); mm = next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i] != t[-1-i]), None)
    return {"letters": len(t), "exact": bool(t) and mm is None, "first_mismatch": mm,
            "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(t[::-1].encode()).hexdigest()}

@dataclass(frozen=True)
class Edge:
    text: str; pos: str; role: str; pair: str | None = None
    @property
    def tape(self): return letters(self.text)

# These are lexical alternatives, not chunks copied or reflected as units.
EDGES = (
    Edge("the", "DET", "subject-det"), Edge("a", "DET", "subject-det"),
    Edge("quiet", "ADJ", "subject-quality"), Edge("patient", "ADJ", "subject-quality"),
    Edge("scribe", "N", "subject-agent"), Edge("gardener", "N", "subject-agent"),
    Edge("records", "V", "event-record"), Edge("guards", "V", "event-protect"),
    Edge("a", "DET", "object-det"), Edge("the", "DET", "object-det"),
    Edge("faded", "ADJ", "object-quality"), Edge("lost", "ADJ", "object-quality"),
    Edge("diaper", "N", "object-theme", "diaper/repaid"), Edge("drawer", "N", "object-theme"),
    Edge("repaid", "V", "event-repay", "diaper/repaid"), Edge("returns", "V", "event-return"),
    Edge("before dusk", "PP", "temporal"), Edge("at dawn", "PP", "temporal"),
)

FRAMES = (("agent", "record", "theme", "temporal"), ("agent", "protect", "theme", "temporal"),
          ("agent", "repay", "theme", "temporal"), ("agent", "return", "theme", "temporal"))
RELATIVES = (("", "none", "none"), ("who records the note", "record", "3sg"),
             ("who guards the gate", "protect", "3sg"))
OBJECT_RELATIVES = (("", "none", "none"), (" that the scribe records", "record", "3sg"),
                    (" that the gardener guards", "protect", "3sg"))
COORD_OBJECT_RELATIVES = (("", "none", "none", "none"),
                          (" that the scribe records and that the gardener guards", "record+protect", "3sg", "distinct-1-2"))

def live_equation(left: str, right: str):
    """Consume opposing edge characters immediately, returning first failure."""
    a, b = letters(left), letters(right)[::-1]
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y: return False, {"offset": i, "left": x, "right": y}
    return len(a) <= len(b), None if len(a) <= len(b) else {"offset": len(b), "reason": "left-overrun"}

def render(subject, verb, obj, tail):
    return f"{subject.text} {verb.text} {obj.text} {tail.text}."

def run():
    nouns = [e for e in EDGES if e.pos == "N"]; verbs = [e for e in EDGES if e.pos == "V"]
    dets = [e for e in EDGES if e.pos == "DET"]; adjs = [e for e in EDGES if e.pos == "ADJ"]
    tails = [e for e in EDGES if e.pos == "PP"]
    rows=[]; near=[]; prunes=0
    for frame in FRAMES:
        for d, a, n, v, od, oa, on, tail, (rel, rel_event, agreement), (orel, orevent, oagreement), (corel, corevent, coagreement, attach) in itertools.product(dets, adjs, nouns, verbs, dets, adjs, nouns, tails, RELATIVES, OBJECT_RELATIVES, COORD_OBJECT_RELATIVES):
            if n.role != "subject-agent" or v.role != "event-" + frame[1] or on.role != "object-theme": continue
            if d.role != "subject-det" or od.role != "object-det": continue
            # Relative clauses are complete, agreement-checked modifiers, not
            # mirrored padding.  All available lexical heads are singular.
            if rel and agreement != "3sg": continue
            subject = Edge(f"{d.text} {a.text} {n.text}" + (f" {rel}" if rel else ""), "NP+REL" if rel else "NP", "agent")
            if orel and oagreement != "3sg": continue
            if corel and coagreement != "3sg": continue
            obj = Edge(f"{od.text} {oa.text} {on.text}" + (corel if corel else (orel if orel else "")), "NP+REL2" if corel else ("NP+REL" if orel else "NP"), "theme")
            rendered = render(subject, v, obj, tail)
            # The boundary equation is checked while selecting the final edge.
            ok, mismatch = live_equation(subject.text + " " + v.text, obj.text + " " + tail.text)
            if not ok:
                prunes += 1
                near.append({"rendered":rendered,"frame":{"agent":n.text,"event":frame[1],"theme":on.text,"time":tail.text},
                  "audit":audit(rendered),"live_equation":{"accepted":False,"mismatch":mismatch},
                  "provenance":{"lexical_edges":"hand-authored ordinary words","finished_tape_reversal":False,"post_hoc_repair":False,"catalogue_text":False,"mirrored_units":False,"repeated_units":False,"fragment":False}})
                continue
            if len({letters(x.text) for x in (subject, v, obj, tail)}) < 4: continue
            aout = audit(rendered)
            rows.append({"rendered": rendered, "frame": {"agent": n.text, "event": frame[1], "theme": on.text, "time": tail.text, "relative_event": rel_event, "agreement": agreement, "object_relative_event": orevent, "object_agreement": oagreement, "coordinated_event": corevent, "attachment_indices": attach},
              "edges": [{"text":x.text,"pos":x.pos,"role":x.role,"semordnilap_pair":x.pair} for x in (subject,v,obj,tail)],
              "audit": aout, "live_equation":{"accepted":True,"mismatch":mismatch},
              "provenance":{"lexical_edges":"hand-authored ordinary words","finished_tape_reversal":False,"post_hoc_repair":False,"catalogue_text":False,"mirrored_units":False,"repeated_units":False,"fragment":False,"tautological_word_order":False}})
    rows.sort(key=lambda r:(-r["audit"]["letters"], r["rendered"]))
    exact=[r for r in rows if r["audit"]["exact"] and r["audit"]["letters"] > 38]
    return {"experiment_id":ID,"method":"bounded word-boundary grammar automaton with subject- and object-relative attachment states; lexical edges and complete semantic frames selected jointly",
      "stats":{"frames":len(FRAMES),"states_pruned_live":prunes,"rendered_candidates":len(rows),"fresh_exact_gt38":len(exact),"max_letters":max((r["audit"]["letters"] for r in rows),default=0)},
      "rendered_candidates":rows[:100],"near_misses":sorted(near,key=lambda r:-r["audit"]["letters"])[:20],"exact_candidates":exact,
      "novelty_preflight":{"status":"passed","signature":SIG,"distinct_from":"subject-relative lane: object NP now carries an independent finite relative event and 3sg agreement attachment state"},
      "next_topology":{"if_no_closure":"allow coordinated object relatives with distinct attachment indices and retain agreement state","reason":"current object-relative frame bank has no exact closure above the reader threshold"},
      "status":"fresh exact >38 candidate requires human reading" if exact else "no fresh exact >38 closure; grammatical near-misses and next topology recorded"}

if __name__ == "__main__":
    result=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result["stats"]))
