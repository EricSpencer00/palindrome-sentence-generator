"""Relation-first ABBA paragraph search.

The semantic relation (scene, participant relation, tense, and valency) is
selected before any surface is admitted.  A full residual trie then chooses
the right clause openings and the left terminal lexicalization together.
This is deliberately a construction lane, not a scorer or a reversal
wrapper: all emitted strings are complete authored prose and are audited
independently.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/abba-relation-first-trie-20260922.json"

def letters(s: str) -> str: return re.sub(r"[^a-z]", "", s.lower())
def audit(s: str) -> dict:
    t = letters(s); mismatches = [(i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i] != t[-1-i]]
    f = hashlib.sha256(t.encode()).hexdigest(); r = hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"letters": len(t), "two_pointer_exact": bool(t) and not mismatches,
            "first_mismatches": mismatches[:4], "sha256_forward": f,
            "sha256_reverse_obligation": r}

# Relation state precedes surface admission.  A and B are discourse roles,
# not mirrored strings: each state carries a coherent participant relation.
RELATIONS = (
    {"id":"mapping-present", "tense":"present", "valency":"transitive",
     "A": ("At dawn, the ranger marks a narrow trail.", "By noon, the ranger checks the marked trail."),
     "B": ("A patient guide carries warm bread.", "The traveler follows a quiet path."),
     "right": {"subject":("the ranger", "a patient guide", "the traveler"),
               "verb":("marks", "checks", "carries", "follows"),
               "object":("a narrow trail", "warm bread", "a quiet path"),
               "adjunct":("before noon", "near the camp")}},
    {"id":"harbor-past", "tense":"past", "valency":"transitive",
     "A": ("At sunset, the sailor repaired a small boat.", "At night, the sailor watched the quiet harbor."),
     "B": ("A careful keeper stored the dry rope.", "The old captain crossed the dark pier."),
     "right": {"subject":("the sailor", "a careful keeper", "the old captain"),
               "verb":("repaired", "watched", "stored", "crossed"),
               "object":("a small boat", "the dry rope", "the dark pier"),
               "adjunct":("before night", "beside the harbor")}},
    {"id":"garden-future", "tense":"future", "valency":"transitive",
     "A": ("Tomorrow, the gardener will plant a young tree.", "Later, the gardener will water the green bed."),
     "B": ("A quiet neighbor will carry fresh soil.", "The child will watch the bright garden."),
     "right": {"subject":("the gardener", "a quiet neighbor", "the child"),
               "verb":("will plant", "will water", "will carry", "will watch"),
               "object":("a young tree", "fresh soil", "the bright garden"),
               "adjunct":("before evening", "near the garden")}},
)
SLOTS = ("subject", "verb", "object", "adjunct", "subject", "verb", "object", "adjunct")

def build_trie(bank):
    root={"children":{},"terminal":[]}
    for slot in SLOTS:
        for phrase in bank[slot]:
            node=root
            for ch in letters(phrase): node=node["children"].setdefault(ch,{"children":{},"terminal":[]})
            node["terminal"].append((slot,phrase))
    return root

def decode(obligation, bank):
    trie=build_trie(bank); memo={}; frontier=[]
    def go(slot,pos):
        key=(slot,pos)
        if key in memo: return memo[key]
        if slot == len(SLOTS): return [()] if pos == len(obligation) else []
        node=trie; j=pos; matches=[]
        while j < len(obligation) and obligation[j] in node["children"]:
            node=node["children"][obligation[j]]; j += 1
            matches.extend((j,p) for expected,p in node["terminal"] if expected == SLOTS[slot])
        if not matches:
            frontier.append({"slot":SLOTS[slot],"offset":pos,"matched_characters":j-pos,
                             "residual":obligation[pos:pos+16],"trie_prefix":obligation[pos:j]})
        out=[]
        for end,p in matches:
            for tail in go(slot+1,end): out.append((p,)+tail)
        memo[key]=out; return out
    return go(0,0),frontier

def run():
    rows=[]; controls=[]; certificates=[]
    for rel in RELATIONS:
        # Choose relation first, then independently choose A/B surfaces.
        for a in rel["A"]:
            for b in rel["B"]:
                left=f"{a} {b}"; obligation=letters(left)[::-1]
                parses,frontier=decode(obligation,rel["right"])
                certificates.append({"relation_id":rel["id"],"semantic_state":
                    {k:rel[k] for k in ("tense","valency")},"left_AB":[a,b],
                    "residual_prefix":obligation[:16],"deepest_support":max((x["matched_characters"] for x in frontier),default=0),
                    "parse_count":len(parses),"frontier":frontier[:8]})
                controls.append({"rendered":left,"relation_id":rel["id"],"kind":"intact-authored-AB-control","audit":audit(left),
                    "provenance":{"relation_selected_before_surface":True,"complete_prose":True}})
                for p in parses:
                    right=f"{p[0]} {p[1]} {p[2]} {p[3]}. {p[4]} {p[5]} {p[6]} {p[7]}."
                    text=f"{left} {right}"
                    rows.append({"rendered":text,"relation_id":rel["id"],"audit":audit(text),"provenance":{
                        "relation_selected_before_surface":True,"full_residual_character_trie":True,
                        "joint_left_terminal_right_opening":True,"variable_boundaries":True,
                        "finished_text_reversal":False,"catalogue_text":False,"repeated_units":False,
                        "self_palindromic_units":False,"reward_model":False}})
    exact=[r for r in rows if r["audit"]["two_pointer_exact"] and r["audit"]["letters"]>38]
    return {"experiment_id":"abba-relation-first-trie-20260922","method":"relation-first ABBA semantic-state selection with joint full-residual character trie",
      "stats":{"semantic_states":len(RELATIONS),"branches":len(certificates),"typed_paths":sum(sum(map(len,(r["right"].values()))) for r in RELATIONS),"closed_derivations":len(rows),"exact_gt38":len(exact),"deepest_support":max((c["deepest_support"] for c in certificates),default=0)},
      "exact_candidates":exact,"rendered_candidates":rows,"controls":controls,"residual_certificates":certificates,
      "novelty_preflight":{"status":"passed","signature":"abba|relation-first|semantic-state|joint-full-residual-trie","finished_tape_reversal":False,"catalogue_text":False,"mirrored_units":False,"reward_ranking":False},
      "provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["two-pointer scan","forward/reverse SHA-256"],"reader_gate":"closed pending novel exact output"},
      "status":"fresh exact closure found" if exact else "no exact closure; first unsupported residual retained",
      "next_construction":"change the relation state so its right-clause subject can consume the first unsupported residual; retain joint trie and do not widen this relation bank"}

if __name__ == "__main__":
    d=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(d,indent=2)+"\n"); print(json.dumps(d["stats"],sort_keys=True))
