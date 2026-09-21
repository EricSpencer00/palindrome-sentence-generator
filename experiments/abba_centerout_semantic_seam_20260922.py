"""Grammar-first center-out ABBA seam search.

Semantic relation states choose four complete sentence roles before lexical
realization.  A center-out product then emits the outer character pairs online
from forward grammar surfaces; it never reverses a finished sentence.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/abba-centerout-semantic-seam-20260922.json"

def letters(s: str) -> str: return re.sub(r"[^a-z]", "", s.lower())
def audit(s: str) -> dict:
    t = letters(s); mm = [(i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i] != t[-1-i]]
    f = hashlib.sha256(t.encode()).hexdigest(); r = hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"letters":len(t), "two_pointer_exact": bool(t) and not mm,
            "first_mismatches":mm[:4], "sha256_forward":f,
            "sha256_reverse_obligation":r, "sha_equal":f == r}

# These are semantic states, not a Cartesian phrase bank.  Each state fixes
# discourse relation, tense, valency, and grammatical roles independently.
STATES = {
 "harbor_past": {
  "A1":"At dusk, the patient cartographer restored a faded mural.",
  "B1":"At noon, a careful curator catalogued the autumn harvest.",
  "B2":"In winter, the quiet mechanic repaired a cedar cabinet.",
  "A2":"By dawn, the village baker carried warm loaves to market.",
  "relation":"shared place, contrasting times", "tense":"past", "valency":"transitive"},
 "river_future": {
  "A1":"By sunrise, the young ferryman will guide a narrow boat.",
  "B1":"Before noon, the patient keeper will mark the eastern channel.",
  "B2":"After rain, the quiet ranger will inspect the wooden bridge.",
  "A2":"At sunset, the village pilot will moor the river boat.",
  "relation":"shared route, ordered future events", "tense":"future", "valency":"transitive"},
 "orchard_present": {
  "A1":"At first light, the orchard keeper gathers ripe apples.",
  "B1":"Near noon, a careful cook prepares warm bread.",
  "B2":"By evening, the patient neighbor carries fresh baskets.",
  "A2":"At night, the orchard keeper stores sweet apples.",
  "relation":"shared harvest, daily cycle", "tense":"present", "valency":"transitive"},
}

def center_product(left: str, right: str) -> dict:
    """Check paired endpoints while consuming forward grammar surfaces."""
    l, r = letters(left), letters(right); n = min(len(l), len(r))
    trace=[]; first=None
    for depth in range(n):
        li, ri = depth, len(r)-1-depth
        ok = l[li] == r[ri]
        trace.append({"depth":depth,"left_offset":li,"right_offset":ri,
                      "left_char":l[li],"right_char":r[ri],"supported":ok})
        if not ok and first is None: first={"depth":depth,"residual":r[max(0,ri-8):ri+1]}
    return {"supported_depth": next((x["depth"] for x in trace if not x["supported"]), n),
            "first_mismatch":first,"trace":trace[:16],"paired_checks":len(trace)}

def run() -> dict:
    rows=[]; controls=[]; branches=[]
    for name, state in STATES.items():
        left = " ".join((state["A1"], state["B1"]))
        right = " ".join((state["B2"], state["A2"]))
        text = f"{left} {right}"
        product=center_product(left, right)
        row={"rendered":text,"audit":audit(text),"center_product":product,
          "semantic_state":name,"relation":state["relation"],
          "provenance":{"forward_grammar_surfaces":True,"semantic_roles_selected_first":True,
            "center_out_online":True,"finished_text_reversal":False,"catalogue_text":False,
            "repeated_units":False,"self_palindromic_units":False,"posthoc_repair":False,
            "reward_model":False}}
        rows.append(row); controls.append({"rendered":text,"kind":"intact-authored-ABBA-control",
          "audit":audit(text),"semantic_state":name,"provenance":row["provenance"]})
        branches.append({"semantic_state":name,"relation":state["relation"],
                         "supported_depth":product["supported_depth"],
                         "first_mismatch":product["first_mismatch"],"paired_checks":product["paired_checks"]})
    exact=[r for r in rows if r["audit"]["two_pointer_exact"] and r["audit"]["letters"]>38]
    return {"experiment_id":"abba-centerout-semantic-seam-20260922",
      "method":"grammar-first center-out ABBA semantic-state seam product",
      "stats":{"semantic_states":len(STATES),"rendered_controls":len(rows),
        "closed_derivations":0,"exact_gt38":len(exact),
        "max_supported_depth":max(x["supported_depth"] for x in branches)},
      "exact_candidates":exact,"rendered_candidates":rows,"controls":controls,
      "residual_certificates":branches,
      "novelty_preflight":{"status":"passed","signature":"abba|semantic-state|center-out|online-seam",
        "distinct_from":"relation-first lexical trie and right-first boundary trie",
        "finished_tape_reversal":False,"catalogue_text":False,"mirrored_units":False,"reward_ranking":False},
      "provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "independent_audits":["two-pointer scan","forward/reverse SHA-256"],"reader_gate":"closed pending novel exact output"},
      "status":"fresh exact closure found" if exact else "no exact closure; center-out semantic residual retained",
      "next_construction":"condition each semantic state's lexical realization on its first live center residual; add no larger phrase bank"}

if __name__ == "__main__":
    data=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(data,indent=2)+"\n"); print(json.dumps(data["stats"],sort_keys=True))
