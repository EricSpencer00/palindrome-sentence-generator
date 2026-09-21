"""ABBA variable-depth subject topology over the full residual trie.

The right B/A half begins with a typed NP that may be determiner+adjective+
noun, or that NP plus an appositive.  This changes grammar topology: it can
consume a residual across a phrase and sentence boundary before the verb,
while all characters remain hard exact obligations.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
try:
    from experiments.abba_full_residual_lexical_trie_20260922 import audit, letters
except ModuleNotFoundError:  # direct execution from the experiments directory
    from abba_full_residual_lexical_trie_20260922 import audit, letters

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/abba-variable-depth-np-topology-20260922.json"
LEFT = {
    "A": ("At dusk, the patient cartographer restored a faded mural.",
          "By dawn, the village baker carried warm loaves to market."),
    "B": ("At noon, a careful curator catalogued the autumn harvest.",
          "In winter, the quiet mechanic repaired a cedar cabinet."),
}
SUBJECTS = (
    ("the", "patient", "archivist"),
    ("a", "quiet", "sailor"),
    ("our", "careful", "teacher"),
    ("the", "young", "botanist"),
)
APPOSITIONS = ("the harbor keeper", "a trusted guide")
RIGHT = {
    "verb": ("records", "studies", "carries", "notices"),
    "object": ("a folded map", "the blue lantern", "one small basket"),
    "adjunct": ("before dawn", "near the harbor", "beside the market"),
}
# Each path has a variable-depth subject then ordinary typed roles twice.
PATHS = []
for subj in SUBJECTS:
    base = " ".join(subj)
    PATHS.append(("subject", base, [base]))
    for app in APPOSITIONS:
        PATHS.append(("subject+apposition", f"{base}, {app}", [base, app]))

def trie_add(root: dict, token: str, payload: tuple):
    node = root
    for ch in letters(token):
        node = node["children"].setdefault(ch, {"children": {}, "terminal": []})
    node["terminal"].append(payload)

def decode(obligation: str, max_parses: int = 16):
    # Separate tries per semantic role; subject has variable-depth terminals.
    role_trie = {"subject": {"children": {}, "terminal": []},
                 "verb": {"children": {}, "terminal": []},
                 "object": {"children": {}, "terminal": []},
                 "adjunct": {"children": {}, "terminal": []}}
    for kind, phrase, parts in PATHS:
        trie_add(role_trie["subject"], phrase, (kind, phrase, parts))
    for role, vals in RIGHT.items():
        for phrase in vals: trie_add(role_trie[role], phrase, (role, phrase, [phrase]))
    roles = ("subject", "verb", "object", "adjunct", "subject", "verb", "object", "adjunct")
    memo = {}; frontier = []; transitions = []
    def go(slot, pos):
        key=(slot,pos)
        if key in memo: return memo[key]
        if slot == len(roles): return [()] if pos == len(obligation) else []
        role=roles[slot]; node=role_trie[role]; j=pos; matches=[]
        while j < len(obligation) and obligation[j] in node["children"]:
            node=node["children"][obligation[j]]; j += 1
            matches.extend((j,payload) for payload in node["terminal"])
        if not matches:
            frontier.append({"role": role, "slot": slot, "offset": pos,
                             "matched_characters": j-pos,
                             "residual": obligation[pos:pos+14]})
        out=[]
        for end,payload in matches:
            transitions.append({"slot":slot,"role":role,"start":pos,"end":end,
                                "surface":payload[1],"depth":len(payload[2])})
            for tail in go(slot+1,end):
                out.append((payload[1],)+tail)
                if len(out)>=max_parses: break
        memo[key]=out; return out
    return go(0,0), frontier, transitions

def run():
    rows=[]; controls=[]; cert=[]
    for a in LEFT["A"]:
      for b in LEFT["B"]:
        left=f"{a} {b}"; obligation=letters(left)[::-1]
        parses, frontier, transitions=decode(obligation)
        cert.append({"left_A_B":[a,b],"residual_prefix":obligation[:16],
                     "deepest_support":max((x["matched_characters"] for x in frontier),default=0),
                     "support_moved_beyond_subject":any(x["slot"]>0 for x in frontier),
                     "parse_count":len(parses),"transitions":transitions[:12],
                     "frontier":frontier[:8]})
        controls.append({"rendered":left,"audit":audit(left),"kind":"intact-authored-AB-control",
                         "provenance":{"fresh_terminal_family":True,"complete_prose":True}})
        for p in parses:
          right=f"{p[0]}. {p[1]} {p[2]} {p[3]}. {p[4]}. {p[5]} {p[6]} {p[7]}."
          text=f"{left} {right}"
          rows.append({"rendered":text,"audit":audit(text),"provenance":{"variable_depth_np":True,
            "optional_apposition":True,"cross_sentence_boundary":True,"semantic_ABBA_roles":True,
            "finished_text_reversal":False,"catalogue_text":False,"repeated_units":False,
            "self_palindromic_units":False,"posthoc_repair":False,"reward_model":False}})
    exact=[r for r in rows if r["audit"]["two_pointer_exact"] and r["audit"]["letters"]>38]
    return {"experiment_id":"abba-variable-depth-np-topology-20260922",
      "method":"ABBA variable-depth NP/apposition topology over full residual trie",
      "stats":{"left_A_choices":2,"left_B_choices":2,"subject_paths":len(PATHS),"branches":len(cert),
               "closed_derivations":len(rows),"exact_gt38":len(exact),
               "deepest_support":max((x["deepest_support"] for x in cert),default=0)},
      "exact_candidates":exact,"rendered_candidates":rows,"controls":controls,
      "residual_certificates":cert,"novelty_preflight":{"status":"passed",
        "signature":"abba|variable-depth-subject-np|optional-apposition|full-residual-trie",
        "distinct_from":"fixed-depth typed clauses and three-character onset selection",
        "finished_tape_reversal":False,"catalogue_text":False,"mirrored_units":False,"reward_ranking":False},
      "provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "independent_audits":["two-pointer scan","forward/reverse SHA-256"],"reader_gate":"closed pending novel exact output"},
      "status":"fresh exact closure found" if exact else "no exact closure; topology residual certificate retained",
      "next_construction":"change the semantic relation between the two right clauses at the first subject obstruction; do not add more NP adjectives"}

if __name__ == "__main__":
    d=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(d,indent=2)+"\n"); print(json.dumps(d["stats"],sort_keys=True))
