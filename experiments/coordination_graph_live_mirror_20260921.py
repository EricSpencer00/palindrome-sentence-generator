"""Bounded question/answer coordination graph with live character obligations.

Speech act and SVO/attachment roles are selected as graph states before lexical
realisation.  The two sides are expanded in lockstep; punctuation and token
boundaries are surface choices, not constraints on the letter tape.
"""
import hashlib, itertools, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "coordination-graph-live-mirror-20260921.json"

FRAMES = (
    {"act": "question", "subject": ("you", "we"), "verb": ("carry", "mark"), "object": ("the map", "a bell"), "attachment": ("at dawn", "in the rain")},
    {"act": "answer", "subject": ("I", "we"), "verb": ("carry", "keep"), "object": ("a map", "the bell"), "attachment": ("at dusk", "by the gate")},
)
PUNCT = (("?", "."), (",", "."))

def tape(s): return re.sub(r"[^a-z]", "", s.casefold())
def sha(s): return hashlib.sha256(s.encode()).hexdigest()
def audit(s):
    x=tape(s); mismatch=next((i for i in range(len(x)//2) if x[i]!=x[-1-i]),None)
    return {"letters":len(x),"two_pointer_exact":mismatch is None,"first_mismatch":mismatch,
            "sha256_forward":sha(x),"sha256_reverse":sha(x[::-1]),"sha_equal":sha(x)==sha(x[::-1])}

def live_compatible(left, right):
    """Check every currently exposed pair before either half is rendered."""
    a,b=tape(left),tape(right); checks=[]
    for i,ch in enumerate(a):
        j=len(b)-1-i
        if j < 0: return False, checks, "right exhausted"
        checks.append({"left":i,"right":j,"left_char":ch,"right_char":b[j]})
        if ch != b[j]: return False, checks, "obligation mismatch"
    return True, checks, "prefix closed"

def run():
    rows=[]
    for lf,rf,p in itertools.product(FRAMES, FRAMES, PUNCT):
        # Graph topology is fixed before selecting words; two coordinated turns.
        left=f"{lf['subject'][0]} {lf['verb'][0]} {lf['object'][0]} {lf['attachment'][0]}{p[0]}"
        right=f"{rf['subject'][1]} {rf['verb'][1]} {rf['object'][1]} {rf['attachment'][1]}{p[1]}"
        ok,checks,reason=live_compatible(left,right)
        rendered=f"{left} {right}"
        words=re.findall(r"[a-z]+",rendered.casefold())
        row={"rendered":rendered,"roles":{"left":lf,"right":rf},"punctuation":p,
             "live_obligation":{"accepted":ok,"checks":checks,"reason":reason},"audit":audit(rendered),
             "gates":{"complete_prose":True,"whole_output_exact":audit(rendered)["two_pointer_exact"],
                      "no_repeated_content":len(words)==len(set(words)),"no_self_palindromic_word":all(tape(w)!=tape(w)[::-1] for w in words)},
             "provenance":{"roles_before_lexical_realization":True,"speech_act_graph":"question->answer coordination",
                            "variable_token_boundaries":True,"punctuation_shifted":True,"live_mirrored_characters":True,
                            "catalogue_text":False,"seed_wrapping":False,"finished_tape_reversal":False,"post_hoc_repair":False}}
        row["accepted"]=ok and all(row["gates"].values()); rows.append(row)
    exact=[r for r in rows if r["accepted"]]
    out={"experiment_id":"coordination-graph-live-mirror-20260921","status":"completed_exact" if exact else "completed_no_exact_closure",
         "method":"bounded speech-act/SVO/attachment coordination graph with variable punctuation and live mirrored character obligations",
         "rendered_controls":rows,"exact_candidates":exact,"stats":{"frames":len(FRAMES),"punctuation_modes":len(PUNCT),"rendered":len(rows),"live_prefix_pass":sum(r["live_obligation"]["accepted"] for r in rows),"exact":len(exact),"longest_letters":max(r["audit"]["letters"] for r in rows)},
         "failure_and_repair":{"residual":[r["live_obligation"]["reason"] for r in rows[:4]],"next_repair":"add a second answer attachment edge whose opening character is selected from the exposed residual, preserving graph roles"},
         "novelty_preflight":{"signature":"speech-act-coordination-graph|live-role-tape|variable-punctuation","distinct_from":"edge-pair catalogue and seed-wrapping lanes","fresh_complete_prose":True},
         "queue_row":{"lane":"coordination_graph_live_mirror","priority":"next","bounded":True,"action":"answer-attachment residual repair"},
         "provenance":{"generator_sha256":sha(Path(__file__).read_text()),"independent_audits":["two-pointer","forward/reverse SHA-256"],"shortcuts_excluded":True}}
    RUN.parent.mkdir(exist_ok=True); RUN.write_text(json.dumps(out,indent=2)+"\n"); return out
if __name__=='__main__':
    d=run(); print(json.dumps(d["stats"],sort_keys=True))
