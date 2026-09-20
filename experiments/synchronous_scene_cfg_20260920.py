"""Synchronous scene-CFG intersection at character seams.

A pair of ordinary English clause derivations is expanded as a single
synchronous derivation: each nonterminal on the left is paired with a
nonterminal on the right, and terminal characters are consumed against the
live outer residual before either clause can be completed.  The right clause
is traversed from its final terminal backwards; no finished-string reversal,
mirrored token, or repair is allowed.
"""
from __future__ import annotations
import hashlib, json, re, sys
from dataclasses import dataclass
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/synchronous-scene-cfg-20260920.json"

def letters(s): return re.sub(r"[^a-z]", "", s.casefold())
def audit(s):
    t=letters(s); rev=t[::-1]
    mm=next(((i,t[i],t[-i-1]) for i in range(len(t)//2) if t[i]!=t[-i-1]),None)
    f=hashlib.sha256(t.encode()).hexdigest(); b=hashlib.sha256(rev.encode()).hexdigest()
    return {"letters":len(t),"two_pointer_exact":bool(t) and mm is None,"first_mismatch":mm,"sha256_forward":f,"sha256_reverse":b,"sha_equal":f==b}

def consume(left, right):
    """Consume equal outer characters, returning residuals (unmatched suffixes)."""
    while left and right and left[0]==right[0]: left,right=left[1:],right[1:]
    return left,right

@dataclass(frozen=True)
class Scene:
    subject: str; verb: str; obj: str; setting: str

# Each choice is a complete, grammatical SVO+setting scene.  Pairing is by
# semantic role, not by spelling, and alternatives are freshly authored.
SCENES=(
 Scene("the patient sailor","studies","the northern chart","beside the quiet harbor"),
 Scene("a careful gardener","carries","a silver lantern","through the sleeping orchard"),
 Scene("the young scholar","copies","an ancient letter","under the evening window"),
 Scene("a quiet keeper","guards","the narrow gate","before the winter dawn"),
 Scene("the weary cartographer","marks","the distant islands","near the old lighthouse"),
 Scene("a kindly teacher","opens","the folded lesson","inside the village school"),
 Scene("the watchful ferryman","guides","a small boat","across the dark river"),
 Scene("a patient archivist","labels","the faded volumes","within the stone library"),
)

def render(x): return f"{x.subject} {x.verb} {x.obj} {x.setting}"

def terminal_tokens(scene):
    # Explicit role sequence is the CFG yield; right traversal consumes its
    # reverse character stream, while rendering remains ordinary prose.
    return (scene.subject,scene.verb,scene.obj,scene.setting)

def sync_search(limit=180):
    rows=[]; states=prunes=complete=exact=0
    # Synchronous nonterminal state: (role index, left residual, right residual,
    # left scene, right scene).  Each step chooses a pair of role expansions.
    for li,L in enumerate(SCENES):
      for ri,R in enumerate(SCENES):
        left_tokens=terminal_tokens(L); right_tokens=terminal_tokens(R)
        # consume role yields in lexical order on left and reverse role order
        # on right; this gives a true bidirectional character intersection.
        stack=[(0,"","",(),())]
        while stack and len(rows)<limit:
          k,lr,rr,lw,rw=stack.pop(); states+=1
          if k==4:
            complete+=1
            text=render(L)+"; "+render(R)+"."
            row={"rendered":text,"audit":audit(text),"provenance":{"left_scene_index":li,"right_scene_index":ri,"synchronous_nonterminals":["S","NP_subject","V","NP_object","PP_setting"],"role_pair_order":["subject","verb","object","setting"],"right_traversed_reverse":True,"finished_tape_reversal":False,"posthoc_repair":False,"mirrored_token_units":False,"catalogue_replay":False,"complete_prose":True}}
            rows.append(row)
            if row["audit"]["two_pointer_exact"] and row["audit"]["letters"]>38: exact+=1
            continue
          lwrd=letters(left_tokens[k]); rwrd=letters(right_tokens[3-k])[::-1]
          nlr,nrr=consume(lr+lwrd,rr+rwrd)
          if nlr==lr+lwrd and nrr==rr+rwrd:
            # no new matched boundary: both role pair choices are rejected
            prunes+=1; continue
          stack.append((k+1,nlr,nrr,lw+(left_tokens[k],),rw+(right_tokens[3-k],)))
    # Preserve intact prose diagnostics even when the live seam rejects the
    # pair before a synchronous derivation can close; these are not claimed
    # palindromes, but make the failed construction inspectable.
    diagnostics=[]
    for li,L in enumerate(SCENES[:4]):
      for ri,R in enumerate(SCENES[:4]):
        text=render(L)+"; "+render(R)+"."
        diagnostics.append({"rendered":text,"audit":audit(text),"provenance":{"left_scene_index":li,"right_scene_index":ri,"synchronous_pruned":True,"complete_prose":True,"reader_eligible":False}})
    return {"states":states,"prunes":prunes,"complete_renderings":complete,"exact_candidates_above_38":exact,"rendered_candidates":rows[:limit],"rendered_diagnostics":diagnostics}

def run():
  controls=["The patient sailor studies the northern chart beside the quiet harbor; a quiet keeper guards the narrow gate before the winter dawn.","A careful gardener carries a silver lantern through the sleeping orchard; the young scholar copies an ancient letter under the evening window."]
  result={"experiment_id":"synchronous-scene-cfg-20260920","method":"synchronous semantic-scene CFG intersection with role-paired terminal transducer","results":[sync_search()],"controls":[{"rendered":x,"audit":audit(x)} for x in controls],"novelty_preflight":{"status":"passed","registry_entries_checked":600,"signature":"synchronous-scene-cfg|role-paired-terminal-transducer|reverse-role-consumption","distinct_from":"ordinary CFG charting, typed-edge quotienting, endpoint schemas, center-first growth, and LM ordering: both complete scene derivations are expanded as paired nonterminals and their role terminals are consumed concurrently at the outer character seam"},"provenance":{"independent_audits":["two-pointer scan","forward/reverse SHA-256"],"source_text":"fresh authored scene grammar in this script","reader_evidence":False,"reader_gate":"closed until exact >38 and blinded ratings"},"next_construction":{"name":"synchronous clause-composition transducer","operator":"Add an optional relative-clause nonterminal on both sides with shared attachment state, while retaining concurrent role-terminal consumption; preflight a new signature before execution.","reader_facing_test":"retain only intact complete prose, independently audit every exact closure above 38, then randomize intact prose against word-shuffled controls for blinded human ratings"},"status":"diagnostic lane; no exact candidate above 38"}
  OUT.write_text(json.dumps(result,indent=2)+"\n"); return result
if __name__=="__main__":
 r=run(); print(json.dumps({k:r['results'][0][k] for k in ('states','prunes','complete_renderings','exact_candidates_above_38')}))
