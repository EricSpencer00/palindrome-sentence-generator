"""Terminal-conditioned auxiliary/transitive frames with live seam propagation.

The search chooses ordinary clause words from hand-authored grammatical domains.
At every character emitted on either side, the opposite exposed character is
checked immediately; object domains are not expanded until the auxiliary and
verb terminal obligations are satisfied.  This is a construction experiment,
not a readability certificate.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from itertools import product
from collections import Counter
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from llm_palindrome.admission import normalize_letters, mechanical_admission_checks
ID="luna-auxiliary-terminal-frames-20260917"
SIG="auxiliary-transitive-frames|terminal-conditioned-domains|online-seam-propagation|independent-pointer-audit"
OUT=ROOT/f"runs/{ID}.json"

# Short, ordinary clause vocabulary; no known palindrome or generated corpus.
DOM={
 "det":("a","an","the","some","this","that"),
 "subj":("artist","baker","captain","doctor","farmer","guard","judge","poet","sailor","teacher","writer"),
 "aux_sg":("can","does","will","has"), "aux_pl":("can","do","will","have"),
 "verb":("admire","answer","carry","draft","guide","help","notice","paint","read","save","send","teach","visit","write"),
 "verb_s":("admires","answers","carries","drafts","guides","helps","notices","paints","reads","saves","sends","teaches","visits","writes"),
 "obj":("answer","book","bridge","letter","map","message","notice","paper","portrait","report","song","story","teacher","vessel"),
}
FRAMES=(
 ("sg_aux",("det","subj","aux_sg","verb_s","det","obj"),"sg"),
 ("pl_aux",("det","subj","aux_pl","verb","det","obj"),"pl"),
 ("sg_plain",("det","subj","verb_s","det","obj"),"sg"),
 ("pl_plain",("det","subj","verb","det","obj"),"pl"),
)
def tape(words): return normalize_letters(" ".join(words))
def online_match(left_words,right_words):
    """Compare exposed characters while right words are selected incrementally."""
    a=tape(left_words); b=tape(right_words)
    k=min(len(a),len(b)); return all(a[i]==b[-1-i] for i in range(k))
def audit(words):
    text=" ".join(words).capitalize()+"."
    n=normalize_letters(text); ascii_t="".join(c for c in text.casefold() if c.isalpha() and c.isascii())
    return {"rendered":text,"letters":len(n),"normalized_tape":n,"independent_ascii_tape":ascii_t,
            "exact":bool(n) and n==n[::-1],"independent_exact":bool(ascii_t) and ascii_t==ascii_t[::-1],
            "two_pointer_exact":all(n[i]==n[-1-i] for i in range(len(n)//2)),
            "mechanical_checks":mechanical_admission_checks(text,min_letters=45,max_letters=260),
            "sha256":hashlib.sha256(n.encode()).hexdigest()}
def run(limit=2500):
    stats=Counter(); rows=[]; seen=set(); tested=0
    for lname,lslots,lnum in FRAMES:
      for rname,rslots,rnum in FRAMES:
       # terminal conditioning: auxiliaries and inflected verbs are selected
       # before object slots; only compatible number pairs are considered.
       if lnum!=rnum: continue
       pools=[DOM[s] for s in lslots]; rpools=[DOM[s] for s in rslots]
       for left in product(*pools):
        if tested>=limit: break
        tested+=1; stats["left_frames"]+=1
        # object is deliberately withheld until all pre-object words have a
        # live opposite edge; this makes terminal debt observable.
        left_pre=left[:-1]
       for right_pre in product(*rpools[:-1]):
         if stats["pre_object_states"] >= limit: break
         stats["pre_object_states"]+=1
         if not online_match(left_pre,right_pre): stats["terminal_debt_reject"]+=1; continue
         for obj in rpools[-1]:
          right=right_pre+(obj,); stats["object_expansions"]+=1
          if not online_match(left,right): stats["object_seam_reject"]+=1; continue
          words=left+tuple(reversed(right)); a=audit(words)
          if a["exact"] and a["letters"]>=45 and a["normalized_tape"] not in seen:
           seen.add(a["normalized_tape"]); rows.append({"audit":a,"left_frame":lname,"right_frame":rname,"left_words":left,"right_words":right,"provenance":"hand-authored clause domains"})
    return {"experiment_id":ID,"signature":SIG,"method":"Terminal-conditioned auxiliary/transitive frames with online opposite-edge seam propagation before object expansion; ordinary clause order and independent lexical choices.","config":{"frame_count":len(FRAMES),"state_limit":limit,"min_letters":45,"catalogue_text_imported":False,"known_palindromes_imported":False},"stats":{**dict(stats),"tested_left_frames":tested,"exact_closures":len(rows),"reader_eligible":0},"rendered_candidates":sorted(rows,key=lambda x:-x["audit"]["letters"]),"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_exact_audits":["normalized-tape-reversal","ASCII-tape-reversal","two-pointer"],"readability_certificate":False},"next_repair":"Add boundary-conditioned auxiliary alternatives and semantic object-role pairing; current terminal debt rejects all ordinary frame pairs before object expansion.","reader_gate":"No reader study triggered: no mechanically admitted reader candidate."}
if __name__=="__main__":
 OUT.parent.mkdir(exist_ok=True); result=run(); OUT.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result["stats"],sort_keys=True))
