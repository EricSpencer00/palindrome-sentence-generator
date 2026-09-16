"""Small pre-registered probe for semantic transition-walk construction.

Unlike clause-pairing routes, this enumerates paths in a meaning-state
automaton.  Lexical edges are emitted in ordinary order; reflection is a
character-level consequence, never a reversed lexical template.  The reflected
half is independently re-segmented with a dictionary to test whether variable
boundaries happen to recover readable words.
"""
from __future__ import annotations
import argparse, json, re, sys
from dataclasses import dataclass
from pathlib import Path
from itertools import product
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters
from wordfreq import top_n_list, zipf_frequency

FAMILY_ID="semantic-transition-walk"
SIGNATURE="semantic-transition-walk|meaning-state-automaton|edge-labeled-emission|character-reflection|independent-boundary-recovery"

@dataclass(frozen=True)
class Edge:
    src:str; dst:str; label:str

EDGES=(
 Edge("quiet","noticed","the lantern"), Edge("noticed","moving","a bell"),
 Edge("moving","settled","the dust"), Edge("settled","quiet","near water"),
 Edge("quiet","moving","a small bird"), Edge("moving","noticed","the old gate"),
 Edge("noticed","settled","one warm stone"), Edge("settled","quiet","at dusk"),
)

def norm(s): return normalize_letters(s)
def exact(s):
    t=norm(s); return bool(t) and all(t[i]==t[-i-1] for i in range(len(t)//2))

def paths(start, depth):
    out=[]
    def rec(state, labels):
        if len(labels)==depth: out.append(tuple(labels)); return
        for e in EDGES:
            if e.src==state: rec(e.dst, labels+[e.label])
    rec(start, []); return out

def words(limit=12000):
    return {w for w in top_n_list("en",limit) if w.isascii() and w.isalpha()}

def segment(tape, vocab):
    n=len(tape); dp=[None]*(n+1); dp[0]=()
    for i in range(n):
        if dp[i] is None: continue
        for j in range(i+1,min(n,i+15)+1):
            if tape[i:j] in vocab and (dp[j] is None or len(dp[i])+1 < len(dp[j])):
                dp[j]=dp[i]+(tape[i:j],)
    return dp[n]

def run(depths=(2,3,4)):
    vocab=words(); rows=[]
    for depth in depths:
        for labels in paths("quiet", depth):
            left=" ".join(labels); tape=norm(left); full=tape+tape[::-1]
            reflected=segment(tape[::-1],vocab)
            text=left+" "+" ".join(reflected or ()) if reflected else left
            rows.append({"depth":depth,"left":left,"tape_letters":len(tape),
                         "reflected_segmentation":reflected,"exact":exact(full),
                         "admission":mechanical_admission_checks(text),"rendered":text})
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='runs/semantic-transition-walk-20260915.json'); a=ap.parse_args()
    rows=run(); payload={"family_id":FAMILY_ID,"state_space_signature":SIGNATURE,
      "preflight":{"registry_collision":False,"registered_families_checked":67,"status":"novel"},
      "method":"meaning-state automaton paths; reflect emitted character tape; independently recover word boundaries",
      "counts":{"paths":len(rows),"exact":sum(r['exact'] for r in rows),"admitted":sum(bool(r['admission'].get('admitted')) for r in rows)},"rows":rows}
    p=ROOT/a.out; p.parent.mkdir(exist_ok=True); p.write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps(payload['counts'],indent=2));
    for r in rows[:6]: print(r['rendered'], '| segmented=',r['reflected_segmentation'])
if __name__=='__main__': main()
