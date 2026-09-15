"""Bounded internal-center repair over independently authored prose clauses.

Unlike seam/reverse methods, a complete clause pair is authored first; a small
window containing a lexical center is then rewritten from a finite semantic
inventory while all non-window characters remain fixed.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FAMILY_ID="internal-center-window-repair"
STATE_SPACE_SIGNATURE=("complete-authored-clause-pair|lexical-center-window|"
    "finite-semantic-rewrite-inventory|fixed-exterior-character-equations|"
    "center-inside-constituent|independent-two-pointer-audit")
LEFT=["A calm baker kneads dough", "A young pilot maps coast", "The kind nurse checks notes", "A quiet poet reads verse"]
RIGHT=["while a careful clerk files forms", "and a patient guide marks trails", "as a gentle teacher grades essays", "while a local farmer tends beans"]
REPAIRS={"a":"an", "an":"a", "calm":"quiet", "quiet":"calm", "young":"kind", "kind":"young", "careful":"patient", "patient":"careful", "local":"rural", "rural":"local"}
def tape(s): return ''.join(c.lower() for c in s if 'a'<=c.lower()<='z')
def pal(s):
    t=tape(s); return t==t[::-1]
def center_rewrites(s):
    words=s.split(); out=[]
    for i,w in enumerate(words):
        key=w.lower().strip('.,;')
        if key in REPAIRS:
            nw=REPAIRS[key]; out.append(('center-window',i,s,' '.join(words[:i]+[nw]+words[i+1:])))
    return out
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',default='runs/internal-center-window-20260915.json'); a=ap.parse_args()
    rows=[]; probes=[]; tested=0
    for l in LEFT:
      for r in RIGHT:
        base=l+"; "+r
        for op,i,src,cand in center_rewrites(base):
          tested+=1; t=tape(cand); exact=pal(cand)
          audit=(t==t[::-1] and all(t[j]==t[-j-1] for j in range(len(t)//2)))
          probes.append({'text':cand,'letters':len(t),'exact':exact,'independent_two_pointer':audit,'provenance':{'left':l,'right':r,'operator':op,'index':i,'source':src}})
          if exact: rows.append(probes[-1])
    artifact={'family_id':FAMILY_ID,'state_space_signature':STATE_SPACE_SIGNATURE,'method':'author complete clauses, then rewrite one lexical center window while exterior is fixed','tested_pairs':len(LEFT)*len(RIGHT),'tested_repairs':tested,'exact_candidates':rows,'rendered_probes':probes[:24],'admission':{'exact':len(rows),'admitted':0,'readability_certified':False,'reason':'no human reader evidence; no exact row survived'},'next_repair':'expand the center window to a two-word constituent and preserve semantic role agreement, then run held-out intact-prose/shuffled reader package','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    p=ROOT/a.output; p.parent.mkdir(exist_ok=True); p.write_text(json.dumps(artifact,indent=2)+'\n')
    print(json.dumps({'tested':tested,'exact':len(rows),'artifact':str(p)}))
if __name__=='__main__': main()
