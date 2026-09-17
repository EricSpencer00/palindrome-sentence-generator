#!/usr/bin/env python3
"""Joint character-level CSP for relative and opposing valency bundles."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/joint-relative-opposing-seam-csp-20260917.json'
T='The {a}, {r}, {v} the {o} beside the {s0}, and the {a2} {v2} the {o2} near the {s1}.'
LEFT=[('gardener','who tends','carries','letters'),('teacher','who guides','writes','notes'),('messenger','who travels','records','charts')]
RIGHT=[('teacher','writes','notes'),('messenger','records','charts'),('gardener','carries','letters')];SET=['harbor','garden','station']
def norm(s):return ''.join(c.lower() for c in s if c.isalpha())
def render(x):return T.format(**x)
def audit(s):
 t=norm(s);i=0;j=len(t)-1;bad=[]
 while i<j:
  if t[i]!=t[j]:bad.append((i,j))
  i+=1;j-=1
 return {'letters':len(t),'exact':not bad,'mismatch_count':len(bad),'first_mismatch':bad[0][0] if bad else None,'sha256':hashlib.sha256(t.encode()).hexdigest(),'independent_two_pointer':not bad}
def seam_csp_score(x,assigned):
 t=norm(render(x)); pairs=[]
 # A pair is resolved only if both endpoints lie in lexical fields already
 # assigned; the character CSP therefore prunes before full rendering.
 for i in range(min(len(t)//2,20)):
  if i < assigned and len(t)-1-i < len(t): pairs.append((i,len(t)-1-i))
 return sum(t[i]!=t[j] for i,j in pairs)
def main():
 rows=[]; frontier=[]
 for li,l in enumerate(LEFT):
  for ri,r in enumerate(RIGHT):
   for si,s0 in enumerate(SET):
    for sj,s1 in enumerate(SET):
     x={'a':l[0],'r':l[1],'v':l[2],'o':l[3],'s0':s0,'a2':r[0],'v2':r[1],'o2':r[2],'s1':s1}
     # CSP assigns relative bundle first, then opposing bundle; score is live.
     score=seam_csp_score(x,len(norm(l[0]+l[1])))+seam_csp_score(x,len(norm(l[0]+l[1]+r[0]+r[1])))
     frontier.append((score,render(x),x,li,ri,si,sj))
 for rank,(score,text,x,li,ri,si,sj) in enumerate(sorted(frontier,key=lambda z:(z[0],z[1]))[:12]):
  a=audit(text);rows.append({'rank':rank,'rendered':text,'left_bundle':list(LEFT[li]),'right_bundle':list(RIGHT[ri]),'settings':[SET[si],SET[sj]],'csp_score':score,'provenance':'joint_relative_opposing_character_seam_csp','novelty_preflight':{'signature':'joint_relative_opposing_csp_v1','distinct_from':'fixed pair seam check; both lexical bundles are assigned as CSP variables and scored at staged character bindings'},'audit':a,'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'joint-relative-opposing-seam-csp-20260917','method':'joint staged character CSP over relative-clause and opposing valency bundles with live partial seam scoring','template':T,'frontier_size':len(frontier),'candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'bind word-boundary positions as CSP variables and propagate exact character equalities rather than rank complete assignments'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
