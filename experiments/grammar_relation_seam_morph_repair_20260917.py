"""Typed seam-local inflection repair; matching occurs during construction."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs'/'grammar-relation-seam-morph-repair-20260917.json'
ID='grammar-relation-seam-morph-repair-20260917'
FAMILIES={'DET':('the','a','one'),'SUBJ':('teacher','writer','artist','child'),'VERB3':('reads','writes','helps','guides','carries'),'OBJ':('letter','story','garden','message','river'),'ADV':('again','often','still','quietly'),'VERBBASE':('read','write','help','guide','carry')}
FUNCTION=frozenset('the a one again often still quietly'.split())
PATTERN=('DET','SUBJ','VERB3','DET','OBJ','ADV','VERBBASE','DET','OBJ')

def letters(s): return ''.join(re.findall('[a-z]',''.join(s).lower()))
def audit(s):
 t=letters(s); mm=[(i,len(t)-1-i) for i in range(len(t)//2) if t[i]!=t[-i-1]]
 return {'letters':len(t),'exact':bool(t) and not mm,'mismatches':mm[:8],'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def shortcut(ws):
 c=[w for w in ws if w not in FUNCTION]
 return {'repeated_content':len(c)!=len(set(c)),'self_palindromic_words':[w for w in ws if len(w)>1 and w==w[::-1]],'word_order_mirror':list(ws)==[w[::-1] for w in reversed(ws)]}

def solve(slots=PATTERN,budget=150000):
 stack=[(0,len(slots)-1,None,0,None,0,(None,)*len(slots),frozenset(),0)]
 seen=set(); exact=[]; rejected=[]; states=0; mismatches=0; best=None
 def consider(assign,reason,matched):
  if any(x is None for x in assign): return
  ws=tuple(assign); rendered=' '.join(ws); a=audit(rendered); sc=shortcut(ws)
  row={'rendered':rendered,'words':ws,'audit':a,'shortcuts':sc,'provenance':{'experiment':ID,'construction':'seam-local inflection family repair','reason':reason,'matched_character_pairs':matched,'pattern':slots}}
  if a['exact'] and not any(sc.values()): exact.append(row)
  elif a['exact']: rejected.append(row)
 while stack and states<budget:
  li,ri,lw,lp,rw,rp,assign,used,matched=stack.pop(); states+=1
  if lw is not None and lp==len(lw): stack.append((li+1,ri,None,0,rw,rp,assign,used,matched)); continue
  if rw is not None and rp==0: stack.append((li,ri-1,lw,lp,None,0,assign,used,matched)); continue
  key=(li,ri,lw,lp,rw,rp,assign,used)
  if key in seen: continue
  seen.add(key)
  if best is None or matched>best['matched_character_pairs']: best={'matched_character_pairs':matched,'assignment':assign,'pattern':slots,'rendered_partial':' '.join(x for x in assign if x is not None)}
  if li>ri: consider(assign,'complete relation closure',matched); continue
  if li==ri and lw is None and rw is None:
   for w in FAMILIES[slots[li]]:
    if len(w)==1: a=list(assign);a[li]=w;consider(tuple(a),'one-letter center',matched)
   continue
  if li==ri and lw is None and rw is not None:
   rem=rw[:rp]
   if rem and rem==rem[::-1]: a=list(assign);a[li]=rw;consider(tuple(a),'palindromic right residual',matched)
   continue
  if li==ri and rw is None and lw is not None:
   rem=lw[lp:]
   if rem and rem==rem[::-1]: a=list(assign);a[ri]=lw;consider(tuple(a),'palindromic left residual',matched)
   continue
  if lw is None:
   for w in FAMILIES[slots[li]]:
    if w not in used or w in FUNCTION: a=list(assign);a[li]=w;stack.append((li,ri,w,0,rw,rp,tuple(a),used|({w} if w not in FUNCTION else set()),matched))
   continue
  if rw is None:
   for w in FAMILIES[slots[ri]]:
    if w not in used or w in FUNCTION: a=list(assign);a[ri]=w;stack.append((li,ri,lw,lp,w,len(w),tuple(a),used|({w} if w not in FUNCTION else set()),matched))
   continue
  if lp<len(lw) and rp>0:
   if lw[lp]!=rw[rp-1]: mismatches+=1;continue
   stack.append((li,ri,lw,lp+1,rw,rp-1,assign,used,matched+1))
 return {'candidates':exact,'rejected_exact':rejected,'states':states,'mismatch_edges':mismatches,'budget_exhausted':states>=budget,'best_partial':best}

def main():
 r=solve();r['pattern_name']='morph_scene';r['next_repair']='add seam-conditioned subject/object agreement pairs while retaining relation state'
 r['near_miss_probes']=[{'rendered':'The teacher reads a letter often; the writer reads a story.', 'audit':audit('The teacher reads a letter often; the writer reads a story.'), 'provenance':'fresh authored grammatical control, not a generated palindrome'}]
 OUT.write_text(json.dumps({'experiment_id':ID,'status':'completed','independent_validator':'audit','novelty_preflight':{'catalogue_family_imported':False,'construction_signature':ID},'patterns':[r]},indent=2)+'\n')
 print(json.dumps({'states':r['states'],'mismatches':r['mismatch_edges'],'candidates':len(r['candidates']),'best':r['best_partial']},indent=2))
if __name__=='__main__': main()
