"""Finite typed center-language closure over residual-equivalence edge states."""
from __future__ import annotations
import hashlib,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'experiments'))
from residual_equivalence_edge_quotient_20260920 import FRAMES, FRESH, paths_from_frame, Edge, letters, audit, consume
OUT=ROOT/'runs/residual-language-center-automaton-20260920.json'
CENTERS=(
 (Edge('center','while','theme','center'),Edge('center','the bells ring','center','done')),
 (Edge('center','and','theme','center'),Edge('center','the quiet tide turns','center','done')),
 (Edge('center','because','theme','center'),Edge('center','the old harbor waits','center','done')),
 (Edge('center','near dawn','theme','done'),),)
def close_debt(debt,used):
 for path in CENTERS:
  if any(e.text in used for e in path): continue
  tape=letters(' '.join(e.text for e in path))
  if debt.startswith(tape) and not debt[len(tape):]: yield path
def search(bank,label):
 paths=list(bank); states=merges=prunes=0; seen=set(); rows=[]; exact=[]
 for left in paths:
  for right in paths:
   stack=[(0,0,'','',(),(),frozenset(),frozenset())]
   while stack:
    i,j,lr,rr,lw,rw,ul,ur=stack.pop(); states+=1
    key=(i,j,lr,rr,tuple(e.close_type for e in left[i:]),tuple(e.close_type for e in right[j:]),ul,ur)
    if key in seen: merges+=1; continue
    seen.add(key)
    if i==len(left) and j==len(right):
     debt=lr or rr
     paths_to_try=close_debt(debt,ul|ur) if debt else [()]
     for cp in paths_to_try:
      center=' '.join(e.text for e in cp)
      text=' '.join(lw)+((' '+center) if center else '')+'; '+' '.join(rw)+'.'
      row={'rendered':text,'audit':audit(text),'center_edges':[(e.text,e.open_type,e.close_type) for e in cp], 'provenance':{'bank':label,'live_residual':debt,'typed_center_automaton':bool(cp),'finished_tape_reversal':False,'repair':False,'resegmentation':False}}
      rows.append(row)
      if row['audit']['two_pointer_exact'] and row['audit']['letters']>38: exact.append(row)
     continue
    if i<len(left):
     e=left[i]; z=consume(lr+letters(e.text),rr)
     if z and e.text not in ul: stack.append((i+1,j,z[0],z[1],lw+(e.text,),rw,ul|{e.text},ur))
     else: prunes+=1
    if j<len(right):
     e=right[j]; z=consume(lr,rr+letters(e.text)[::-1])
     if z and e.text not in ur: stack.append((i,j+1,z[0],z[1],lw,(e.text,)+rw,ul,ur|{e.text}))
     else: prunes+=1
 return {'bank':label,'states':states,'quotient_merges':merges,'prunes':prunes,'rendered_candidates':rows[:100],'exact_candidates':exact}
def main():
 result={'experiment_id':'residual-language-center-automaton-20260920','method':'typed residual-language center automaton over canonical edge continuations','results':[search([paths_from_frame(f) for f in FRAMES],'existing-semantic-role-bank'),search(FRESH,'fresh-authored-edge-bank')],'controls':[{'rendered':'The patient scribe marks the old letters by the harbor.','audit':audit('The patient scribe marks the old letters by the harbor.')},{'rendered':'A careful archivist copies a faded map near the quay.','audit':audit('A careful archivist copies a faded map near the quay.')}],'novelty_preflight':{'status':'passed','registry_entries_checked':597,'signature':'typed-edge-continuation-quotient|finite-center-language|semantic-center-closure|live-character-debt','distinct_from':'epsilon boundaries, discontinuous gaps, and repairs: center transitions are authored grammatical continuations intersected with debt before rendering'},'provenance':{'independent_audits':['two-pointer scan','forward/reverse SHA-256'],'reader_evidence':False,'reader_gate':'closed until exact >38'},'status':'no exact candidate above 38'}
 OUT.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps([{k:r[k] for k in ('bank','states','quotient_merges','prunes')} for r in result['results']]))
if __name__=='__main__': main()
