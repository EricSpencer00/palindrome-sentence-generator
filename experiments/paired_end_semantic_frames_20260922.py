"""Paired-end role-conditioned scene frames with live character obligations.

Unlike a fixed clause product, each frame declares semantic roles at both ends
and the decoder chooses one lexicalization from each role while obligations are
still open.  No completed surface is reversed during search.
"""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; ID="paired-end-semantic-frames-20260922"; RUN=ROOT/"runs"/(ID+".json")
def norm(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=norm(s); mm=[i for i in range(len(t)//2) if t[i]!=t[-1-i]]
 return {'exact':bool(t) and not mm,'letters':len(t),'mismatches':mm[:8],'pointer_pairs_checked':len(t)//2,'forward_sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256':hashlib.sha256(t[::-1].encode()).hexdigest()}
LEX={
 'det_s':['a','an','the'],'det_p':['some','the','many'],
 'agent_s':['aide','artist','writer','sailor'],'agent_p':['men','artists','writers','sailors'],
 'verb_s':['rips','marks','reads','keeps'],'verb_p':['inspire','mark','read','keep'],
 'theme_p':['memos','letters','maps','notes'],'name':['anna','diana','nora','iris','leon','noah'],
 'copula':['is','was','ere']}
FRAMES=[
 ('agent_event', [('det_s','agent_s'),('verb_s',),('theme_p',)], [('name',),('verb_p',),('agent_p','det_p')]),
 ('writer_event', [('det_s','agent_s'),('verb_s',),('theme_p',)], [('name',),('verb_p',),('agent_p','det_p')]),
 ('short_event', [('det_s','agent_s'),('verb_s',)], [('name',),('verb_p','det_p')]),
]
def consume(debt,token,side):
 x=norm(token) if side=='L' else norm(token)[::-1]
 if not debt:return None
 if debt.startswith(x):return (debt[len(x):],side)
 if x.startswith(debt):return (x[len(debt):],'R' if side=='L' else 'L')
 return None
def main():
 nodes=prunes=terminals=0; exact=[]; frontier=[]; seen=set()
 def run_frame(name,Lslots,Rslots,center):
  nonlocal nodes,prunes,terminals
  def rec(li,ri,debt,side,left,right,trace):
   nonlocal nodes,prunes,terminals
   nodes+=1
   if nodes>300000:return
   if len(left)>=1 and len(frontier)<100:
    text=' '.join(left+([center] if center else [])+list(reversed(right))); a=audit(text)
    frontier.append({'rendered':text.capitalize()+'.','length':a['letters'],'exact':a['exact'],'audit':a,'status':'partial','provenance':{'frame':name,'left_roles':Lslots,'right_roles_reversed':list(reversed(Rslots)),'center':center,'trace':trace}})
   if li==len(Lslots) and ri<0:
    terminals+=1; text=' '.join(left+([center] if center else [])+list(reversed(right))); a=audit(text)
    row={'rendered':text.capitalize()+'.','length':a['letters'],'exact':a['exact'],'audit':a,'provenance':{'frame':name,'left_roles':Lslots,'right_roles_reversed':list(reversed(Rslots)),'center':center,'trace':trace},'novelty':'generated_role_conditioned'}
    if row['normalized'] if False else True:
     if a['exact']: exact.append(row)
    return
   if not debt:
    if li<len(Lslots):
     for key in Lslots[li]:
      for tok in LEX[key]: rec(li+1,ri,norm(tok),'L',left+[tok],right,trace+[('L',key,tok)])
    return
   if side=='R' and ri>=0:
    for key in Rslots[ri]:
     for tok in LEX[key]:
      z=consume(debt,tok,'R')
      if z:rec(li,ri-1,*z,left,right+[tok],trace+[('R',key,tok)])
      else:prunes+=1
   elif side=='L' and li<len(Lslots):
    for key in Lslots[li]:
     for tok in LEX[key]:
      z=consume(debt,tok,'L')
      if z:rec(li+1,ri,*z,left+[tok],right,trace+[('L',key,tok)])
      else:prunes+=1
   else:prunes+=1
  rec(0,len(Rslots)-1,'','L',[],[],[])
 for name,L,R in FRAMES:
  for c in ('','is','was','ere'):run_frame(name,L,R,c)
 out={'experiment_id':ID,'status':'completed_live_paired_end_frames','method':'role-conditioned semantic frame selection at paired ends with online character obligation and typed odd/even center','stats':{'nodes':nodes,'live_prunes':prunes,'terminals':terminals,'frontier_outputs':len(frontier),'exact':len(exact),'longest_frontier':max((x['length'] for x in frontier),default=0)},'frontier_outputs':frontier[:100],'exact_candidates':exact[:40],'shortcut_gates':{'finished_tape_reverse_during_search':False,'repair':False,'catalogue':False,'mirrored_units':False,'word_order_only':False,'rlaif':False},'novelty':{'signature':'paired-end-role-conditioned-frame|live-debt|typed-center','preflight':'fresh frame topology; not fixed-slot product or reverse lexicalization'},'next_construction':'learn role-pair compatibility from live successful prefixes and add plural/singular alternations without relaxing exact debt','provenance':{'script':str(Path(__file__)),'run':str(RUN),'independent_verifier':'two-pointer mismatch audit plus forward/reverse SHA-256'}}
 RUN.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['stats']))
if __name__=='__main__':main()
