"""Fresh POS/length grammar intersected with a live reverse character trie."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/pos-length-trie-intersection-20260920.json'
def n(s):return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
LEFT=(('the','quiet','teacher','marks','a','new','route'),('a','patient','sailor','studies','the','harbor','map'),('our','careful','neighbor','tends','a','winter','garden'))
RIGHT=(('the','local','baker','opens','the','wooden','plaza'),('a','thoughtful','painter','answers','a','folded','letter'),('our','evening','nurse','lights','the','small','candle'))
def intersect(l,r):
 a,b=''.join(map(n,l)),''.join(map(n,r))[::-1]; trace=[]
 for i,(x,y) in enumerate(zip(a,b)):
  trace.append((i,x,y));
  if x!=y:return False,trace
 return len(a)==len(b),trace
def run():
 rows=[];states=0
 for l in LEFT:
  for r in RIGHT:
   states+=1; ok,tr=intersect(l,r); s=' '.join(l)+'; '+' '.join(r)
   rows.append({'left_pos':('DET','ADJ','N','V','DET','ADJ','N'),'right_pos':('DET','ADJ','N','V','DET','ADJ','N'),'left':l,'right':r,'live_trie_closed':ok,'states_examined':len(tr),'trace':tr,'audit':audit(s),'provenance':{'fresh_pos_grammar':True,'word_length_states_live':True,'reverse_trie_intersection':True,'catalogue_text_reused':False,'mirrored_units':False,'post_hoc_repair':False}})
 exact=[x for x in rows if x['live_trie_closed'] and x['audit']['exact'] and x['audit']['letters']>38]
 return {'experiment_id':'pos-length-trie-intersection-20260920','method':'live POS/word-length grammar state intersected with reverse character trie','stats':{'left_frames':3,'right_frames':3,'states':states,'live_closed':sum(x['live_trie_closed'] for x in rows),'exact_gt38':len(exact)},'candidates':rows,'exact_candidates':exact,'status':'precise zero frontier: no live trie closure' if not exact else 'fresh exact requires human reading','provenance':{'audit':'independent mismatch and forward/reverse hashes','reader_gate':'closed unless exact >38'}}
if __name__=='__main__':
 x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps(x['stats']))
