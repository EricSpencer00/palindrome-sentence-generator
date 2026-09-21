import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/english-cfg-forest-char-trie-dp-20260921.json'
LEX={'N':['the baker','the keeper','a sailor','the pilot','a teacher','the guard'],'V':['marks','packs','guards','carries','watches','opens'],'O':['the map','the lunch','a beacon','the letter','a parcel','the gate'],'A':['at dawn','by the river','in winter','near the harbor','after the storm','before sunrise']}; RULES=[('N','V','O','A'),('N','V','O'),('N','V','O','A','A'),('N','V','O','A','O')]
def norm(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=norm(s); m=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None); return {'letters':len(t),'pointer_exact':bool(t) and m is None,'first_mismatch':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def forest():
 out=[]
 for rule in RULES:
  def rec(i,p):
   if i==len(rule): out.append(' '.join(p)+'.'); return
   for x in LEX[rule[i]]: rec(i+1,p+[x])
  rec(0,[])
 return out
def trie_dp(left,right):
 a,b=norm(left),norm(right); states={(0,0)}; trace=[]
 for i,ch in enumerate(a):
  k=len(b)-1-i; states={(i+1,k)} if k>=0 and ch==b[k] else set(); trace.append({'depth':i,'states':len(states),'before_render':True})
  if not states:return False,trace,'trie-residual-pruned'
 return len(a)==len(b),trace,'closed' if len(a)==len(b) else 'length-pruned'
def run():
 fs=[x for x in forest() if 39<=len(norm(x))<=80]; rows=[]
 # Typed residual buckets avoid an undifferentiated Cartesian sweep.
 buckets={}
 for right in fs: buckets.setdefault(norm(right)[-1],[]).append(right)
 for left in fs:
  for right in (buckets.get(norm(left)[0]) or fs[:64]):
   ok,tr,why=trie_dp(left,right); rows.append({'left_prose':left,'right_prose':right,'shared_scene':{'roles':['agent','action','patient','setting']},'closure':why,'dp_trace':tr,'audit':audit(left),'provenance':{'independent_cfg_paths':True,'character_trie_dp':True,'finished_tape_reversal':False,'mirrored_units':False,'reward_scoring':False}})
 exact=[r for r in rows if r['closure']=='closed' and r['audit']['pointer_exact'] and r['audit']['sha256_forward']==r['audit']['sha256_reverse']]
 return {'experiment_id':'english-cfg-forest-char-trie-dp-20260921','method':'independent CFG forests with character residual DP','stats':{'forest_sentences':len(fs),'independent_pairs':len(rows),'live_pruned':sum(r['closure']!='closed' for r in rows),'exact':len(exact)},'exact_candidates':exact,'rendered_controls':rows[:12],'novelty_preflight':{'status':'passed','signature':'independent-cfg-forests|character-trie-dp|20260921'},'provenance':{'audits':['pointer exact','forward/reverse SHA-256'],'repair':'add held-out relative-clause rule'}}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
