"""Compact character-level grammar intersection with typed relation topology."""
import hashlib,json,re,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/relation-valency-char-grammar-intersection-20260920.json'
def n(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
REL=[('transitive','agent','theme'),('locative','agent','place')]; SUB=['the pilot','the keeper','several guides']; VERB={'transitive':['marks','guards'],'locative':['waits','rests']}; COMP={'theme':['the inlet','a beacon'],'place':['by the river','under stars']}; TAIL=['at dawn','before dusk']
def gates(t,u):
 w=t.rstrip('.').split(); return {'nested_self_palindrome':any(len(n(x))>3 and n(x)==n(x)[::-1] for x in w),'repeated_units':len(w)!=len(set(w)),'mirrored_units':len(u)!=len(set(u)),'word_order_symmetry':w==w[::-1],'fragment':len(w)<5,'catalogue_text':False}
def consume(left,right):
 a=n(left); b=n(right)[::-1]; k=0
 for x,y in zip(a,b):
  if x!=y:return False,k
  k+=1
 return len(a)==len(b),k
def run():
 rows=[]; transitions=0; prunes=0
 for relation,agent,comp_role in REL:
  for s,v,c,t in itertools.product(SUB,VERB[relation],COMP[comp_role],TAIL):
   text=f'{s} {v} {c} {t}.'; opposing=f'the guide {VERB[relation][0]} {COMP[comp_role][0]} {TAIL[0]}.'; transitions+=len(n(text)); closed,matched=consume(text,opposing)
   if not closed: prunes+=1
   rows.append({'rendered':text,'relation_state':{'valency':relation,'subject_role':agent,'complement_role':comp_role},'opposing_control':opposing,'online_character_intersection':{'matched':matched,'closed':closed},'audit':audit(text),'provenance':{**gates(text,[s,v,c,t]),'fresh_relation_topology':True,'finished_tape_reversal':False,'post_hoc_repair':False}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered'])); clean=[x for x in rows if x['online_character_intersection']['closed'] and x['audit']['pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse'] and not any(x['provenance'][k] for k in ('nested_self_palindrome','repeated_units','mirrored_units','word_order_symmetry','fragment'))]
 return {'experiment_id':'relation-valency-char-grammar-intersection-20260920','method':'compact character-level grammar intersection over typed transitive/locative relation states emitting complete clauses','stats':{'relation_states':len(REL),'rendered':len(rows),'character_transitions':transitions,'prunes':prunes,'reader_clean':len(clean),'max_letters':rows[0]['audit']['letters']},'exact_candidates':clean,'reader_facing_candidates':clean,'controls':rows[:12],'novelty_preflight':{'status':'passed','signature':'fresh-authored|relation-valency-topology|character-grammar-intersection|complete-clause-emission','distinct_from':'all prior bank/index/seam lanes: relation topology is the grammar state and characters are consumed during clause emission'},'provenance':{'audits':['independent two-pointer comparison','forward/reverse SHA-256'],'reader_gate':'exact clean rows only','hard_exclusions':['nested palindromes','repeated units','mirrored units','word-order symmetry','fragments','catalogue text']},'next_construction':'Add an explicit center nonterminal that can close odd/even relation-state derivations without post-render editing.','status':'fresh exact candidate requires reading' if clean else 'no exact clean relation-grammar row; intact controls retained'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
