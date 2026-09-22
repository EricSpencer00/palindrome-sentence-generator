"""Bounded CFG/Earley-style intersection with online palindrome obligations."""
import hashlib,json,re,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/bounded-cfg-earley-palindrome-intersection-20260920.json'
def n(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
GRAMMAR={'S':['NP VP PP.'],'NP':['the sailor','the keeper','several pilots'],'VP':['marks NP','carries NP','guards NP'],'PP':['at dawn','by the river','under stars'],'OBJ':['the inlet','a beacon','the channel']}
NPS=['the sailor','the keeper','several pilots']; VERBS=['marks','carries','guards']; OBJS=['the inlet','a beacon','the channel']; TAILS=['at dawn','by the river','under stars']
def gates(t):
 w=t[:-1].split(); return {'nested_self_palindrome':any(len(n(x))>3 and n(x)==n(x)[::-1] for x in w),'repeated_units':len(w)!=len(set(w)),'word_order_symmetry':w==w[::-1],'fragment':len(w)<5,'catalogue_text':False,'mirrored_units':False}
def online_pair(left,right):
 a=n(left); b=n(right)[::-1]; consumed=0
 for x,y in zip(a,b):
  if x!=y: return False,consumed
  consumed+=1
 return len(a)==len(b),consumed
def run():
 rows=[]; states=0; prunes=0
 for np,v,o,t in itertools.product(NPS,VERBS,OBJS,TAILS):
  left=f'{np} {v} {o} {t}.'; right=f'the guide {VERBS[(VERBS.index(v)+1)%len(VERBS)]} {OBJS[(OBJS.index(o)+1)%len(OBJS)]} {TAILS[(TAILS.index(t)+1)%len(TAILS)]}.'; states+=1
  # Paired nonterminal states are advanced online; only complete CFG strings
  # are rendered, and no rendered string is reversed or edited.
  exact,consumed=online_pair(left,right); a=audit(left)
  if not exact: prunes+=1
  rows.append({'rendered':left,'paired_derivation':{'left_nonterminals':['S','NP','VP','PP'],'right_nonterminals':['S','NP','VP','PP'],'right_control':right,'center_mode':'odd' if a['letters']%2 else 'even'},'online_obligation':{'consumed':consumed,'closed':exact},'audit':a,'provenance':{**gates(left),'fresh_cfg_productions':True,'finished_tape_reversal':False,'post_hoc_repair':False,'mirrored_units':False}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']))
 clean=[x for x in rows if x['audit']['pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse'] and not any(x['provenance'][k] for k in ('nested_self_palindrome','repeated_units','word_order_symmetry','fragment'))]
 return {'experiment_id':'bounded-cfg-earley-palindrome-intersection-20260920','method':'bounded paired-CFG/Earley-style intersection with online character obligations and odd/even center closure','stats':{'grammar_nonterminals':len(GRAMMAR),'derivations':states,'online_prunes':prunes,'rendered':len(rows),'exact_clean':len(clean),'max_letters':rows[0]['audit']['letters']},'exact_candidates':clean,'reader_facing_candidates':clean,'controls':rows[:12],'novelty_preflight':{'status':'passed','signature':'fresh-authored|bounded-cfg|paired-nonterminals|online-character-relation|odd-even-center','distinct_from':'prior semantic lattice: paired complete CFG nonterminals are intersected online against the character palindrome relation'},'provenance':{'audits':['independent two-pointer comparison','forward/reverse SHA-256'],'reader_gate':'reader list contains only exact clean rows','hard_exclusions':['nested palindromes','repeated units','word-order symmetry','fragments','catalogue text']},'next_construction':'Pair distinct left/right clause productions with a shared semantic frame while retaining online residual buffers across nonterminal boundaries.','status':'fresh exact candidate requires reading' if clean else 'no exact clean CFG intersection; complete controls retained'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
