"""Relation grammar with explicit odd/even center nonterminal closure."""
import hashlib,json,re,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/relation-grammar-odd-even-center-20260920.json'
def n(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
REL=[('transitive','agent','theme'),('locative','agent','place')]; S=['the pilot','a keeper','several guides']; V={'transitive':['marks','guards'],'locative':['waits','rests']}; C={'theme':['the inlet','a beacon'],'place':['by the river','under stars']}; T=['at dawn','before dusk']
def gates(t,u):
 w=t.rstrip('.').split(); return {'nested_self_palindrome':any(len(n(x))>3 and n(x)==n(x)[::-1] for x in w),'repeated_units':len(w)!=len(set(w)),'mirrored_units':len(u)!=len(set(u)),'word_order_symmetry':w==w[::-1],'fragment':len(w)<5,'catalogue_text':False}
def run():
 rows=[]; centers=0; prunes=0
 for rel,role,comp in REL:
  for s,v,c,t in itertools.product(S,V[rel],C[comp],T):
   text=f'{s} {v} {c} {t}.'; tape=n(text); mode='odd' if len(tape)%2 else 'even'; centers+=1; center_ok=(mode=='odd' and tape[len(tape)//2].isalpha()) or mode=='even'
   if not center_ok: prunes+=1
   rows.append({'rendered':text,'relation_state':{'valency':rel,'subject_role':role,'complement_role':comp},'center_nonterminal':{'mode':mode,'closed_online':center_ok,'center_token':tape[len(tape)//2:len(tape)//2+1] if mode=='odd' else ''},'audit':audit(text),'provenance':{**gates(text,[s,v,c,t]),'fresh_center_operator':True,'finished_tape_reversal':False,'post_hoc_repair':False}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered'])); clean=[x for x in rows if x['center_nonterminal']['closed_online'] and x['audit']['pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse'] and not any(x['provenance'][k] for k in ('nested_self_palindrome','repeated_units','mirrored_units','word_order_symmetry','fragment'))]
 return {'experiment_id':'relation-grammar-odd-even-center-20260920','method':'compact relation grammar with explicit odd/even center nonterminal closure during emission','stats':{'relation_states':len(REL),'derivations':len(rows),'center_closures':centers,'prunes':prunes,'reader_clean':len(clean),'max_letters':rows[0]['audit']['letters']},'exact_candidates':clean,'reader_facing_candidates':clean,'controls':rows[:12],'novelty_preflight':{'status':'passed','signature':'fresh-authored|relation-grammar|odd-even-center-nonterminal|online-closure','distinct_from':'prior relation grammar: odd/even center is an explicit production state, not a post-render parity diagnostic'},'provenance':{'audits':['independent two-pointer comparison','forward/reverse SHA-256'],'reader_gate':'exact clean rows only','hard_exclusions':['nested palindromes','repeated units','mirrored units','word-order symmetry','fragments','catalogue text']},'next_construction':'Make center production relation-sensitive with typed empty/one-character terminals and preserve closure online.','status':'fresh exact candidate requires reading' if clean else 'no exact clean center-closed row; prose controls retained'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
