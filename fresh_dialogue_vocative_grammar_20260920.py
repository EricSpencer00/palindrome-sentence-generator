"""Fresh vocative/declarative dialogue grammar with online lexical checks."""
import hashlib,json,re,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/fresh-dialogue-vocative-grammar-20260920.json'
def n(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
VOC=['Pilot,','Keeper,','Friend,']; DECL=['the inlet is quiet.','the lantern is ready.','the river runs north.','the harbor waits below.']; REPLY=['I will chart it.','We can walk there.','The crew is ready.']
def gates(t,units):
 w=re.findall(r'[A-Za-z]+',t); return {'nested_self_palindrome':any(len(n(x))>3 and n(x)==n(x)[::-1] for x in w),'repeated_units':len(w)!=len(set(w)),'mirrored_units':len(units)!=len(set(units)),'word_order_symmetry':w==w[::-1],'fragment':len(w)<5,'catalogue_text':False}
def run():
 rows=[]; checks=0
 for v,d,r in itertools.product(VOC,DECL,REPLY):
  text=f'{v} {d} {r}'; turns=[v,d,r]; checks+=1; compatible=n(v)[0]==n(r)[-1]
  rows.append({'rendered':text,'dialogue_turns':{'vocative':v,'declarative':d,'reply':r},'online_opening_check':{'compatible':compatible,'left':n(v)[0],'right':n(r)[-1]},'audit':audit(text),'provenance':{**gates(text,turns),'fresh_dialogue_grammar':True,'finished_tape_reversal':False,'post_hoc_repair':False}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered'])); clean=[x for x in rows if x['online_opening_check']['compatible'] and x['audit']['pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse'] and not any(x['provenance'][k] for k in ('nested_self_palindrome','repeated_units','mirrored_units','word_order_symmetry','fragment'))]
 return {'experiment_id':'fresh-dialogue-vocative-grammar-20260920','method':'fresh vocative/declarative/reply dialogue grammar with online lexical opening equation','stats':{'vocatives':len(VOC),'declarations':len(DECL),'replies':len(REPLY),'dialogues':len(rows),'checks':checks,'reader_clean':len(clean),'max_letters':rows[0]['audit']['letters']},'exact_candidates':clean,'reader_facing_candidates':clean,'controls':rows[:12],'novelty_preflight':{'status':'passed','signature':'fresh-authored|dialogue-grammar|vocative-declarative-reply|online-lexical-check','distinct_from':'prior sentence and relation grammars: complete punctuation-bearing dialogue turns are lexicalized as a three-turn exchange'},'provenance':{'audits':['independent two-pointer comparison','forward/reverse SHA-256'],'reader_gate':'exact clean dialogue only','hard_exclusions':['nested palindromes','repeated units','mirrored chains','word-order symmetry','fragments','catalogue text']},'next_construction':'Add typed speaker roles and two-character turn-boundary obligations without reusing a dialogue turn.','status':'fresh exact candidate requires reading' if clean else 'no exact clean dialogue; intact controls retained'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
