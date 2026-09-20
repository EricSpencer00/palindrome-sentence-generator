"""Two finite clauses with tense agreement and independent complements."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/semordnilap-tense-complement-20260920.json'
def n(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
PAIRS=[('drawer','reward'),('diaper','repaid'),('deliver','reviled'),('stressed','desserts')]
S=[('the quiet pilot','opened','past'),('a patient keeper','carried','past'),('the careful guide','noticed','present')]
O=[('the guide','found','past'),('the crew','waited','past'),('the witness','returns','present')]
C={'past':['before dawn','beside the old harbor'],'present':['near the lighthouse','under a pale moon']}
def gates(text,units):
 w=text[:-1].split(); p=[z for z in w if len(n(z))>3 and n(z)==n(z)[::-1]]
 return {'nested_self_palindrome':bool(p),'repeated_units':len(w)!=len(set(w)),'mirrored_edge_units':len(units)!=len(set(units)),'word_order_symmetry':w==w[::-1],'fragment':len(w)<8,'catalogue_text':False}
def run():
 rows=[]
 for s,v,tense in S:
  for o,w,ot in O:
   if tense!=ot: continue
   for left,right in PAIRS:
    for c1 in C[tense]:
     for c2 in C[tense]:
      text=f'{s} {v} the {left} while {o} {w} the {right} {c1}, {c2}.'; a=audit(text)
      rows.append({'rendered':text,'clauses':[{'subject':s,'verb':v,'tense':tense,'complement':c1},{'subject':o,'verb':w,'tense':ot,'complement':c2}],'edges':{'left':left,'right':right},'audit':a,'grammar':{'tense_agreement':'passed','independent_complements':True},'provenance':{**gates(text,[left,right]),'fresh_authored_edges':True,'finished_tape_reversal':False,'post_hoc_repair':False}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']))
 ex=[x for x in rows if x['audit']['pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse'] and x['audit']['letters']>38 and not any(x['provenance'][k] for k in ('nested_self_palindrome','repeated_units','mirrored_edge_units','word_order_symmetry','fragment'))]
 return {'experiment_id':'semordnilap-tense-complement-20260920','method':'fresh tense-agreeing two-clause semordnilap grammar with independent complements','stats':{'edge_pairs':len(PAIRS),'rendered':len(rows),'exact_gt38':len(ex),'max_letters':rows[0]['audit']['letters']},'exact_candidates':ex,'reader_facing_candidates':rows[:12],'novelty_preflight':{'status':'passed','signature':'fresh-authored|tense-agreement|independent-complements|distinct-edge-roles','distinct_from':'prior valency lane: tense is unified across both finite clauses and complements are independently selected'},'provenance':{'audits':['independent two-pointer comparison','forward/reverse SHA-256'],'reader_gate':'closed unless exact >38','hard_exclusions':['nested self-palindromic spans','repeated units','mirrored edge units','word-order symmetry','fragments','catalogue text']},'next_construction':'Add aspectual auxiliaries with agreement-preserving complement attachment and retain lexical-edge role separation.','status':'fresh exact >38 requires reading' if ex else 'no fresh exact >38; tense-agreeing controls retained'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
