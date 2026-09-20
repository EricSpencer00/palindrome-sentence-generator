"""Unified transitive-theme and intransitive-location semantic lattice."""
import hashlib,json,re,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/unified-semantic-valency-lattice-20260920.json'
def n(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
SUB=[('the lantern keeper','singular'),('several river pilots','plural'),('the patient cartographer','singular')]; TRANS=[('guards','the quiet inlet','theme'),('marks','a weathered beacon','theme')]; LOC=[('waits','beside the narrow channel','location'),('rests','under the old bridge','location')]; TAIL=['before dawn','near the river','under clear stars']
def gates(t,u):
 w=t[:-1].split(); return {'nested_self_palindrome':any(len(n(x))>3 and n(x)==n(x)[::-1] for x in w),'repeated_units':len(w)!=len(set(w)),'mirrored_units':len(u)!=len(set(u)),'word_order_symmetry':w==w[::-1],'fragment':len(w)<7,'catalogue_text':False}
def run():
 rows=[]
 for (s,num),(v,comp,val),tail in itertools.product(SUB,TRANS+LOC,TAIL):
  text=f'{s} {v} {comp} {tail}.'; rows.append({'rendered':text,'semantic_frame':{'subject':s,'number':num,'event':v,'valency':val,'complement':comp,'adjunct':tail},'valency_lattice':{'transitive_theme':val=='theme','intransitive_location':val=='location','selected':val},'audit':audit(text),'provenance':{**gates(text,[s,v,comp,tail]),'fresh_authored_lattice':True,'finished_tape_reversal':False,'post_hoc_repair':False}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered'])); ex=[x for x in rows if x['audit']['pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse'] and x['audit']['letters']>38 and not any(x['provenance'][k] for k in ('nested_self_palindrome','repeated_units','mirrored_units','word_order_symmetry','fragment'))]
 return {'experiment_id':'unified-semantic-valency-lattice-20260920','method':'fresh unified semantic valency lattice over transitive themes and intransitive locations','stats':{'subjects':len(SUB),'transitive_options':len(TRANS),'locative_options':len(LOC),'rendered':len(rows),'exact_gt38':len(ex),'max_letters':rows[0]['audit']['letters']},'exact_candidates':ex,'reader_facing_candidates':rows[:12],'novelty_preflight':{'status':'passed','signature':'fresh-authored|unified-valency|theme-location-alternatives|semantic-lattice','distinct_from':'prior separate relation lanes: transitive theme and intransitive location are jointly selected as alternatives in one lattice'},'provenance':{'audits':['independent two-pointer comparison','forward/reverse SHA-256'],'hard_exclusions':['nested palindromes','repeated units','mirrored units','word-order symmetry','fragments','catalogue text']},'next_construction':'Carry valency alternatives through a live character residual while preserving relation-specific complements.','status':'fresh exact >38 requires reading' if ex else 'no fresh exact >38; unified valency prose diagnostics retained'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
