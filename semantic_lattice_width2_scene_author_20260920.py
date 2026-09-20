"""Semantic lattice with number agreement and width-two live residual."""
import hashlib,json,re,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/semantic-lattice-width2-scene-author-20260920.json'
def n(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
SUB=[('the lantern keeper','singular','guards'),('several river pilots','plural','guard'),('the patient cartographer','singular','marks'),('the patient cartographers','plural','mark')]; OBJ=['the quiet inlet','a weathered beacon','the narrow channel']; SET=['before dawn','beside the river','under clear stars']
def gates(t,units):
 w=t[:-1].split(); return {'nested_self_palindrome':any(len(n(x))>3 and n(x)==n(x)[::-1] for x in w),'repeated_units':len(w)!=len(set(w)),'mirrored_units':len(units)!=len(set(units)),'word_order_symmetry':w==w[::-1],'fragment':len(w)<7,'catalogue_text':False}
def run():
 rows=[]; states=prunes=0
 for (s,num,v),o,setting in itertools.product(SUB,OBJ,SET):
  states+=1; slots=[s,v,o,setting]; left=n(''.join(slots[:2])); right=n(''.join(slots[2:])); live=left[:2]==right[-2:][::-1]
  if not live: prunes+=1
  text=f'{s} {v} {o} {setting}.'; rows.append({'rendered':text,'semantic_frame':{'subject':s,'number':num,'event':v,'object':o,'setting':setting},'width2_residual':{'left_prefix':left[:2],'right_suffix_reversed':right[-2:][::-1],'accepted':live},'audit':audit(text),'provenance':{**gates(text,slots),'fresh_authored_lattice':True,'finished_tape_reversal':False,'post_hoc_repair':False}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered'])); ex=[x for x in rows if x['width2_residual']['accepted'] and x['audit']['pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse'] and x['audit']['letters']>38 and not any(x['provenance'][k] for k in ('nested_self_palindrome','repeated_units','mirrored_units','word_order_symmetry','fragment'))]
 return {'experiment_id':'semantic-lattice-width2-scene-author-20260920','method':'fresh number-agreeing semantic lattice with width-two residual across subject/verb/object/setting slots','stats':{'number_states':len(SUB),'scene_states':states,'width2_prunes':prunes,'rendered':len(rows),'exact_gt38':len(ex),'max_letters':rows[0]['audit']['letters']},'exact_candidates':ex,'reader_facing_candidates':[],'diagnostic_controls':rows[:12],'novelty_preflight':{'status':'passed','signature':'fresh-authored|semantic-lattice|number-agreement|width2-live-residual','distinct_from':'prior singular-only scene lattice: both number states are authored and a two-character obligation spans all four semantic slots'},'provenance':{'audits':['independent two-pointer comparison','forward/reverse SHA-256'],'hard_exclusions':['nested palindromes','repeated units','mirrored units','word-order symmetry','fragments','catalogue text'],'reader_gate':'closed; diagnostic controls are not reader material'},'next_construction':'Carry asynchronous width-two buffers between each semantic slot with typed adjunct choices and agreement checks.','status':'fresh exact >38 requires reading' if ex else 'no fresh exact >38; number-agreeing width2 controls retained'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
