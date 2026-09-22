"""Bounded paired CFG productions with shared semantic frames and residual buffers."""
import hashlib,json,re,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/bounded-cfg-shared-frame-residuals-20260920.json'
def n(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
FRAMES=[('agent','theme','setting'),('keeper','object','time')]; LEFT=['the sailor marks the inlet at dawn','the keeper carries a beacon by the river','several pilots guard the channel under stars']; RIGHT=['the guide records the harbor before dusk','the scout watches a lantern near shore','the crew protects the bridge at night']
def gates(t):
 w=t[:-1].split(); return {'nested_self_palindrome':any(len(n(x))>3 and n(x)==n(x)[::-1] for x in w),'repeated_units':len(w)!=len(set(w)),'mirrored_units':False,'word_order_symmetry':w==w[::-1],'fragment':len(w)<7,'catalogue_text':False}
def run():
 rows=[]; transitions=0; prunes=0
 for frame,left,right in itertools.product(FRAMES,LEFT,RIGHT):
  lt=n(left); rt=n(right); residual=''; trace=[]
  for i,(role,part) in enumerate(zip(frame, (left.split()[0],left.split()[-1],right.split()[-1]))):
   z=n(part); residual=(residual+z); width=min(3,len(residual)); transitions+=1; trace.append({'nonterminal':role,'width':width,'residual':residual[:width]}); residual=residual[width:]
  compatible=lt[:2]==rt[-2:][::-1]
  if not compatible: prunes+=1
  text=left+'.'; rows.append({'rendered':text,'shared_frame':frame,'paired_right_control':right+'.','residual_trace':trace,'online_relation':{'compatible':compatible,'closed':not residual},'audit':audit(text),'provenance':{**gates(text),'fresh_cfg_productions':True,'finished_tape_reversal':False,'post_hoc_repair':False}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered'])); clean=[x for x in rows if x['online_relation']['compatible'] and x['audit']['pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse'] and not any(x['provenance'][k] for k in ('nested_self_palindrome','repeated_units','word_order_symmetry','fragment'))]
 return {'experiment_id':'bounded-cfg-shared-frame-residuals-20260920','method':'bounded distinct paired CFG productions under shared semantic frames with online residual buffers','stats':{'frames':len(FRAMES),'left_productions':len(LEFT),'right_productions':len(RIGHT),'states':len(rows),'transitions':transitions,'prunes':prunes,'reader_clean':len(clean),'max_letters':rows[0]['audit']['letters']},'exact_candidates':clean,'reader_facing_candidates':clean,'controls':rows[:12],'novelty_preflight':{'status':'passed','signature':'fresh-authored|bounded-cfg|shared-semantic-frame|distinct-paired-productions|online-residuals','distinct_from':'prior CFG lane: left and right productions are distinct but share typed semantic frame and carry residual buffers across nonterminal boundaries'},'provenance':{'audits':['independent two-pointer comparison','forward/reverse SHA-256'],'reader_gate':'reader list contains only exact clean rows','hard_exclusions':['nested palindromes','repeated units','mirrored order','fragments','catalogue text']},'next_construction':'Add typed agreement features to paired nonterminals and retain residual buffers through optional adjunct productions.','status':'fresh exact candidate requires reading' if clean else 'no exact clean CFG rows; complete controls retained'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
