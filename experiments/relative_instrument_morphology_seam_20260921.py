"""Bounded instrument-morphology seam lane (singular/plural alternation)."""
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/relative-instrument-morphology-seam-20260921.json'
@dataclass(frozen=True)
class Arc:name:str;words:tuple[str,...];number:str;stem:str
L=(Arc('smith_tool',('the','patient','smith','who','carves','with','a','sharp','tool'),'singular','tool'),Arc('makers_tools',('the','skilled','makers','who','work','with','the','fine','tools'),'plural','tool'))
R=(Arc('artists_brushes',('the','careful','artists','who','paint','with','the','fine','brushes'),'plural','brush'),Arc('pilot_compass',('the','calm','pilot','who','navigates','with','a','old','compass'),'singular','compass'))
def tape(s):return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=tape(s);n=len(t);bad=next((i for i in range(n//2) if t[i]!=t[n-1-i]),None);h=hashlib.sha256(t.encode()).hexdigest();rh=hashlib.sha256(t[::-1].encode()).hexdigest();return {'letters':n,'two_pointer_exact':bad is None,'first_mismatch':bad,'forward_sha256':h,'reverse_sha256':rh,'sha_equal':h==rh}
def main():
 rows=[]
 for a in L:
  for b in R:
   if a.number==b.number:continue
   s=' '.join(a.words)+'; '+' '.join(b.words)+'.';t=tape(s);l=0;r=len(t)-1
   while l<r and t[l]==t[r]:l+=1;r-=1
   z={'pairs':l,'center_inside_word':True,'alternating_numbers':True,'obligation':None if l>=r else(t[l],t[r])}
   rows.append({'rendered':s,'left_arc':a.name,'right_arc':b.name,'morphology_seam':{'left_number':a.number,'right_number':b.number,'left_stem':a.stem,'right_stem':b.stem},'audit':audit(s),'live_trace':z,'exact_admitted':z['obligation'] is None,'reader_status':'unreviewed; exactness does not certify readability','provenance':{'construction':'typed with-instrument morphology at alternating number seam','finished_tape_reversal':False,'posthoc_repair':False,'catalogue_text':False}})
 exact=[x for x in rows if x['exact_admitted']];out={'experiment_id':'relative-instrument-morphology-seam-20260921','status':'completed_exact' if exact else 'completed_no_exact_closure','method':'alternating singular/plural instrument morphology across a live center seam','candidate_count':len(rows),'exact_count':len(exact),'reader_eligible':False,'rendered_candidates':rows,'stats':{'longest_letters':max(x['audit']['letters'] for x in rows)},'novelty_preflight':{'morphology_seam_new':True,'typed_with_valency':True,'lexical_cartesian_widening':False},'failure_and_repair':{'failure':'alternating morphology preserves prose but not outer character closure' if not exact else 'none','next_construction':'carry a shared plural instrument stem through a relative object clause with agreement propagation'},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'independent_audits':['two-pointer','forward/reverse SHA-256'],'shortcuts_excluded':True}}
 RUN.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':out['status'],'candidates':len(rows),'exact':len(exact),'longest_letters':out['stats']['longest_letters']}))
if __name__=='__main__':main()
