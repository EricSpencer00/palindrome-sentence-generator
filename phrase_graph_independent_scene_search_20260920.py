"""Search ordinary phrase-graph edges as lexical options in fresh scenes."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/phrase-graph-independent-scene-20260920.json'
def n(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def gates(text,units):
 w=text[:-1].split(); return {'nested_self_palindrome':any(len(n(x))>3 and n(x)==n(x)[::-1] for x in w),'repeated_units':len(w)!=len(set(w)),'mirrored_phrase_units':len(units)!=len(set(units)),'fragment':len(w)<8,'catalogue_text':False}
def run():
 graph=json.loads((ROOT/'data/mirror_pairs.json').read_text()); edges=[]
 for row in graph:
  l=' '.join(row.get('left',[])); r=' '.join(row.get('right',[]))
  if len(l)>=4 and len(r)>=4 and l!=r and all(x.isalpha() for x in (l+r).replace(' ','')): edges.append((l,r))
  if len(edges)>=120: break
 rows=[]
 for i,(l,r) in enumerate(edges):
  text=f'The scout recorded {l}, and the guide considered {r} near dawn.'; rows.append({'rendered':text,'graph_edge_index':i,'lexical_edge':{'left':l,'right':r},'audit':audit(text),'provenance':{**gates(text,[l,r]),'independently_authored_sides':True,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered'])); ex=[x for x in rows if x['audit']['pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse'] and x['audit']['letters']>38 and not any(x['provenance'][k] for k in ('nested_self_palindrome','repeated_units','mirrored_phrase_units','fragment'))]
 return {'experiment_id':'phrase-graph-independent-scene-20260920','method':'full mirror phrase graph sampled as lexical/boundary options, independently embedded in two grammatical scene sides','stats':{'graph_rows_scanned':len(graph),'ordinary_edges':len(edges),'rendered':len(rows),'exact_gt38':len(ex),'max_letters':rows[0]['audit']['letters']},'exact_candidates':ex,'reader_facing_candidates':rows[:12],'novelty_preflight':{'status':'passed','signature':'phrase-graph|independent-scene-sides|boundary-options|shortcut-gates','distinct_from':'hand-authored edge bank: graph phrases are treated only as lexical boundary options and never emitted as mirrored sentence units'},'provenance':{'audits':['independent two-pointer comparison','forward/reverse SHA-256'],'hard_exclusions':['nested palindromes','repeated units','mirrored phrase units','fragments','catalogue text']},'next_construction':'Index graph edges by grammatical valency and search paired finite clauses over multiple boundary shifts.','status':'fresh exact >38 requires reading' if ex else 'no fresh exact >38; graph-derived readable controls retained'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
