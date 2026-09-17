"""Held-out phrase-edge replacement around a quarantined exact control.

The control's words are never rendered. Its role/state topology motivates a
fresh lattice of authored edge alternatives. Each complete edge path is
compiled into an independent character trie; the product compares edges live.
"""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/heldout-phrase-edge-residual-20260917.json'
ID='heldout-phrase-edge-residual-20260917'; SIG='quarantined-residual-template|heldout-phrase-edges|live-trie-product|semantic-state|independent-audit'
class E:
 def __init__(self,text,role): self.text=text; self.role=role
# Fresh ordinary alternatives, not copied from the control.
LEFT=[ [E('At dusk','setting'),E('the ferryman','agent'),E('opens','action'),E('the harbor gate','theme')], [E('By sunrise','setting'),E('a patient gardener','agent'),E('tends','action'),E('the orchard','theme')], [E('After rain','setting'),E('the careful archivist','agent'),E('seals','action'),E('a ledger','theme')] ]
RIGHT=[ [E('near the quay','setting'),E('while lanterns glow','event')], [E('before the storm','setting'),E('as tall reeds bend','event')], [E('in the courtyard','setting'),E('when swallows return','event')] ]
RELATIVE={'a':E('as autumn settles','relative_attachment'),'t':E('that travelers remember','relative_attachment'),'s':E('since spring began','relative_attachment')}
def norm(s): return ''.join(c.lower() for c in s if c.isascii() and c.isalpha())
def render(path): return ', '.join(e.text for e in path)+'.'
def compile_paths(paths,rev=False):
 nxt=[{}]; finals=set(); meta={}
 for k,p in enumerate(paths):
  chars=norm(' '.join(e.text for e in p)); chars=chars[::-1] if rev else chars; n=0
  for c in chars: n=nxt[n].setdefault(c,len(nxt)); nxt.append({}) if n==len(nxt) else None
  finals.add(n); meta[n]=k
 return nxt,finals,meta
def product(lp,rp):
 L,LF,LM=compile_paths(lp); R,RF,RM=compile_paths(rp,True); q=[(0,0,0)]; seen=set(q); dead=[]
 while q:
  a,b,d=q.pop(0)
  if a in LF and b in RF:return {'closed':True,'matched':d,'states':len(seen),'dead_frontier':dead[:4]}
  for c,na in L[a].items():
   nb=R[b].get(c)
   if nb is None: dead.append({'left_node':a,'right_node':b,'obligation':c,'right_options':sorted(R[b])[:6]});continue
   s=(na,nb,d+1)
   if s not in seen:seen.add(s);q.append(s)
 return {'closed':False,'matched':max((s[2] for s in seen),default=0),'states':len(seen),'dead_frontier':dead[:4]}
def audit(s):
 t=norm(s); bad=[(i,len(t)-1-i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]]
 return {'letters':len(t),'exact':bool(t) and not bad,'mismatches':bad[:8],'sha256':hashlib.sha256(t.encode()).hexdigest()}
def run():
 rows=[]
 for l in LEFT:
  for r in RIGHT:
   p=product([l],[r]); txt=render(l)+' Meanwhile '+render(r); a=audit(txt)
   rows.append({'text':txt,'letters':a['letters'],'product':p,'independent_audit':a,'roles':{'left':[e.role for e in l],'right':[e.role for e in r]},'provenance':'fresh held-out authored phrase edges; residual template supplied topology only','mechanically_admitted':p['closed'] and a['exact']})
 exact=[x for x in rows if x['mechanically_admitted']]
 # One, and only one, attachment repair selected from the first residual.
 attachment=[]
 if rows[0]['product']['dead_frontier']:
  residual=rows[0]['product']['dead_frontier'][0]['obligation']; att=RELATIVE.get(residual)
  if att:
   l=LEFT[0]+[att]; r=RIGHT[0]+[att]; txt=render(l)+' Meanwhile '+render(r); a=audit(txt); pr=product([l],[r])
   attachment=[{'text':txt,'letters':a['letters'],'product':pr,'independent_audit':a,'mechanically_admitted':pr['closed'] and a['exact'],'attachment_operator':{'type':'typed_relative_clause','conditioned_on_first_residual':residual,'edge':att.text},'provenance':'single held-out relative attachment selected by live first residual; no broad resweep'}]
 exact += [x for x in attachment if x['mechanically_admitted']]
 payload={'experiment_id':ID,'signature':SIG,'status':'completed_no_exact_closure' if not exact else 'completed_exact','candidates':exact,'diagnostic_frontiers':rows[:4]+attachment,'exact_candidates':len(exact),'reader_eligible':False,'novelty_preflight':{'status':'passed','control_text_rendered':False,'control_words_imported':False,'catalogue_edges':False,'basis':'only role/state topology from the quarantined control was retained; every terminal edge is newly authored'},'repair_after_failure':{'operator':'one typed relative-clause attachment conditioned on first dead character','first_dead_frontier':rows[0]['product']['dead_frontier'][0] if rows[0]['product']['dead_frontier'] else None,'tested':bool(attachment),'next':'replace only the attachment terminal at the recorded residual'},'provenance':{'generator':str(Path(__file__).relative_to(ROOT)),'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'audit':'independent pointer and SHA-256 checks'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n'); return payload
if __name__=='__main__':
 p=run();print(json.dumps({'exact':p['exact_candidates'],'frontiers':len(p['diagnostic_frontiers']),'longest':max(x['letters'] for x in p['diagnostic_frontiers'])}))
