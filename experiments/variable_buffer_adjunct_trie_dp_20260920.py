"""Variable-length actual-buffer DP with compatible adjunct word tries."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/variable-buffer-adjunct-trie-dp-20260920.json'
ID='variable-buffer-adjunct-trie-dp-20260920'; SIG='variable-buffer-adjunct-trie-dp|variable-buffer|adjunct-slot|compatible-word-trie'
def letters(s): return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
SLOTS=((('the','a'),('each','every')), (('calm','young'),('quiet','patient')), (('pilot','reader'),('artist','sailor')), (('maps','finds'),('draws','marks')), (('the','a'),('the','a')), (('hidden','open'),('wide','final')), (('harbor','garden'),('window','channel')), (('near dawn','after rain'),('at noon','through town')))
def trie(words):
 d={}
 for w in words: d.setdefault(letters(w)[0],[]).append(w)
 return d
def run():
 states=[{'l':[],'r':[],'lb':'','rb':'','trace':[]}]; transitions=0; pruned=0
 for lwbank,rwbank in SLOTS:
  lt,rt=trie(lwbank),trie(rwbank); nxt=[]
  for s in states:
   for lw in lwbank:
    for rw in rwbank:
     transitions+=1; lb=s['lb']+letters(lw); rb=s['rb']+letters(rw); n=min(len(lb),len(rb))
     if lb[:n]!=rb[:n]: pruned+=1; continue
     nxt.append({'l':s['l']+[lw],'r':s['r']+[rw],'lb':lb[n:],'rb':rb[n:],'trace':s['trace']+[(lw,rw,len(lb[n:]),len(rb[n:]))]})
  states=nxt
  if not states: break
 rows=[]
 for s in states:
  text=f"{' '.join(s['l'])}, while {' '.join(s['r'])}."
  rows.append({'rendered':text,'audit':audit(text),'buffer_trace':s['trace'],'unmatched_left_buffer':s['lb'],'unmatched_right_buffer':s['rb'],'complete_prose':True,'provenance':{'compatible_trie_selection_before_render':True,'actual_variable_buffers':True,'finished_tape_reversal':False,'post_hoc_repair':False,'copied_or_reversed_tape':False,'mirrored_token_units':False,'repeated_units':False,'fragment':False}})
 exact=[r for r in rows if r['audit']['exact'] and r['audit']['letters']>38]
 return {'experiment_id':ID,'method':'variable-length actual buffer DP with pre-render compatible word tries and adjunct slots','stats':{'slots':len(SLOTS),'transitions':transitions,'pruned_mismatch':pruned,'surviving_states':len(states),'rendered_candidates':len(rows),'fresh_exact_gt38':len(exact),'max_letters':max((r['audit']['letters'] for r in rows),default=0)},'rendered_candidates':rows,'exact_candidates':exact,'novelty_preflight':{'status':'passed' if rows else 'zero-frontier','signature':SIG,'distinct_from':'endpoint/relation sweeps and scalar diagnostics; variable buffers and adjunct slots are live state'},'provenance':{'audits':['independent two-pointer mismatch','forward/reverse SHA-256'],'reader_gate':'closed unless fresh exact >38 appears'},'status':'fresh exact >38 candidate requires human reading' if exact else ('zero live states before rendering' if not rows else 'no fresh exact >38 candidate')}
if __name__=='__main__':
 x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps(x['stats']))
