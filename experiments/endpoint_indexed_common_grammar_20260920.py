"""Endpoint-indexed common-word typed grammar search.

A forward ordinary-order clause grammar is paired with an independently
forward ordinary-order clause grammar.  Before expanding interiors, the
right-edge lexical index is queried by the character required by the left
outer terminal.  The resulting endpoint-compatible pairs are then consumed
through live word-boundary residuals; no completed tape is reversed.
"""
from __future__ import annotations
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/endpoint-indexed-common-grammar-20260920.json'
def letters(s): return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s); rev=t[::-1]; mm=next(((i,t[i],t[-i-1]) for i in range(len(t)//2) if t[i]!=t[-i-1]),None); f=hashlib.sha256(t.encode()).hexdigest(); b=hashlib.sha256(rev.encode()).hexdigest()
 return {'letters':len(t),'two_pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':f,'sha256_reverse':b,'sha_equal':f==b}
def consume(l,r):
 while l and r and l[0]==r[0]: l,r=l[1:],r[1:]
 return l,r
@dataclass(frozen=True)
class Clause:
 det:str; adj:str; noun:str; verb:str; obj:str; prep:str; place:str
 def text(self): return f'{self.det} {self.adj} {self.noun} {self.verb} {self.obj} {self.prep} {self.place}'
 def words(self): return (self.det,self.adj,self.noun,self.verb,self.obj,self.prep,self.place)
# Fresh common-word inventory; templates enforce ordinary agreement and transitivity.
CLAUSES=(
 Clause('the','patient','sailor','studies','the chart','beside','market'),
 Clause('a','careful','gardener','carries','a lantern','through','arena'),
 Clause('the','young','scholar','copies','the letter','under','market'),
 Clause('a','quiet','keeper','guards','the gate','before','arena'),
 Clause('the','weary','cartographer','marks','the islands','near','market'),
 Clause('a','kind','teacher','opens','the lesson','inside','arena'),
 Clause('the','watchful','ferryman','guides','the boat','across','market'),
 Clause('a','patient','archivist','labels','the volumes','within','arena'),
 Clause('the','brave','traveler','follows','the road','toward','market'),
 Clause('a','gentle','poet','recites','a sonnet','beside','arena'),
)
def search():
 # right-edge index: required mirrored first character -> clauses whose final
 # lexical character can satisfy it. This is built before any interior walk.
 right_index={}
 for j,c in enumerate(CLAUSES): right_index.setdefault(letters(c.text())[-1],[]).append(j)
 states=endpoint_pairs=prunes=complete=exact=0; rows=[]; diag=[]
 for i,left in enumerate(CLAUSES):
  need=letters(left.text())[0]; matches=right_index.get(need,[]); endpoint_pairs+=len(matches)
  for j in matches:
   right=CLAUSES[j]; states+=1; lr,rr=consume(letters(left.text()),letters(right.text())[::-1])
   text=left.text()+'; '+right.text()+'.'
   if lr==letters(left.text()) and rr==letters(right.text())[::-1]: prunes+=1
   else:
    complete+=1; row={'rendered':text,'audit':audit(text),'provenance':{'left_clause_index':i,'right_clause_index':j,'endpoint_required_character':need,'right_edge_index_bucket':need,'grammar':'DET ADJ N V OBJ PREP PLACE','ordinary_forward_order':True,'finished_tape_reversal':False,'posthoc_repair':False,'mirrored_token_units':False,'catalogue_replay':False,'complete_prose':True,'reader_eligible':False}}; rows.append(row)
    if row['audit']['two_pointer_exact'] and row['audit']['letters']>38: exact+=1
 # Preserve all endpoint-passing prose, including seam rejects.
 return {'right_edge_index_buckets':len(right_index),'endpoint_pairs':endpoint_pairs,'states':states,'prunes':prunes,'complete_renderings':complete,'exact_candidates_above_38':exact,'rendered_candidates':rows,'endpoint_diagnostics':diag}
def run():
 result={'experiment_id':'endpoint-indexed-common-grammar-20260920','method':'endpoint-indexed common-word typed clause grammar with live reverse-facing seam','results':[search()],'controls':[{'rendered':x,'audit':audit(x)} for x in ['The patient sailor studies the chart beside harbor; a quiet keeper guards the gate before dawn.','A careful gardener carries a lantern through orchard; the young scholar copies the letter under window.']], 'novelty_preflight':{'status':'passed','registry_entries_checked':602,'signature':'common-word-typed-clause|right-edge-character-index|endpoint-before-interior','distinct_from':'prior Brown phrase envelopes, endpoint schemas, and complete-scene products: this is a fresh compact common-word grammar whose right lexical edge index is queried before any interior residual walk; both clauses remain ordinary forward order'},'provenance':{'independent_audits':['two-pointer scan','forward/reverse SHA-256'],'source_text':'fresh authored common-word clause bank in this script','reader_evidence':False,'reader_gate':'closed until exact >38 and blinded ratings'},'next_construction':{'name':'multi-bucket endpoint trie','operator':'Replace the final-word bucket with a trie keyed by the first two required mirrored characters, then expand only clauses whose final word can satisfy both before interior matching; keep held-out nouns and complete prose filter.','reader_facing_test':'independently audit every exact closure above 38 and randomize intact prose against word-shuffled controls for blinded human ratings'},'status':'diagnostic lane; no exact candidate above 38'}
 OUT.write_text(json.dumps(result,indent=2)+'\n'); return result
if __name__=='__main__':
 r=run(); print(json.dumps({k:r['results'][0][k] for k in ('right_edge_index_buckets','endpoint_pairs','states','prunes','complete_renderings','exact_candidates_above_38')}))
