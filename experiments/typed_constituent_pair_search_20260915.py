"""Bounded typed constituent-pair experiment.

Both sides are independently generated ordinary phrase templates.  Pairing is
by normalized tape reversal, so word boundaries may cross; neither side is a
trie segmentation of the other.  All acceptance gates are mechanical.
"""
from __future__ import annotations
import argparse,json,random,sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from llm_palindrome.validator import normalize,is_palindrome

NOUNS='artist author baker child doctor farmer friend gardener neighbor nurse parent sailor teacher worker writer dog cat bird river house garden letter story music idea plan map note book school town world day night rain wind fire water light song'.split()
VERBS='admired answered baked built called carried changed cleaned closed found fixed followed helped held kept learned liked listened loved made marked noticed opened painted planned read rescued saved saw sent showed studied taught thanked told used watched wrote'.split()
ADJS='old young kind quiet brave calm small great new bright dark clear open warm cold long short good wise true ready safe gentle patient useful'.split()
DETS='a the this that my our'.split()
PREPS='by for from in near over under with without'.split()

def phrases(limit=300000):
  # Complete constituents: NP, transitive clause, copular clause, and PP adjunct.
  nps=[f'{d} {n}' for d in DETS for n in NOUNS]
  npa=[f'{d} {a} {n}' for d in 'a the this that'.split() for a in ADJS for n in NOUNS]
  out=set(nps+npa)
  for s in nps:
    for v in VERBS:
      for o in nps+npa:
        out.add(f'{s} {v} {o}')
        if len(out)>=limit: return sorted(out)
    for a in ADJS: out.add(f'{s} is {a}')
  # PP constituents and clauses with a PP tail.
  for p in PREPS:
    for x in nps: out.add(f'{p} {x}')
    for s in nps:
      for v in VERBS:
        for x in nps:
          out.add(f'{s} {v} {x} {p} {nps[(len(s)+len(v)+len(x))%len(nps)]}')
          if len(out)>=limit: return sorted(out)
  return sorted(out)

def boundary_cross(a,b):
  # Require reversal to not preserve every word boundary position.
  ta,tb=normalize(a),normalize(b)
  left_bounds={len(normalize(' '.join(a.split()[:i]))) for i in range(1,len(a.split()))}
  right_bounds={len(tb)-len(normalize(' '.join(b.split()[:i]))) for i in range(1,len(b.split()))}
  return not (left_bounds & right_bounds)

def run(max_phrases=300000,seed=0,min_letters=20,max_letters=70):
  allp=phrases(max_phrases); rng=random.Random(seed); rng.shuffle(allp)
  allp=[p for p in allp if min_letters<=len(normalize(p))<=max_letters][:max_phrases]
  idx=defaultdict(list)
  for p in allp: idx[normalize(p)].append(p)
  rows=[]; rejected=defaultdict(int)
  for left in allp:
    tape=normalize(left)
    for right in idx.get(tape[::-1],[]):
      words=(left+' '+right).split()
      if left==right: rejected['identical']+=1; continue
      if len(words)<6: rejected['short_phrase']+=1; continue
      if sum(len(w)<=2 for w in words)>2: rejected['short_words']+=1; continue
      if len(set(words))<len(words): rejected['repeated_word']+=1; continue
      if not boundary_cross(left,right): rejected['aligned_boundaries']+=1; continue
      text=left+' '+right
      if not is_palindrome(text): rejected['validator']+=1; continue
      rows.append({'text':text,'left':left,'right':right,'letters':len(normalize(text)),'boundary_crossing':True,'exact_palindrome':True})
  rows.sort(key=lambda r:(-r['letters'],r['text']))
  return {'status':'no_readability_claim','config':{'max_phrases':max_phrases,'seed':seed,'min_letters':min_letters,'max_letters':max_letters},'generated_phrases':len(allp),'candidate_count':len(rows),'rejected':dict(rejected),'candidates':rows[:200]}

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--max-phrases',type=int,default=300000);p.add_argument('--seed',type=int,default=0);a=p.parse_args();r=run(a.max_phrases,a.seed);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k] for k in ('generated_phrases','candidate_count')},indent=2))
if __name__=='__main__': main()
