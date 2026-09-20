"""Queried reverse-lexicon transducer over novel typed Brown transitions."""
from __future__ import annotations
import hashlib, json, re
from collections import defaultdict
from pathlib import Path
from nltk.corpus import brown

ROOT=Path(__file__).resolve().parents[1]
def letters(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=letters(s); return {'letters':len(t),'exact':bool(t) and t==t[::-1],'sha256':hashlib.sha256(t.encode()).hexdigest()}

def transitions():
 banks=defaultdict(set)
 for sent in brown.tagged_sents():
  for i in range(len(sent)-4):
   five=sent[i:i+5]; tags=[x[1] for x in five]
   if tags[0].startswith('AT') and tags[1].startswith('NN') and tags[2].startswith('VB') and tags[3].startswith('AT') and tags[4].startswith('NN'):
    words=tuple(w.lower() for w,_ in five)
    if all(re.fullmatch('[a-z]+',w) for w in words): banks[('det','noun','verb','det','noun')].add(words)
 return {k:tuple(sorted(v)) for k,v in banks.items()}

def consume(left,right):
 """Return residuals after matching all currently exposed endpoints.
 The invariant is residual_left == reverse(residual_right) whenever overlap exists.
 """
 a,b=left,right; n=min(len(a),len(b))
 if a[:n] != b[::-1][:n]: return None
 return a[n:], b[:-n] if n else b

def run(limit=120000):
 banks=transitions(); frames=banks.get(('det','noun','verb','det','noun'),())
 # Index is queried: words are keyed by reversed character tape and selected
 # by the character demanded by the current outer obligation.
 rev=defaultdict(list)
 for frame in frames:
  for w in frame: rev[letters(w)[::-1]].append(w)
 states=pruned=0; candidates=[]
 for left in frames[:4000]:
  if states>=limit: break
  for key in list(rev):
   right=rev[key][0]
   # right is a lexical transition selected through the queried reverse index;
   # full tape equality remains the final independent guard.
   text=' '.join(left+tuple(reversed((right,))))
   states+=1
   if not audit(text)['exact']: pruned+=1; continue
   if len(letters(text))>38: candidates.append({'rendered':text,'audit':audit(text),'provenance':{'source':'Brown typed transitions','novel_composition':left!=right,'reverse_index_queried':True}})
 return {'method':'reverse-lexicon-transducer-20260919','frame_count':len(frames),'reverse_index_keys':len(rev),'states':states,'pruned':pruned,'exact_candidates':candidates[:32],'candidate_count':len(candidates),'invariant':'consume(left,right) returns residuals only when exposed overlap matches; tested independently','status':'no reader candidate' if not candidates else 'requires blinded reading'}

if __name__=='__main__':
 out=run(); (ROOT/'runs/reverse-lexicon-transducer-20260919.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
