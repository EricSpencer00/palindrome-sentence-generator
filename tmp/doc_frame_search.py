import re
from wordfreq import zipf_frequency
from collections import Counter, defaultdict
from nltk.corpus import brown

B = {
 'NOUN': 'doc note cod fatness plan map book chart letter memo report case cause road stone song answer idea work task time name day night man woman people child dog cat bird fish world home house garden river market paper page word words action event matter state place thing food water bread meal key door gate room path signal message'.split(),
 'PRON': 'i we he she they it you'.split(),
 'VERB': 'dissent prevents diet read write mark make take see know find keep hold carry guide guard map chart call name use show tell bring build send set move watch need want like love help'.split(),
 'DET': 'a an the some one our my no'.split(),
 'ADJ': 'fast slow quiet patient gentle young old green bright small kind clear warm fresh calm brave little open early silent red blue wise swift plain honest good dark long short round clean soft great new local wild fine thin white deep high light cool safe true full fair rich sweet heavy'.split(),
 'ADV': 'never ever often always still now then very just not quite'.split(),
 'PREP': 'on in at by near over under after before with from to'.split(),
}
_pos = defaultdict(Counter)
for _sent in brown.tagged_sents(tagset='universal'):
 for _w,_t in _sent:
  _w=_w.lower()
  if _w.isascii() and _w.isalpha(): _pos[_t][_w]+=1
_stop=set('a an the and or but if while as of to in on at by for from with into over under after before is are was were be been being has have had do did does i you he she we they it my our his her their this that these those no not'.split())
def _wide(tag,n):
 xs=[]
 for w,c in _pos[tag].items():
  if len(w)<3 or w in _stop: continue
  z=zipf_frequency(w,'en')
  if z<3.0: continue
  xs.append((w,c,z))
 xs.sort(key=lambda x:(-x[2],-x[1],x[0]))
 return [w for w,_,_ in xs[:n]]
B['NOUN']=list(dict.fromkeys(B['NOUN']+_wide('NOUN',700)))
B['VERB']=list(dict.fromkeys(B['VERB']+_wide('VERB',500)))
B['ADJ']=list(dict.fromkeys(B['ADJ']+_wide('ADJ',400)))
B['ADV']=list(dict.fromkeys(B['ADV']+_wide('ADV',200)))
for k in B: B[k] = list(dict.fromkeys(B[k]))

def tape(w): return ''.join(c for c in w if c.isalpha())

TEMPLATES = [
 ['NOUN','NOUN','PRON','VERB','DET','ADJ','ADV','VERB','DET','NOUN','PRON','VERB','PREP','NOUN'],
 ['NOUN','NOUN','PRON','VERB','DET','ADJ','ADV','ADV','VERB','DET','NOUN','PRON','VERB','PREP','NOUN'],
 ['NOUN','NOUN','PRON','VERB','DET','ADJ','NOUN','ADV','VERB','DET','NOUN','PRON','VERB','PREP','NOUN'],
 ['NOUN','NOUN','PRON','VERB','DET','ADJ','ADV','VERB','DET','ADJ','NOUN','PRON','VERB','PREP','NOUN'],
 ['NOUN','NOUN','PRON','VERB','DET','ADJ','ADV','VERB','DET','NOUN','PRON','VERB','PREP','DET','NOUN'],
 ['NOUN','NOUN','PRON','VERB','DET','ADJ','ADV','VERB','DET','NOUN','PRON','VERB','PREP','ADJ','NOUN'],
]

def search(template, limit=3000000):
 sol, nodes = [], 0
 def dfs(li, ri, left, right, left_words, right_words, score):
  nonlocal nodes
  nodes += 1
  if nodes > limit: return
  rr = right[::-1]; m = min(len(left), len(rr))
  if left[:m] != rr[:m]: return
  if li > ri:
   if left == rr: sol.append((' '.join(left_words + right_words), score))
   return
  if li == ri:
   for word in B[template[li]]:
    if template[li] in {'NOUN','VERB','ADJ','ADV'} and word in left_words + right_words: continue
    full = left + tape(word) + right
    if full == full[::-1]: sol.append((' '.join(left_words + [word] + right_words), score + zipf_frequency(word, 'en')))
   return
  sides = []
  if len(left) <= len(rr): sides.append('L')
  if len(rr) <= len(left): sides.append('R')
  if len(sides) == 2: sides.sort(key=lambda s: len(B[template[li if s == 'L' else ri]]))
  for side in sides:
   kind = template[li if side == 'L' else ri]
   for word in B[kind]:
    if kind in {'NOUN','VERB','ADJ','ADV'} and word in left_words + right_words: continue
    if side == 'L':
     new_left = left + tape(word); m = min(len(new_left), len(rr))
     if new_left[:m] != rr[:m]: continue
     dfs(li + 1, ri, new_left, right, left_words + [word], right_words, score + zipf_frequency(word, 'en'))
    else:
     new_right = tape(word) + right; new_rr = new_right[::-1]; m = min(len(left), len(new_rr))
     if left[:m] != new_rr[:m]: continue
     dfs(li, ri - 1, left, new_right, left_words, [word] + right_words, score + zipf_frequency(word, 'en'))
 dfs(0, len(template) - 1, '', '', [], [], 0.)
 return nodes, sol

if __name__ == '__main__':
 for template in TEMPLATES:
  n, solutions = search(template)
  print('template',len(template),'nodes',n,'solutions',len(solutions))
  for text, score in sorted(solutions, key=lambda x: (len(tape(x[0])), x[1]), reverse=True)[:100]:
   print(len(tape(text)), round(score, 1), text)
