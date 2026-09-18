import itertools, re
from wordfreq import zipf_frequency

B = {
 'PROPN': 'satan madam ada anna diana eve iris lena lisa maria maya nora noah olivia otto sara uma alan eric emil ella ian ava art leo ruth'.split(),
 'VERB': 'oscillate deliver repair repay repaid reward draw carry guide guard mark name read write watch water help make take give see use are was live emit turn'.split(),
 'DET': 'a an the my our'.split(),
 'ADJ': 'metallic stressed reviled quiet patient gentle young old green bright small kind clear warm fresh calm brave little open early silent red blue wise swift plain honest good dark long short round clean soft great new local wild fine thin white deep high light cool safe true full fair rich sweet heavy'.split(),
 'NOUN': 'sonatas desserts drawer reward diaper repaid nametag gateman parts strap straw warts regal lager denim mined flow wolf live evil time emit saw was step pets stop pots spot tops ten net star rats stun nuts snap pans spit tips rail liar loop pool room moor raw war rat tar tab bat ton not won now aide memos diana plan canal map road boat book letter name title answer garden stone song meal chart signal'.split(),
 'ADP': 'in on at by near over under after before with'.split(),
}
for k in B: B[k] = list(dict.fromkeys(B[k]))

def tape(w): return ''.join(c for c in w if c.isalpha())

TEMPLATES = [
 ['PROPN','VERB','DET','ADJ','NOUN'],
 ['PROPN','VERB','DET','ADJ','ADJ','NOUN'],
 ['PROPN','VERB','DET','ADJ','NOUN','ADP','DET','NOUN'],
 ['DET','ADJ','NOUN','VERB','DET','ADJ','NOUN'],
]

def search(T):
 sol = []
 def dfs(li, ri, left, right, words, score):
  rr = right[::-1]; m = min(len(left), len(rr))
  if left[:m] != rr[:m]: return
  if li > ri:
   if left == rr: sol.append((' '.join(words), score))
   return
  if li == ri:
   kind = T[li]
   for word in B[kind]:
    if kind not in {'DET','ADP'} and word in words: continue
    full = left + tape(word) + right
    if full == full[::-1]: sol.append((' '.join(words + [word]), score + zipf_frequency(word, 'en')))
   return
  sides = []
  if len(left) <= len(rr): sides.append('L')
  if len(rr) <= len(left): sides.append('R')
  if len(sides) == 2: sides.sort(key=lambda s: len(B[T[li if s == 'L' else ri]]))
  for side in sides:
   kind = T[li if side == 'L' else ri]
   for word in B[kind]:
    if kind not in {'DET','ADP'} and word in words: continue
    if side == 'L':
     nl = left + tape(word); m = min(len(nl), len(rr))
     if nl[:m] != rr[:m]: continue
     dfs(li + 1, ri, nl, right, words + [word], score + zipf_frequency(word, 'en'))
    else:
     nr = tape(word) + right; nrr = nr[::-1]; m = min(len(left), len(nrr))
     if left[:m] != nrr[:m]: continue
     dfs(li, ri - 1, left, nr, [word] + words, score + zipf_frequency(word, 'en'))
 dfs(0, len(T) - 1, '', '', [], 0)
 return sol

if __name__ == '__main__':
 for T in TEMPLATES:
  solutions = search(T)
  print(T, len(solutions))
  for text, score in sorted(solutions, key=lambda row: (len(tape(row[0])), row[1]), reverse=True)[:100]:
   print(len(tape(text)), round(score, 1), text)
