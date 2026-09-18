import itertools, re, time
from wordfreq import zipf_frequency

B = {
    "DET": "a an the some one our my no this that his her".split(),
    "PRON": "i we you he she they it".split(),
    "ADJ": "quiet patient gentle young old green bright small kind clear warm fresh calm brave little open early silent red blue wise swift plain honest good dark long short round clean soft great new local wild fine thin white deep high light cool safe true full fair rich sweet heavy distant narrow broad careful clever".split(),
    "NOUN": "aide artist baker carpenter captain child clerk doctor driver farmer friend gardener guide keeper man master men nurse pilot poet porter reader sailor scholar singer smith soldier student teacher woman writer agent angel author actor elder monk mother father sister brother crew team people girls boys birds dogs cats river garden harbor market house school letter map book bread stone lantern window bridge boat road field story song answer image apple orange ocean island animal horse bird cloud rain water flower tree door gate room path note meal chair table paper signal chart message parcel village castle candle camera idea area data era dawn deer desk dream drum flag gate glass island key lamp line list meal memo memos note page pen poem pond rope sail seal ship sign table tent tool toy train tree vase wall wheel yard action mission number plan report task truth word words".split(),
    "SUBJ": "aide artist baker carpenter captain child clerk doctor driver farmer friend gardener guide keeper man master men nurse pilot poet porter reader sailor scholar singer smith soldier student teacher woman writer agent angel author actor elder monk mother father sister brother crew team people girls boys birds dogs cats diana maria nora lena anna iris otto ada eve noah olivia uma sara maya".split(),
    "VERB": "aids asks bakes calls carries charts checks cleans cooks cuts draws finds folds gives guides guards helps holds inspires keeps leads likes maps marks mends meets moves needs opens packs paints plants reads repairs rides rips sends sets shares shows sings speaks spins takes teaches tells trains uses visits walks watches waters writes inspire review reviews sees knows loves makes brings builds saves names follows crosses turns returns notices wants seeks finds holds opens shuts closes".split(),
    "ADP": "in on at by near under over after before beside toward from with through around beyond".split(),
    "PROPN": "ada anna diana eve iris lena lisa maria maya nora noah olivia otto sara uma alan eric emil ella ian ava art leo susan ruth james jane john mary paul peter rose mark anne clara david nina nina tara tina lisa lori lara sara sam ben tom ray eli".split(),
    "MOD": "a an one two three four five six seven eight nine ten old new red blue green small large quiet fresh warm cool kind clear young wise bright dark long short soft hard fine fair true good last first many some more each odd even".split(),
}
for k in B: B[k] = list(dict.fromkeys(B[k]))

def tape(w): return ''.join(c for c in w if c.isalpha())

CLAUSES = [
    ["DET", "NOUN", "VERB", "DET", "NOUN"],
    ["DET", "SUBJ", "VERB", "MOD", "NOUN"],
    ["DET", "SUBJ", "VERB", "PROPN"],
    ["DET", "SUBJ", "VERB", "MOD", "NOUN", "DET", "SUBJ", "VERB", "PROPN"],
    ["DET", "ADJ", "NOUN", "VERB", "DET", "NOUN"],
    ["DET", "NOUN", "VERB", "DET", "ADJ", "NOUN"],
    ["DET", "ADJ", "NOUN", "VERB", "DET", "ADJ", "NOUN"],
    ["PRON", "VERB", "DET", "NOUN"],
    ["PRON", "VERB", "DET", "ADJ", "NOUN"],
    ["PRON", "VERB", "DET", "NOUN", "ADP", "DET", "NOUN"],
    ["DET", "NOUN", "VERB", "ADP", "DET", "NOUN"],
    ["DET", "ADJ", "NOUN", "VERB", "ADP", "DET", "NOUN"],
    ["DET", "NOUN", "VERB", "DET", "NOUN", "ADP", "DET", "NOUN"],
]

def search(template, limit=300000):
    nodes, sol = 0, []
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
            kind = template[li]
            for word in B[kind]:
                if kind in {'NOUN','VERB','ADJ','PROPN'} and word in left_words + right_words: continue
                full = left + tape(word) + right
                if full == full[::-1]: sol.append((' '.join(left_words + [word] + right_words), score + zipf_frequency(word, 'en')))
            return
        sides = []
        if len(left) <= len(rr): sides.append('L')
        if len(rr) <= len(left): sides.append('R')
        if len(sides) == 2:
            sides.sort(key=lambda x: len(B[template[li if x == 'L' else ri]]))
        for side in sides:
            kind = template[li if side == 'L' else ri]
            for word in B[kind]:
                # ordinary prose can reuse function words, but not content lexemes
                if kind in {'NOUN','VERB','ADJ','PROPN'} and word in left_words + right_words: continue
                if side == 'L':
                    nl = left + tape(word); m = min(len(nl), len(rr))
                    if nl[:m] != rr[:m]: continue
                    dfs(li+1, ri, nl, right, left_words+[word], right_words, score+zipf_frequency(word,'en'))
                else:
                    nr = tape(word) + right; nrr = nr[::-1]; m = min(len(left), len(nrr))
                    if left[:m] != nrr[:m]: continue
                    dfs(li, ri-1, left, nr, left_words, [word]+right_words, score+zipf_frequency(word,'en'))
    dfs(0, len(template)-1, '', '', [], [], 0.)
    return nodes, sol

if __name__ == '__main__':
    hits=[]; start=time.time()
    for arity in (2, 3):
      for choices in itertools.product(range(len(CLAUSES)), repeat=arity):
            t=[]
            for choice in choices: t.extend(CLAUSES[choice])
            n,s=search(t, 600000)
            for text,sc in s:
                hits.append((len(tape(text)),sc,text,*choices,n))
            if s: print('hit',choices,n,len(s),s[:3])
    print('total hits',len(hits),'seconds',time.time()-start)
    for row in sorted(hits,reverse=True)[:100]: print(row)
