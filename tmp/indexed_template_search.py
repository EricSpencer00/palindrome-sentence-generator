import itertools, re, time
from collections import Counter, defaultdict
from nltk.corpus import brown
from wordfreq import zipf_frequency

POS = defaultdict(Counter)
for sent in brown.tagged_sents(tagset='universal'):
    for word, tag in sent:
        word = word.lower()
        if word.isascii() and word.isalpha(): POS[tag][word] += 1
STOPS = set('a an the and or but if while as of to in on at by for from with into over under after before is are was were be been being has have had do did does i you he she we they it my our his her their this that these those no not'.split())
SHORT = {'a','i','an','he','me','my','we','us','it','am','be','do','go','is','as','at','by','if','in','of','on','or','to','up','no','so','ah','oh','ma','pa'}

def bank(tag, limit=1200):
    rows=[]
    for word,count in POS[tag].items():
        if len(word)<2 and word not in SHORT: continue
        if len(word)==2 and word not in SHORT: continue
        if word in STOPS and word not in SHORT: continue
        z=zipf_frequency(word,'en')
        if z<2.5: continue
        rows.append((word,count,z))
    rows.sort(key=lambda x:(-x[2],-x[1],x[0]))
    return [w for w,_,_ in rows[:limit]]
B={
 'DET': ['a','an','the','some','one','our','my','no','his','her'],
 'PRON': ['i','we','you','he','she','they'],
 'NOUN': bank('NOUN',1300),
 'SUBJ': bank('NOUN',1300),
 'VERB': bank('VERB',1100),
 'ADJ': bank('ADJ',800),
 'ADV': bank('ADV',300),
 'ADP': ['in','on','at','by','near','under','over','after','before','beside','toward','from','with','through','around'],
 'CONJ': ['and','while','but','as'],
 'PROPN': 'ada anna diana eve iris lena lisa maria maya nora noah olivia otto sara uma alan eric emil ella ian ava art leo susan ruth james jane john mary paul peter rose mark anne clara david nina tara tina lori lara sara sam ben tom ray eli'.split(),
 'MOD': bank('ADJ',800)+['a','an','one','two','three','four','five','six','seven','eight','nine','ten'],
}
for k in B: B[k]=list(dict.fromkeys(B[k]))

def tape(s): return ''.join(c for c in s if c.isalpha())

class Index:
    def __init__(self, words):
        self.words=words
        self.by_prefix=defaultdict(list);self.by_suffix=defaultdict(list)
        for w in words:
            self.by_prefix[w[0]].append(w); self.by_suffix[w[-1]].append(w)
    def match_left(self, debt):
        # chosen left word must fill beginning of right-side debt
        if not debt:return self.words
        return [w for w in self.words if debt.startswith(w) or w.startswith(debt)]
    def match_right(self, debt):
        # reverse(chosen right word) fills beginning of left-side debt
        rev=debt[::-1]
        return [w for w in self.words if rev.startswith(w) or w.startswith(rev)]
IDX={k:Index(v) for k,v in B.items()}

def search(T, limit=250000):
    nodes=0;sol=[]
    def dfs(li,ri,left,right,words,score):
        nonlocal nodes
        nodes+=1
        if nodes>limit:return
        rr=right[::-1];m=min(len(left),len(rr))
        if left[:m]!=rr[:m]:return
        if li>ri:
            if left==rr:sol.append((' '.join(words),score))
            return
        sides=[]
        if len(left)<=len(rr):sides.append('L')
        if len(rr)<=len(left):sides.append('R')
        if len(sides)==2:
            sides.sort(key=lambda s:len(B[T[li if s=='L' else ri]]))
        debtL=rr[len(left):] if len(rr)>len(left) else ''
        debtR=left[len(rr):] if len(left)>len(rr) else ''
        for side in sides:
            kind=T[li if side=='L' else ri]
            if side=='L':
                choices=IDX[kind].match_left(debtL)
            else:
                choices=IDX[kind].match_right(debtR)
            for w in choices:
                if kind in {'NOUN','SUBJ','VERB','ADJ','PROPN'} and w in words:continue
                if side=='L':
                    nl=left+tape(w);m=min(len(nl),len(rr))
                    if nl[:m]!=rr[:m]:continue
                    dfs(li+1,ri,nl,right,words+[w],score+zipf_frequency(w,'en'))
                else:
                    nr=tape(w)+right;nrr=nr[::-1];m=min(len(left),len(nrr))
                    if left[:m]!=nrr[:m]:continue
                    dfs(li,ri-1,left,nr,[w]+words,score+zipf_frequency(w,'en'))
    dfs(0,len(T)-1,'','',[],0.0)
    return nodes,sol

CLAUSES=[
 ['PRON','VERB','DET','NOUN'],
 ['DET','NOUN','VERB','DET','NOUN'],
 ['DET','ADJ','NOUN','VERB','DET','NOUN'],
 ['PRON','VERB','DET','ADJ','NOUN'],
 ['DET','NOUN','VERB','ADP','DET','NOUN'],
 ['PRON','VERB','ADP','DET','NOUN'],
 ['DET','NOUN','VERB','ADV','ADJ'],
 ['DET','SUBJ','VERB','MOD','NOUN'],
 ['DET','SUBJ','VERB','MOD','NOUN','DET','SUBJ','VERB','PROPN'],
 ['DET','SUBJ','VERB','PROPN'],
 ['PRON','VERB','DET','ADJ','NOUN','ADP','DET','NOUN'],
 ['DET','ADJ','NOUN','VERB','ADP','DET','NOUN'],
]

if __name__=='__main__':
    hits=[];start=time.time()
    for arity in (2,3):
      for choices in itertools.product(range(len(CLAUSES)),repeat=arity):
        T=[]
        for i in choices:T+=CLAUSES[i]
        if len(T)>16:continue
        n,s=search(T)
        for text,score in s:
            hits.append((len(tape(text)),score,text,choices,n))
        if s:print('hit',choices,'nodes',n,'count',len(s),flush=True)
    print('done',len(hits),'sec',time.time()-start)
    for row in sorted(hits,reverse=True)[:100]:print(row)
