import itertools, random, re, time
from llm_palindrome.search import WordTries
from llm_palindrome.centerout import centerout_search
from llm_palindrome.bigram import BigramModel
from llm_palindrome.scoring import CoherentScorer

SUBJ = 'the baker the carpenter the teacher the gardener the sailor the nurse the pilot the poet the farmer the artist the keeper a quiet sailor a patient baker a young teacher Mara Nora Lena Diana Iris'.split('|')
# Replace the above intentionally as phrases, not single words.
SUBJ = ['the baker','the carpenter','the teacher','the gardener','the sailor','the nurse','the pilot','the poet','the farmer','the artist','the keeper','a quiet sailor','a patient baker','a young teacher','Mara','Nora','Lena','Diana','Iris']
VERB = ['bakes','carries','guides','marks','keeps','reads','finds','holds','mends','maps','plants','repairs','writes','watches','waters','opens','packs','paints','guards','helps']
OBJ = ['fresh bread','old maps','a green lantern','the small garden','quiet letters','a blue boat','the warm meal','a clear chart','new seeds','the red gate','a kind note','the long road','a bright signal','the little room','a stone bridge']
PP = ['at dawn','after rain','near shore','by the quay','under the tree','beside the gate','in the garden','before dusk','with care','for the crew']

def clauses():
    out=[]
    for s,v,o,p in itertools.product(SUBJ,VERB,OBJ,PP):
        # Keep ordinary clause shapes and avoid same-word collisions.
        text=f'{s} {v} {o} {p}'
        if len(set(re.findall('[a-z]+',text.lower()))) < len(re.findall('[a-z]+',text.lower()))*.72: continue
        out.append(text)
    return list(dict.fromkeys(out))

class Zero:
    def word_delta(self,*args,**kwargs): return 0.0

def main():
    units=clauses(); print('units',len(units))
    # phrase units are individual clauses; the outer renderer will insert
    # semicolons, so every returned sequence is an intact clause chain.
    tries=WordTries(units)
    rows=[]
    for seed in range(16):
        words=centerout_search(tries,Zero(),min_letters=50,beam_width=300,max_steps=8,candidate_limit=500,seed=seed,diversity=1.0,max_overhang=36,maximize='letters')
        if words:
            text='; '.join(words)+'.'; tape=''.join(c for c in text.lower() if c.isalpha())
            if tape==tape[::-1]: rows.append((len(tape),text))
    print('exact',len(rows))
    for row in sorted(set(rows),reverse=True)[:30]:print(row)

if __name__=='__main__':main()
