import itertools, random, re, time

B = {
    'DET': 'a an the some one our my no his her'.split(),
    'SUBJ': 'aide artist baker carpenter captain child clerk doctor driver farmer friend gardener guide keeper man master men nurse pilot poet porter reader sailor scholar singer smith soldier student teacher woman writer agent angel author actor elder monk mother father sister brother crew team people girls boys birds dogs cats diana maria nora lena anna iris otto ada eve noah olivia uma sara maya'.split(),
    'VERB': 'aids asks bakes calls carries charts checks cleans cooks cuts draws finds folds gives guides guards helps holds inspires keeps leads likes maps marks mends meets moves needs opens packs paints plants reads repairs rides rips sends sets shares shows sings speaks spins takes teaches tells trains uses visits walks watches waters writes inspires reviews sees knows loves makes brings builds saves names follows crosses turns returns notices wants seeks'.split(),
    'ADJ': 'quiet patient gentle young old green bright small kind clear warm fresh calm brave little open early silent red blue wise swift plain honest good dark long short round clean soft great new local wild fine thin white deep high light cool safe true full fair rich sweet heavy distant narrow broad careful clever'.split(),
    'NOUN': 'aide artist baker carpenter captain child clerk doctor driver farmer friend gardener guide keeper man master men nurse pilot poet porter reader sailor scholar singer smith soldier student teacher woman writer agent angel author actor elder monk mother father sister brother crew team people girls boys birds dogs cats river garden harbor market house school letter map book bread stone lantern window bridge boat road field story song answer image apple orange ocean island animal horse bird cloud rain water flower tree door gate room path note meal chair table paper signal chart message parcel village castle candle camera idea area data era dawn deer desk dream drum flag gate glass island key lamp line list meal memo memos note page pen poem pond rope sail seal ship sign table tent tool toy train tree vase wall wheel yard action mission number plan report task truth word words'.split(),
    'ADP': 'in on at by near under over after before beside toward from with through around beyond'.split(),
}
for k in B:B[k]=list(dict.fromkeys(B[k]))
def tape(s):return ''.join(c for c in s if c.isalpha())
TEMPLATES=[
 ['DET','SUBJ','VERB','DET','NOUN'],
 ['DET','SUBJ','VERB','DET','ADJ','NOUN'],
 ['DET','SUBJ','VERB','DET','ADJ','NOUN','ADP','DET','NOUN'],
 ['SUBJ','VERB','DET','NOUN','ADP','DET','NOUN'],
 ['DET','ADJ','NOUN','VERB','DET','ADJ','NOUN'],
 ['DET','SUBJ','VERB','ADP','DET','NOUN'],
]

def parse(target, template, max_nodes=10000):
    out=[];nodes=0; memo={}
    def go(i,j,words):
        nonlocal nodes
        nodes+=1
        if nodes>max_nodes:return
        key=(i,j,tuple(words))
        if i==len(template):
            if j==len(target):out.append(words)
            return
        kind=template[i]
        # word must match target prefix at position j; no need try others
        for w in B[kind]:
            tw=tape(w)
            if target.startswith(tw,j):
                if kind in {'SUBJ','VERB','NOUN','ADJ'} and w in words:continue
                go(i+1,j+len(tw),words+[w])
    go(0,0,[]);return out

def main():
    rng=random.Random(7);hits=[];start=time.time();seen=set()
    # Generate ordinary left clauses from each template, with a large but finite
    # product and deterministic random lexeme choices.
    for ti,T in enumerate(TEMPLATES):
        for n in range(50000):
            words=[];used=set()
            for k in T:
                choices=B[k]
                w=rng.choice(choices)
                if k in {'SUBJ','VERB','NOUN','ADJ'}:
                    tries=0
                    while w in used and tries<20:w=rng.choice(choices);tries+=1
                    used.add(w)
                words.append(w)
            text=' '.join(words); left=tape(text); target=left[::-1]
            for rt in TEMPLATES:
                rows=parse(target,rt)
                for right in rows:
                    joined=text+'; '+' '.join(right)+'.'; joined_tape=tape(joined)
                    if joined_tape==joined_tape[::-1] and joined not in seen:
                        seen.add(joined);hits.append((len(joined_tape),joined,ti,rt))
            if len(hits)>100:break
    print('hits',len(hits),'secs',time.time()-start)
    for row in sorted(hits,reverse=True)[:100]:print(row)
if __name__=='__main__':main()
