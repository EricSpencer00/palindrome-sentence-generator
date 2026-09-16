"""Character-LM ranking under a tiny clause grammar (lane 1)."""
from pathlib import Path
import hashlib, json, sys
sys.path.insert(0, str(Path(__file__).parents[1]))
from llm_palindrome.wordpair_graph import tape

# A local, transparent character bigram model; it ranks legal words only.
CORPUS = "the river carries light and patient hands repair the old gate"
BIGRAM = {(a,b): CORPUS.count(a+b) for a,b in zip(CORPUS, CORPUS[1:])}
SLOTS = [("subject", ["the keeper", "quiet workers", "a patient child"]),
         ("verb", ["marks", "carries", "opens"]),
         ("object", ["the river trail", "a small lantern", "old wooden gates"]),
         ("ending", ["at dusk", "before dawn", "in silence"])]

def score(text):
    chars=' '+text.lower(); return sum(BIGRAM.get((a,b),0) for a,b in zip(chars,chars[1:]))
def decode():
    text=''; trace=[]
    for state, choices in SLOTS:
        ranked=sorted(choices,key=lambda x:(score(text+' '+x),x),reverse=True)
        choice=ranked[0]; text += (' ' if text else '') + choice
        trace.append({'state':state,'legal':choices,'chosen':choice,'char_score':score(text)})
    return text+'.',trace
def pointer(a,b):
    i,j=0,len(b)-1
    while i<len(a) and j>=0 and a[i]==b[j]: i+=1; j-=1
    return {'equal':i==len(a) and j<0,'matched':i,'left_remaining':a[i:],'right_remaining':b[:j+1]}
def main():
    left,trace=decode(); right,_=decode(); rendered=left+' '+right
    lt,rt=tape(left),tape(right); p=pointer(lt,rt)
    hashes={'left':hashlib.sha256(lt.encode()).hexdigest(),'right':hashlib.sha256(rt.encode()).hexdigest(),'rendered':hashlib.sha256(tape(rendered).encode()).hexdigest()}
    out={'method':'character_lm_grammar_constrained_decode_v1','candidates':[{'text':rendered,'letters':len(tape(rendered)),'exact':p['equal'],'admitted':p['equal'] and len(tape(rendered))>100,'pointer_audit':p,'hash_audit':hashes}],
      'frontier':{'left_clause':left,'right_clause':right,'slots':trace},'provenance':{'lm':'local character bigram counts from inline corpus','grammar':'subject-verb-object-ending complete-clause slots','constraint':'LM ranks legal choices; it never relaxes grammar'},'novelty_preflight':{'tape_sha256':hashes['rendered'],'action':'compare digest with prior artifacts'},'next_repair':'Train character counts on an expanded ordinary-prose corpus, then add a reverse-residual state so decoding can target the first unmatched character.'}
    path=Path(__file__).parents[1]/'runs/char-lm-grammar-2026-09-16.json'; path.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({'output':str(path),'letters':len(tape(rendered)),'exact':p['equal']}))
if __name__=='__main__': main()
