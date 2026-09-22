"""Bounded copular/inverted poetic topology; live bilateral character CSP."""
import json,sys,hashlib,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from experiments.bilateral_grammar_csp_20260920 import bilateral_grammar_csp
from experiments.forward_lexicalized_grammar_20260920 import Word, letters
# Unlike prior transitive SVO lanes, this grammar permits nominal predicates and
# adjective complements, giving ordinary descriptive/proverbial English paths.
G={"S":(("CLAUSE","CLAUSE"),),"CLAUSE":(("NP","COP","ADJ","NP"),),
   "NP":(("PROPN",),("DET","N"),("DET","ADJ","N"),("N",)),
}
lex=[]
for w,p in [('a','DET'),('the','DET'),('an','DET'),('old','ADJ'),('wise','ADJ'),('kind','ADJ'),('red','ADJ'),('calm','ADJ'),('brave','ADJ'),('dear','ADJ'),('is','COP'),('was','COP'),('am','COP'),('diana','PROPN'),('leon','PROPN'),('noel','PROPN'),('sara','PROPN'),('man','N'),('men','N'),('maid','N'),('king','N'),('queen','N'),('poet','N'),('ranger','N'),('sage','N'),('rose','N'),('star','N'),('moon','N'),('river','N')]: lex.append(Word(w,p))

def audit(s):
 t=letters(s); return {'normalized':t,'letters':len(t),'exact':bool(t) and t==t[::-1], 'sha256':hashlib.sha256(t.encode()).hexdigest()}
def run():
 r=bilateral_grammar_csp(tuple(lex),max_words=14,max_nodes=500000,grammar=G)
 out=[]
 for p in r['paths']:
  text=p['rendered']; a=audit(text)
  words=re.findall('[a-z]+',text.lower()); nested=[]
  for i in range(len(words)):
   for j in range(i+2,len(words)+1):
    x=letters(''.join(words[i:j]))
    if x==x[::-1]: nested.append((i,j))
  p['audit_independent']=a; p['nested_word_spans']=nested; p['mechanically_admitted']=a['exact'] and len(a['normalized'])>38 and not nested and len(set(words))==len(words)
  if p['mechanically_admitted']: out.append(p)
 r['paths']=out; r['experiment_id']='inverted-copular-bilateral-20260920'; r['stats']['admitted_non_nested']=len(out); r['novelty_preflight']={'duplicate':False,'topology':'NP COP ADJ NP / reverse-edge bilateral','not_svo':True,'catalogue_text':False,'post_hoc_repair':False}; r['failure_and_next_construction']={'failure':'no non-nested reader candidate' if not out else 'inspect candidates','next':'add short copular contractions and coordinated predicate complements, preserving live residual equation'}
 return r
if __name__=='__main__':
 o=run(); (ROOT/'runs/inverted-copular-bilateral-20260920.json').write_text(json.dumps(o,indent=2)+'\n'); print(json.dumps(o['stats'],sort_keys=True)); [print(x['rendered']) for x in o['paths'][:5]]
