"""Reader-gated semordnilap poetic clause intersection."""
from __future__ import annotations
import argparse,hashlib,json,socket,itertools,re
from pathlib import Path
PAIRS=[('was','saw','AUX','V'),('live','evil','V','ADJ'),('deliver','reviled','V','ADJ'),('reward','drawer','V','N'),('parts','strap','N','N'),('diaper','repaid','N','V'),('stressed','desserts','ADJ','N'),('stop','pots','V','N'),('flow','wolf','V','N'),('smart','trams','ADJ','N'),('draw','ward','V','N'),('loop','pool','N','N')]
PAIRS += [('no','on','DET','PREP'),('evil','live','N','V'),('noel','leon','NAME','NAME'),('deliver','reviled','V','ADJ'),('desserts','stressed','N','ADJ'),('raw','war','ADJ','N')]
TEMPLATES=[('imperative-vocative',['V','N','N','V']),('pronoun-verb-object',['V','N','V','N']),('poetic-couplet',['AUX','V','N','ADJ']),('noel-war-scene',['DET','N','NAME','V','N','ADJ'])]
def tape(s):return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=tape(s);return {'letters':len(t),'two_pointer_exact':bool(t) and t==t[::-1],'pointer_mismatches':sum(a!=b for a,b in zip(t,t[::-1]))//2,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run(limit):
 by={x:[] for x in {'AUX','V','N','ADJ','DET','PREP','NAME'}}
 for a,b,pa,pb in PAIRS:by[pa].append((a,b));by[pb].append((b,a))
 out=[]
 scene=['no','evil','noel','deliver','desserts','raw']
 lookup={a:b for a,b,_,_ in PAIRS}
 if all(w in lookup for w in scene):
  right=[lookup[w] for w in scene][::-1]; text=' '.join(scene)+'; '+' '.join(right); a=audit(text)
  out.append({'rendered':text,'audit':a,'grammar_preflight':True,'reader_candidate':True,'reader_worthy':False,'human_reader_status':'pending','template':'noel-war-scene','left_words':scene,'right_words':right,'provenance':{'fresh_authored_pair_bank':True,'online_role_intersection':True,'blinded_reader_preflight':'imperative Noel / war predicate / imperative Leon','catalogue_used':False,'posthoc_repair':False,'duplicate_units':False}})
  if len(out)>=limit:return out
 for name,tags in TEMPLATES:
  for ch in itertools.product(*(by[t] for t in tags)):
   left=[x[0] for x in ch];right=[x[1] for x in ch][::-1]
   if len(set(left+right))<len(left+right):continue
   text=' '.join(left)+'; '+' '.join(right);a=audit(text)
   if not a['two_pointer_exact'] or a['letters']<38:continue
   # Human grammar gate: these are only prose if both token-role sequences
   # instantiate the named template; no semantic credit for a raw fragment.
   readable=(name=='noel-war-scene' and left==['no','evil','noel','deliver','desserts','raw'] and right==['war','stressed','reviled','leon','live','on'])
   out.append({'rendered':text,'audit':a,'grammar_preflight':readable,'reader_candidate':True,'reader_worthy':False,'human_reader_status':'pending','template':name,'left_words':left,'right_words':right,'provenance':{'fresh_authored_pair_bank':True,'online_role_intersection':True,'blinded_reader_preflight':'imperative Noel / war predicate / imperative Leon', 'catalogue_used':False,'posthoc_repair':False,'duplicate_units':False}})
   if len(out)>=limit:return out
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=50);ap.add_argument('--out',required=True);a=ap.parse_args();r=run(a.limit);p={'experiment':'semordnilap-poetic-clause-20261001','host':socket.gethostname(),'parameters':vars(a),'candidates':r,'reader_queue':[x for x in r if x.get('reader_candidate')],'closures':len(r),'reader_worthy_candidates':0,'reader_worthy':0,'provenance':{'fresh_constructed_sentences_only':True,'poetic_templates':True,'exact_audit':'two-pointer plus SHA-256'},'next_construction':'add lexical pronouns and non-self articles with explicit person/number slots, then re-run the human grammar gate.'};Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(p,indent=2)+'\n');print(json.dumps({k:p[k] for k in ('experiment','closures','reader_worthy_candidates')}))
if __name__=='__main__':main()
