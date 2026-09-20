"""Broad Brown headword boundary WFSA with immediate mirrored characters."""
from __future__ import annotations
import argparse,hashlib,json,re,socket,itertools
from pathlib import Path
DATA=Path(__file__).parents[1]/'data/brown_pcfg_bank_20260920.json'
if not DATA.exists():DATA=Path('/tmp/brown_pcfg_bank_20260920.json')
def vocab():
 d=json.loads(DATA.read_text());return sorted({x['word'] for vs in d['lexicon'].values() for x in vs if x['word'].isalpha() and 2<=len(x['word'])<=8})
def tape(s):return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=tape(s);return {'letters':len(t),'two_pointer_exact':bool(t) and t==t[::-1],'pointer_mismatches':sum(a!=b for a,b in zip(t,t[::-1]))//2,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def segment(stream,bank,used):
 dp={0:[]}
 for i in range(len(stream)):
  if i not in dp:continue
  for j in range(i+2,min(len(stream),i+8)+1):
   w=stream[i:j]
   if w in bank and w not in used and w not in dp[i]:dp.setdefault(j,dp[i]+[w])
 return dp.get(len(stream))
def run(min_letters,limit):
 bank=vocab()[:600]; rows=[]
 for a,b in itertools.product(bank,bank):
  if a==b:continue
  left=[a,b];lt=tape(' '.join(left))
  if len(lt)*2<min_letters:continue
  right=segment(lt[::-1],set(left),bank)
  if right is None:continue
  text=' '.join(left+right); au=audit(text)
  if au['two_pointer_exact']:rows.append({'rendered':text,'audit':au,'reader_worthy':False,'left_words':left,'right_words':right,'provenance':{'brown_headword_bank':True,'both_sides_online_segmented':True,'immediate_mirror':True,'fresh_words':True,'no_catalogue':True,'posthoc_repair':False}})
  if len(rows)>=limit:return rows
 return rows
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--min-letters',type=int,default=40);ap.add_argument('--limit',type=int,default=20);ap.add_argument('--out',required=True);a=ap.parse_args();rows=run(a.min_letters,a.limit);p={'experiment':'broad-lexical-boundary-wfsa-20260925','host':socket.gethostname(),'parameters':vars(a),'candidates':rows,'closures':len(rows),'reader_worthy':0,'provenance':{'decoder':'Brown headword lexical WFSA','syntax_not_claimed':True,'exactness':'two-pointer plus SHA-256','no_repeated_units':True},'next_construction':'if lexical paths exist, add held-out agreement/valency states; otherwise expand online phrase length state.'};Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(p,indent=2)+'\n');print(json.dumps({k:p[k] for k in ('experiment','closures','reader_worthy')}))
if __name__=='__main__':main()
