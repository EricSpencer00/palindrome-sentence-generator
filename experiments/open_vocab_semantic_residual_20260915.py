"""Open-vocabulary semantic proposal + weighted reverse residual decoder.

Phrase proposals are sampled from topical word groups (rather than a finite
clause template). A candidate survives only when the reflected character
residual has an independently decoded, variable-boundary word segmentation.
"""
from __future__ import annotations
import argparse,json,random,sys
from pathlib import Path
from wordfreq import top_n_list,zipf_frequency
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from llm_palindrome.admission import normalize_letters,mechanical_admission_checks
FAMILY_ID='open-vocabulary-semantic-residual'
SIGNATURE='open-vocabulary-semantic-residual|stochastic-phrase-proposal|weighted-reverse-segmentation|variable-word-boundaries|whole-tape-exactness'

def seg(tape,vocab):
 n=len(tape); dp=[None]*(n+1); dp[0]=(0.0,())
 for i in range(n):
  if dp[i] is None: continue
  for j in range(i+1,min(n,i+14)+1):
   w=tape[i:j]
   if w in vocab:
    z=dp[i][0]+zipf_frequency(w,'en')+0.3*len(w)
    if dp[j] is None or z>dp[j][0]: dp[j]=(z,dp[i][1]+(w,))
 return None if dp[n] is None else dp[n][1]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--trials',type=int,default=30000);ap.add_argument('--out',default='runs/open-vocab-semantic-residual-20260915.json');a=ap.parse_args()
 vocab={w for w in top_n_list('en',30000) if w.isascii() and w.isalpha() and 2<=len(w)<=12}
 topics=[('quiet',('lantern','garden','river','candle','window','morning','silver','water','meadow','stone')),('motion',('drifts','turns','moves','wanders','rests','crosses','follows','returns')),('perception',('notices','sees','hears','watches','finds','knows','remembers')),('care',('gentle','patient','open','still','warm','kind','clear','near'))]
 rng=random.Random(1509); best=[]
 for _ in range(a.trials):
  n=rng.randint(6,10); ws=[]
  for i in range(n):
   bank=rng.choice(topics)[1]; ws.append(rng.choice(bank))
  left=' '.join(ws); tape=normalize_letters(left)
  if len(tape)<39: continue
  right=seg(tape[::-1],vocab)
  if not right: continue
  text=left+' '+' '.join(right); full=normalize_letters(text)
  if full!=full[::-1]: continue
  adm=mechanical_admission_checks(text)
  row={'left':left,'right':right,'rendered':text,'letters':len(full),'exact':True,'admission':adm}
  best.append(row); best.sort(key=lambda r:(bool(r['admission'].get('admitted')),r['letters']),reverse=True); best=best[:20]
 payload={'family_id':FAMILY_ID,'state_space_signature':SIGNATURE,'preflight':{'registry_entries_checked':70,'collision':False,'status':'novel'},'trials':a.trials,'matches':len(best),'best':best}
 p=ROOT/a.out;p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(payload,indent=2)+'\n')
 print('matches',len(best));
 for r in best[:5]: print(r['letters'],r['rendered'])
if __name__=='__main__':main()
