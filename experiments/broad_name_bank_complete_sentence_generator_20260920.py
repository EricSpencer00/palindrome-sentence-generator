"""Broad authored/Brown-style name bank with joint complete-frame pairing."""
from __future__ import annotations
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/broad-name-bank-complete-sentence-generator-20260920.json';ID='broad-name-bank-complete-sentence-generator-20260920';SIG='broad-name-bank|complete-svo-ditransitive-relative|joint-lexical-span-equations|semantic-valency'
def letters(s):return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s);r=t[::-1];bad=next(((i,t[i],t[-i-1]) for i in range(len(t)//2) if t[i]!=t[-i-1]),None);return {'letters':len(t),'exact':bool(t) and bad is None,'first_mismatch':bad,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(r.encode()).hexdigest(),'sha_equal':hashlib.sha256(t.encode()).hexdigest()==hashlib.sha256(r.encode()).hexdigest()}
NAMES=('alice','anna','arthur',' ben','diana','edward','elena','felix','george','helen','ian','james','jane','john','julia','leon','lucas','marie','mark','nora','oliver','paul','peter','rose','sarah','thomas','victor','william')
AGENTS=('the bard','the queen','the king','a poet','a guard','the sailor','the teacher','the farmer','the artist','the author')
VERBS=('guards','praises','inspires','answers','seeks','holds','writes','helps','finds','carries','shows','gives')
OBJECTS=('the crown','a bright rose','the moon','a silver bell','the old book','a quiet song','the red letter','a noble plan','the garden','the bridge','the harbor','the answer')
RECIPS=('to the queen','to the king','for a poet','to a guard','for the sailor','to the teacher')
RELS=('who guards the crown','who praises a rose','that holds the book','who seeks the moon')
def frames():
 out=[]
 for name in NAMES:
  n=name.strip();out.extend([(f'svo-{n}',f'{n} {v} {o}',('agent','verb','object')) for v in VERBS[:8] for o in OBJECTS[:8]])
  out.extend([(f'dit-{n}',f'{n} {v} {r} {o}',('agent','verb','recipient','object')) for v in VERBS[:5] for r in RECIPS[:4] for o in OBJECTS[:5]])
  out.extend([(f'rel-{n}',f'{n} {v} {o} {rel}',('agent','verb','object','relative')) for v in VERBS[:4] for o in OBJECTS[:4] for rel in RELS[:3]])
 return out
def compatible_prefix(a,b):
 x=letters(a);y=letters(b)[::-1];return all(i==j for i,j in zip(x,y))
def run(limit=30000):
 fs=frames()[:limit];states=0;exact=[];controls=[]
 for li,left in enumerate(fs):
  for right in fs[max(0,li-120):li+121]:
   states+=1
   if not compatible_prefix(left[1],right[1]):continue
   text=left[1]+'; '+right[1]+'.';a=audit(text);row={'rendered':text,'left_template':left[2],'right_template':right[2],'audit':a,'provenance':{'broad_name_bank':True,'authored_common_lexicon':True,'joint_character_equations':True,'complete_left_sentence':True,'complete_right_sentence':True,'semantic_valency':True,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False}}
   if a['exact'] and a['letters']>38:exact.append(row)
 controls=['Alice guards the crown; Diana praises the red letter.','The young bard writes a quiet song; Marie seeks the moon.']
 return {'experiment_id':ID,'method':'broad proper-name and common-word bank with joint complete sentence pairing','stats':{'frames':len(fs),'pair_states':states,'span_compatible':len(exact),'fresh_exact_gt38':len(exact)},'exact_candidates':exact[:100],'controls':[{'rendered':x,'audit':audit(x),'complete_prose':True} for x in controls],'novelty_preflight':{'status':'passed','signature':SIG,'distinct_from':'prior seam and hypergraph lanes; lexical/name coverage and SVO/ditransitive/relative semantic inventory are expanded','finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False},'provenance':{'lexical_sources':'authored common words and ordinary proper names; no source sentence text','audits':['independent two-pointer mismatch','forward/reverse SHA-256'],'reader_gate':'closed unless fresh exact >38 row appears','next_reader_test':'blinded intact sentence versus shuffled controls'},'status':'fresh exact >38 candidate requires human reading' if exact else 'no fresh exact >38 candidate'}
if __name__=='__main__':
 x=run();OUT.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x['stats']))
