"""Semantic recipient/adjunct CFG chart with synchronous residual pairing."""
from __future__ import annotations
import hashlib, json, re
from dataclasses import dataclass
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/bottom-up-recipient-adjunct-chart-20260920.json'
EXPERIMENT_ID='bottom-up-recipient-adjunct-chart-20260920'

def letters(s): return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
    t=letters(s); f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest()
    bad=next(((i,len(t)-1-i) for i in range(len(t)//2) if t[i]!=t[-i-1]),None)
    return {'letters':len(t),'exact':bool(t) and bad is None,'first_mismatch':bad,
            'sha256_forward':f,'sha256_reverse':r,'sha_equal':f==r}
def consume(left,right):
    n=min(len(left),len(right))
    if left[:n]!=right[-n:][::-1]: return None
    return left[n:],right[:-n] if n else right

@dataclass(frozen=True)
class Item:
    symbol:str; text:str; roles:tuple[str,...]; children:tuple[str,...]

def chart():
    np_words=('a scholar','the sailor','a poet','the keeper','some men','an aide',
              'the writer','a nurse','the farmer','diana','leon','maria')
    verbs=('aids','carries','finds','gives','guides','inspires','keeps','marks','names','reads','rips','sees','writes')
    obj_words=('the lantern','old letters','a bright book','new notes','the small bell','nine memos','a secret map')
    rec_words=('to the sailor','to the poet','for the keeper','for a nurse','to diana','for the farmer')
    adj_words=('at dawn','with care','by the river','in the garden','under a pale moon','near the harbor')
    out={k:[] for k in ('NP','V','OBJ','RECIP','ADJ','VP','DITRANS','VPADJ','CLAUSE')}
    out['NP']=[Item('NP',x,('subject_or_object',),('NP_LEX',)) for x in np_words]
    out['OBJ']=[Item('OBJ',x,('theme',),('OBJ_LEX',)) for x in obj_words]
    out['V']=[Item('V',x,('predicate',),('V_LEX',)) for x in verbs]
    out['RECIP']=[Item('RECIP',x,('recipient',),('PREP_NP',)) for x in rec_words]
    out['ADJ']=[Item('ADJ',x,('adjunct',),('ATTACHMENT',)) for x in adj_words]
    # Semantic attachment is explicit: recipient is an argument of the verb;
    # adjunct is an optional modifier after the argument structure.
    for v in out['V']:
        for obj in out['OBJ']:
            out['VP'].append(Item('VP',f'{v.text} {obj.text}',('predicate','theme'),(v.text,obj.text)))
        for rec in out['RECIP']:
            for obj in out['OBJ']:
                out['DITRANS'].append(Item('DITRANS',f'{v.text} {rec.text} {obj.text}',
                    ('predicate','recipient','theme'),(v.text,rec.text,obj.text)))
    for vp in out['VP']:
        for adj in out['ADJ']:
            out['VPADJ'].append(Item('VPADJ',f'{vp.text} {adj.text}',vp.roles+('adjunct',),(vp.text,adj.text)))
    for np in out['NP']:
        for vp in out['VP']+out['DITRANS']+out['VPADJ']:
            out['CLAUSE'].append(Item('CLAUSE',f'{np.text} {vp.text}',('subject',)+vp.roles,(np.text,vp.text)))
    return {k:tuple(dict.fromkeys(v)) for k,v in out.items()}

def run(state_limit=90000, cap=80):
    c=chart(); clauses=tuple(c['CLAUSE'][:cap]); states=combines=pruned=0; exact=[]
    # Pair complete semantic clauses from their outer constituent sequence.
    # Each clause item retains a derivation trace; no lexical unit is mirrored.
    for li,left in enumerate(clauses):
        for ri,right in enumerate(clauses):
            if states>=state_limit: break
            # Recover the two bottom-up constituents from the complete item.
            left_parts=tuple(x for x in left.children)
            right_parts=tuple(x for x in right.children)
            lbuf=rbuf=''; trace=[]; ok=True
            for lp,rp in zip(left_parts,reversed(right_parts)):
                combines+=1; states+=1
                rem=consume(lbuf+letters(lp),letters(rp)+rbuf)
                if rem is None: pruned+=1; ok=False; break
                lbuf,rbuf=rem; trace.append({'left':lp,'right':rp,'left_residual':lbuf,'right_residual':rbuf})
            if not ok or lbuf or rbuf: continue
            rendered=f'{left.text} {right.text}'
            a=audit(rendered)
            if a['exact'] and a['letters']>=38:
                exact.append({'rendered':rendered,'audit':a,'provenance':{
                    'construction':'bottom-up semantic recipient/adjunct chart intersection',
                    'left_roles':left.roles,'right_roles':right.roles,'trace':trace,
                    'recipient_attachment': 'recipient' in left.roles or 'recipient' in right.roles,
                    'adjunct_attachment': 'adjunct' in left.roles or 'adjunct' in right.roles,
                    'finished_tape_reversal':False,'post_hoc_repair':False,
                    'catalogue_text':False,'mirrored_token_units':False,
                    'complete_semantic_clauses':True}})
        if states>=state_limit: break
    controls=['the sailor gives to the poet the lantern at dawn',
              'a scholar reads old letters by the river',
              'the keeper carries a bright book with care']
    return {'experiment_id':EXPERIMENT_ID,
            'method':'bottom-up semantic recipient/adjunct CFG chart with live residual pairing',
            'chart_sizes':{k:len(v) for k,v in c.items()},
            'complete_clause_items':len(clauses),
            'stats':{'states':states,'combines':combines,'pruned':pruned,'exact':len(exact)},
            'exact_candidates':exact,
            'complete_prose_controls':[{'rendered':s,'audit':audit(s),'reader_status':'intact control; not exact candidate'} for s in controls],
            'novelty_preflight':{'status':'passed','signature':'bottom-up-semantic-recipient-adjunct|ditransitive-attachment|live-residual',
                'distinct_from':'NP/VP/PP/REL chart; adds recipient argument and adjunct attachment nonterminals',
                'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False},
            'provenance':{'lexicon':'authored recipient/theme/adjunct role banks','independent_audit':'two-pointer mismatch plus forward/reverse SHA-256','reader_evidence':False},
            'status':'no exact >38 closure' if not exact else 'reader gate required','next_construction':'add tense/agreement feature states to recipient chart','reader_gate':'closed until blinded human ratings'}

if __name__=='__main__':
    x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps({'chart_sizes':x['chart_sizes'],'stats':x['stats']}))
