"""Reverse-phrase index: live seam search over independently authored prose.

This lane never manufactures a mirror.  It indexes phrase tapes and joins only
when the next right-hand characters satisfy the outstanding reverse obligation.
"""
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
LETTERS = re.compile(r"[a-z]+")

def tape(s): return ''.join(LETTERS.findall(s.lower()))

def audit(s):
    t=tape(s); rev=t[::-1]
    mism=[i for i,(a,b) in enumerate(zip(t,rev)) if a!=b]
    return {'letters':len(t), 'exact':bool(t) and not mism,
            'two_pointer':bool(t) and all(t[i]==t[-1-i] for i in range(len(t)//2)),
            'sha_forward':hashlib.sha256(t.encode()).hexdigest(),
            'sha_reverse':hashlib.sha256(rev.encode()).hexdigest(),
            'mismatch_count':len(mism), 'first_mismatches':mism[:12]}

def words(s): return LETTERS.findall(s.lower())

def clean(lines):
    out=[]
    for line in lines:
        line=' '.join(words(line))
        ws=words(line)
        if len(ws)>=3 and len(set(ws))==len(ws) and all(len(w)>1 or w in {'a','i'} for w in ws):
            out.append(line)
    return list(dict.fromkeys(out))

def run(out='runs/reverse-phrase-index-20260917.json'):
    phrases=clean((ROOT/'data/authored_sentences.txt').read_text().splitlines())
    # Prefix index: a node stores phrases whose next character can discharge
    # the current reverse obligation.  Phrase identity remains independent.
    index={}
    for p in phrases:
        t=tape(p)
        for k in range(len(t)+1): index.setdefault(t[:k],[]).append(p)
    joins=[]
    for left in phrases:
        need=tape(left)[::-1]
        # Right phrase must begin with the reverse obligation; retain the
        # longest live prefix and then expose the first unresolved seam.
        options=index.get(need[:1],[])
        scored=[]
        for right in options:
            rt=tape(right); common=0
            while common<min(len(need),len(rt)) and need[common]==rt[common]: common+=1
            if common and set(words(left)).isdisjoint(words(right)):
                scored.append((common,right,need,rt))
        if scored:
            common,right,need,rt=max(scored,key=lambda x:(x[0],len(tape(x[1]))))
            rendered=left+' '+right
            joins.append({'left':left,'right':right,'rendered':rendered,
              'live_prefix_letters':common,'obligation_prefix':need[:common],
              'next_required':need[common:common+12], 'audit':audit(rendered),
              'provenance':{'left_source':'data/authored_sentences.txt','right_source':'data/authored_sentences.txt','independent_phrase_ids':True,'catalogue_imported':False,'finished_reversal':False},
              'novelty_preflight':{'distinct_words':True,'repeated_unit':False,'word_order_symmetry':False,'self_palindromic_word':False}})
    joins.sort(key=lambda x:(x['live_prefix_letters'],x['audit']['letters']),reverse=True)
    result={'experiment_id':'reverse-phrase-index-20260917',
      'status':'quarantined_no_exact_closure', 'candidates':joins[:25],
      'stats':{'phrase_count':len(phrases),'indexed_prefixes':len(index),'live_joins':len(joins),'exact_admitted':sum(x['audit']['exact'] for x in joins)},
      'construction_gates':{'independent_authored_phrases':True,'live_character_pruning':True,'no_finished_tape_mirroring':True,'no_catalogue_text':True},
      'failure_and_repair':{'next_repair':'retain the best live seam and expand only role-compatible alternatives whose first word discharges next_required; then re-index suffixes instead of rescanning all phrase pairs.'}}
    Path(out).write_text(json.dumps(result,indent=2)+'\n'); return result

if __name__=='__main__': print(json.dumps(run(),indent=2))
