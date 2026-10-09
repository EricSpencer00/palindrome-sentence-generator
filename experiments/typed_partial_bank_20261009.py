"""Residual-conditioned finite SVO grammar; no model or human reward."""
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import time
from llm_palindrome.admission import normalize_letters
from llm_palindrome.block_seams import Piece,Seam

ROOT=Path(__file__).resolve().parents[1]
# Human-readable finite grammar/vocabulary, newly curated from local words.
SUBJECTS=('Tara','Nora','Tim','Sam','Dan','Ada','Eva','The clerk','The child','The artist','A man','A nurse')
FRAMES={
 'reads':('book','note','letter','report','diary','map'),
 'writes':('book','note','letter','report','diary'),
 'opens':('book','door','case','letter'),
 'closes':('book','door','case'),
 'carries':('book','note','letter','map','boat','gem','pen','banana'),
 'repairs':('boat','door','case'),
 'moves':('boat','case','gem','sofa'),
 'finds':('book','note','letter','map','gem','pen'),
 'keeps':('book','note','letter','map','gem','pen','permit','diary'),
 'holds':('book','note','letter','map','gem','pen','permit'),
 'notes':('wave','weight','drama'),
 'paints':('boat','door','case','wave'),
 'weighs':('gem','banana'),
 'watches':('boat','wave'),
 'signs':('letter','report','permit'),
}
DETERMINERS=('a','the','her','his','my')
# These are grammar drafts, not a claim of ordinary meaning in every combination.
# All subjects are singular; frames are finite 3sg transitive, NPs singular.


@dataclass(frozen=True)
class Cursor:
    stage:int=0
    subject:str=''
    verb:str=''
    determiner:str=''
    noun:str=''


def choices(cursor,side):
    if cursor.stage==4:return []
    if side=='left':
        options=(SUBJECTS,tuple(FRAMES),DETERMINERS,FRAMES.get(cursor.verb,()))[cursor.stage]
    else:
        nouns=tuple(sorted(set(itertools.chain.from_iterable(FRAMES.values()))))
        options=(nouns,DETERMINERS,tuple(v for v,ns in FRAMES.items() if cursor.noun in ns),SUBJECTS)[cursor.stage]
    return list(options)


def advance(cursor,side,word):
    key=(('subject','verb','determiner','noun') if side=='left' else ('noun','determiner','verb','subject'))[cursor.stage]
    return Cursor(**{**cursor.__dict__,key:word,'stage':cursor.stage+1})


def sentence(cursor):return f'{cursor.subject} {cursor.verb} {cursor.determiner} {cursor.noun}.'


def compatible(residual,tape):return residual.startswith(tape) or tape.startswith(residual)


class YieldTrie:
    """Index licensed oriented yields; include ancestors and descendants."""
    def __init__(self,options,side):
        self.root={}
        for word in options:
            tape=normalize_letters(word)
            if side=='right':tape=tape[::-1]
            node=self.root
            for char in tape:node=node.setdefault(char,{})
            node.setdefault('',[]).append(word)

    def matching(self,residual):
        node=self.root;found=[]
        for char in residual:
            found.extend(node.get('',[]))
            if char not in node:return found
            node=node[char]
        def descendants(n):
            found.extend(n.get('',[]))
            for char,child in n.items():
                if char:descendants(child)
        descendants(node)
        return found


def run(seconds=10,max_states=15000):
    start=time.monotonic();deadline=start+seconds
    vocabulary=ROOT/'tools/polaris/payload/vocab30k.txt';lexicon=ROOT/'data/lexicon.txt'
    local=set(vocabulary.read_text().splitlines())|set(lexicon.read_text().splitlines())
    words=set(w.lower() for phrase in (*SUBJECTS,*FRAMES,*DETERMINERS,*set(itertools.chain.from_iterable(FRAMES.values()))) for w in phrase.split())
    # Common names are explicitly curated proper nouns, not inferred POS/name rescues.
    names={s.lower() for s in SUBJECTS if ' ' not in s}
    assert words<=local|names,words-local-names
    queue=[(Seam(),Cursor(),Cursor())];seen=set(queue);head=0;failures=[];suggestions=[];complete=[];exact=[];indexes={}
    endpoint=[]
    for subject,noun in itertools.product(SUBJECTS,sorted(set(itertools.chain.from_iterable(FRAMES.values())))):
        a,b=normalize_letters(subject),normalize_letters(noun)[::-1]
        if compatible(a,b):endpoint.append(dict(subject=subject,right_object=noun))
    while head<len(queue) and head<max_states and time.monotonic()<deadline:
        st,l,r=queue[head];head+=1;d=st.debt()
        if l.stage==r.stage==4:
            text=sentence(l)+' '+sentence(r);tape=normalize_letters(text)
            row=dict(left_parse=l.__dict__,right_parse=r.__dict__,text=text,
                     syntax_complete=True,endpoint_feasible=True,
                     shared_entities=sorted(set((l.subject,l.noun))&set((r.subject,r.noun))),
                     discourse_human_verified=False,exact_global=bool(tape) and tape==tape[::-1])
            complete.append(row)
            if row['exact_global']:exact.append(row)
            continue
        # Shorter tape gets its next grammar constituent; finite chart cap explicit.
        sides=('left','right') if d['owner'] is None else (('right',) if d['owner']=='left' else ('left',))
        if all((l if side=='left' else r).stage==4 for side in sides):
            sides=tuple(side for side in ('left','right') if (l if side=='left' else r).stage<4)
        for side in sides:
            cursor=l if side=='left' else r
            licensed=choices(cursor,side)
            conditioned=(d['owner'] is not None and side!=d['owner'])
            key=(side,tuple(licensed))
            if key not in indexes:indexes[key]=YieldTrie(licensed,side)
            allowed=set(indexes[key].matching(d['residual'])) if conditioned else set(licensed)
            for word in licensed:
                tape=normalize_letters(word);oriented=tape if side=='left' else tape[::-1]
                conditioned=(d['owner'] is not None and side!=d['owner'])
                if conditioned and word not in allowed:
                    if len(failures)<1000:failures.append(dict(side=side,stage=cursor.stage,residual=d['residual'],rejected=word,reason='irreversible prefix mismatch'))
                    continue
                piece=Piece('left-SVO' if side=='left' else 'right-SVO',cursor.stage,word)
                child=st.add(side,piece)
                if child is None:continue
                new=advance(cursor,side,word)
                state=(child,new,r) if side=='left' else (child,l,new)
                if state not in seen:seen.add(state);queue.append(state)
                if conditioned and len(suggestions)<1000:
                    suggestions.append(dict(side=side,stage=cursor.stage,open_residual=d['residual'],
                         compatible_phrase=word,new_debt=child.debt(),
                         lexical_fragment_cursor=dict(matched_letters=min(len(d['residual']),len(oriented)),
                             unconsumed_yield=oriented[len(d['residual']):],
                             unconsumed_prior_residual=d['residual'][len(oriented):]),left_parse=(new if side=='left' else l).__dict__,
                         right_parse=(new if side=='right' else r).__dict__))
    # Complete grammatical examples extend compatible fragments, but are not palindromes.
    examples=[]
    for s in suggestions:
        for side,key in (('left','left_parse'),('right','right_parse')):
            c=Cursor(**s[key])
            if c.stage>=2:
                while c.stage<4:
                    options=choices(c,side)
                    if not options:break
                    c=advance(c,side,options[0])
                if c.stage==4 and sentence(c) not in examples:examples.append(sentence(c))
    return dict(schema_version=1,scope='Development finite SVO grammar, no held-out or human ratings',
       provenance=dict(grammar='New curated 3sg transitive frame inventory, no Diana/catalogue source',
          vocabulary_source=str(vocabulary.relative_to(ROOT)),vocabulary_sha256=hashlib.sha256(vocabulary.read_bytes()).hexdigest(),
          lexicon_sha256=hashlib.sha256(lexicon.read_bytes()).hexdigest(),curated_proper_names=sorted(names),
          source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
       grammar=dict(subjects=SUBJECTS,frames=FRAMES,determiners=DETERMINERS),
       options=dict(require_nonpalindromic_blocks=False,require_unique_sentences=False,shared_entities_required=False),
       limits=dict(seconds=seconds,max_states=max_states,workers=1),elapsed_seconds=time.monotonic()-start,
       status='exhausted' if head==len(queue) else 'time_or_state_cap',states_visited=head,frontier_remaining=len(queue)-head,
       endpoint_suggestions=endpoint,residual_conditioned_suggestions=suggestions,failed_joins=failures,
       complete_pair_parses=complete,grammatical_completion_examples=examples,exact_constructions=exact,human_ratings=None)


if __name__=='__main__':
    out=ROOT/'research/block-seams/fixtures/typed-partial-bank-002.json'
    if out.exists():raise FileExistsError('preserve frozen result')
    r=run();out.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:r[k] for k in ('status','elapsed_seconds','states_visited','endpoint_suggestions','grammatical_completion_examples','exact_constructions')},indent=2))
