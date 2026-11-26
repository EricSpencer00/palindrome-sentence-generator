"""Finite ordinary grammar NFA with bidirectional constituent completion."""
from dataclasses import dataclass
import hashlib,itertools,json,time
from pathlib import Path
from llm_palindrome.block_seams import Piece,Seam
from llm_palindrome.admission import normalize_letters
from experiments.typed_partial_bank_20261009 import FRAMES,SUBJECTS,YieldTrie
from llm_palindrome.typed_constituents import article

ROOT=Path(__file__).resolve().parents[1]
SG=SUBJECTS+('He','She','The sailor','The writer')
PL=('We','They','You','The children','The artists')
PAST={'reads':'read','writes':'wrote','opens':'opened','closes':'closed','carries':'carried','repairs':'repaired',
'moves':'moved','finds':'found','keeps':'kept','holds':'held','notes':'noted','paints':'painted','weighs':'weighed','watches':'watched','signs':'signed'}
BASE={'reads':'read','writes':'write','opens':'open','closes':'close','carries':'carry','repairs':'repair',
'moves':'move','finds':'find','keeps':'keep','holds':'hold','notes':'note','paints':'paint','weighs':'weigh','watches':'watch','signs':'sign'}
PERSON_ADJ=('calm','tired','ill','ready','happy','asleep','late','safe','well')
MODIFIERS={'book':('old','new'),'note':('brief','old'),'letter':('brief','new'), 'report':('brief','new'),
'map':('old','new'),'boat':('small','old'),'gem':('small','blue'),'pen':('blue','new'), 'wave':('small','large'),
'door':('old','blue'),'case':('small','old'),'permit':('new','old'),'diary':('old','new')}


def grammar():
    paths=[]
    def add(family,slots,roles,event,question=False):
        paths.append(dict(id=f'{family}-{len(paths):03d}',family=family,slots=[tuple(s) for s in slots],roles=roles,
                          event=event,question=question,evidence='Curated agreement/valency grammar; normal lexical senses; no model ratings'))
    # Determiners and modifiers remain separate constituents, not opaque full blocks.
    for verb,nouns in FRAMES.items():
        for tense,subjects,predicate in [('present-sg',SG,verb),('present-pl',PL,BASE[verb]),('past',SG+PL+('I',),PAST[verb])]:
            for modifier in (False,True):
                for noun in nouns:
                    if modifier and noun not in MODIFIERS:continue
                    # Determiner allomorph depends on the next pronounced word.
                    groups={}
                    for word in (MODIFIERS[noun] if modifier else (noun,)):
                        groups.setdefault(article(word),[]).append(word)
                    for indefinite,options in groups.items():
                        np=[(indefinite,'the','her','his','my')]+([tuple(options)] if modifier else [])+[(noun,)]
                        add('SVO-'+tense,[subjects,(predicate,),*np],['subject','predicate','determiner']+(['modifier'] if modifier else [])+['patient'],BASE[verb])
        for noun in nouns:
            for aux in ('can','will','could'):
                add('auxiliary',[SG+PL+('I',),(aux,),(BASE[verb],),('a','the'),(noun,)],['subject','auxiliary','predicate','determiner','patient'],BASE[verb])
                add('aux-question',[(aux,),SG+PL+('I',),(BASE[verb],),('a','the'),(noun,)],['auxiliary','subject','predicate','determiner','patient'],BASE[verb],True)
            add('do-question',[('Does',),SG,(BASE[verb],),('a','the'),(noun,)],['auxiliary','subject','predicate','determiner','patient'],BASE[verb],True)
            add('do-question',[('Do',),PL+('I',),(BASE[verb],),('a','the'),(noun,)],['auxiliary','subject','predicate','determiner','patient'],BASE[verb],True)
    for subjects,cop in [(SG,'is'),(PL,'are'),(('I',),'am'),(SG+('I',),'was'),(PL,'were')]:
        add('copular-adjective',[subjects,(cop,),PERSON_ADJ],['subject','copula','quality'],'attribute')
        add('copular-profession',[subjects,(cop,),('a',),('nurse','writer','sailor','teacher','pilot')],['subject','copula','determiner','profession'],'classify')
    for subj,verbs in [(SG,('walks','runs','waits','rests','sleeps','smiles')),
                       (PL+('I',),('walk','run','wait','rest','sleep','smile')),
                       (SG+PL+('I',),('walked','ran','waited','rested','slept','smiled'))]:
        add('intransitive',[subj,verbs],['subject','predicate'],'intransitive')
        add('intransitive-location',[subj,verbs,('near','beside'),('the',),('door','boat','lake')],['subject','predicate','preposition','determiner','location'],'intransitive')
    # Ordinary vocative questions. Known examples automatically become controls.
    add('vocative-question',[('Eva,','Nora,','Tim,'),('can',),('I','we'),('see','hear'),('bees','birds','bats'),('in',),('a',),('cave','room')],
        ['addressee','auxiliary','subject','predicate','patient','preposition','determiner','location'],'perceive',True)
    return paths


@dataclass(frozen=True)
class Stream:
    nodes:tuple
    active:tuple=()
    completed:tuple=()


def inventory(paths,stream,side):
    result={}
    for tid,pos in stream.nodes:
        slots=paths[tid]['slots'] if side=='left' else paths[tid]['slots'][::-1]
        if pos<len(slots):
            for word in slots[pos]:result.setdefault(word,[]).append((tid,pos+1))
    return result


def terminal(paths,stream):
    return tuple(tid for tid,pos in stream.nodes if pos==len(paths[tid]['slots']))


def run(seconds=10,max_states=12000):
    started=time.monotonic();deadline=started+seconds;paths=grammar();root=tuple((i,0) for i in range(len(paths)))
    blank=Stream(root);queue=[(Seam(),blank,blank)];seen=set(queue);head=0;failed={};examples=[];complete=[]
    while head<len(queue) and head<max_states and time.monotonic()<deadline:
        st,l,r=queue[head];head+=1;d=st.debt()
        count=len(l.completed)+len(r.completed)
        if count>=2 and not l.active and not r.active:
            if st.exact():
                completed=l.completed+r.completed
                text=' '.join((' '.join(words)+('?' if paths[tids[0]]['question'] else '.')) for words,tids in completed)
                tape=normalize_letters(text);assert tape==tape[::-1]
                entities=[dict(subject=next((word for word,role in zip(words,paths[tids[0]]['roles']) if role=='subject'),None),
                              patient=next((word for word,role in zip(words,paths[tids[0]]['roles']) if role=='patient'),None),
                              relation=paths[tids[0]]['event']) for words,tids in completed]
                complete.append(dict(text=text,normalized_sha256=hashlib.sha256(tape.encode()).hexdigest(),
                    grammar_parses=[list(tids) for words,tids in completed],block_tapes=[normalize_letters(' '.join(words)) for words,tids in completed],entities_and_relations=entities,
                    human_coherence_verified=False,duplicate_sentence=len({words for words,tids in completed})<len(completed)))
            if count>=3:continue
        # Epsilon sentence commitments on either side must remain available,
        # even when that side owns the residual. Letter mismatches are still hard.
        sides=('left','right') if d['owner'] is None else (('right','left') if d['owner']=='left' else ('left','right'))
        for side in sides:
            stream=l if side=='left' else r
            tids=terminal(paths,stream)
            if tids and stream.active:
                words=stream.active if side=='left' else stream.active[::-1]
                completed=stream.completed+((words,tids),) if side=='left' else ((words,tids),)+stream.completed
                new=Stream(root,(),completed)
                state=(st,new,r) if side=='left' else (st,l,new)
                if state not in seen:seen.add(state);queue.append(state)
                if len(examples)<30 and words not in [tuple(e['words']) for e in examples]:
                    examples.append(dict(words=words,parse_ids=tids,complete_sentence=True,novelty_not_asserted=True))
            if count>=3:continue
            menu=inventory(paths,stream,side)
            if not menu:continue
            conditioned=d['owner'] is not None and side!=d['owner']
            allowed=YieldTrie(tuple(menu),side).matching(d['residual']) if conditioned else tuple(menu)
            rejected=set(menu)-set(allowed)
            for word in rejected:
                for tid,pos in menu[word]:
                    role=paths[tid]['roles'][pos-1 if side=='left' else len(paths[tid]['roles'])-pos]
                    key=(side,role,d['residual']);failed[key]=failed.get(key,0)+1
            for word in allowed:
                child=st.add(side,Piece('NFA',len(stream.active),word))
                if child is None:continue
                new=Stream(tuple(menu[word]),stream.active+(word,),stream.completed)
                state=(child,new,r) if side=='left' else (child,l,new)
                if state not in seen:seen.add(state);queue.append(state)
    known=set(json.loads((ROOT/'data/known_palindromes.json').read_text()))
    controls=[]
    for text,source in [('Eva, can I see bees in a cave?','data/readable_palindrome_centres.json#eva'),('Do geese see God?','data/known_palindromes.json#dogeeseseegod')]:
        tape=normalize_letters(text);assert tape in known and tape==tape[::-1]
        controls.append(dict(text=text,source=source,label='known_correctness_control',normalized_sha256=hashlib.sha256(tape.encode()).hexdigest(),not_novel=True))
    for row in complete:
        row['catalogue_control']=normalize_letters(row['text']) in known
        row['all_blocks_catalogue_controls']=all(t in known for t in row['block_tapes'])
        row['qualifies_as_new_nonduplicate_construction']=not row['duplicate_sentence'] and not row['catalogue_control'] and not row['all_blocks_catalogue_controls']
    return dict(schema_version=1,status='exhausted' if head==len(queue) else 'time_or_state_cap',
      elapsed_seconds=time.monotonic()-started,states_visited=head,remaining_states=len(queue)-head,
      grammar_paths=paths,grammar_options=dict(nonpalindromic_required=False,duplicates_count_as_progress=False,shared_entity_required=False),
      complete_constructions=complete,complete_parse_examples=examples,known_controls=controls,
      failure_classes=[dict(side=k[0],role=k[1],residual=k[2],rejected_parse_branches=v) for k,v in sorted(failed.items(),key=lambda x:-x[1])],
      limits=dict(seconds=seconds,max_states=max_states,workers=1,max_sentence_blocks=3),human_ratings=None,
      provenance=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       grammar='Curated finite agreement/valency paths, locally retained vocabulary; independent ratings pending'))


if __name__=='__main__':
    out=ROOT/'research/block-seams/fixtures/expanded-grammar-seams-002.json'
    if out.exists():raise FileExistsError('preserve previous results')
    r=run();out.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:r[k] for k in ('status','elapsed_seconds','states_visited','complete_constructions','complete_parse_examples')},indent=2))
