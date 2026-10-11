"""Limited source-independent constituent grammar; no semantic certification.

Slots compose typed subject NPs, agreement-marked predicates and licensed
complements. Stored sentences and provenance IDs never define admission.
The article rule is valid for this declared lexicon, not general English
pronunciation (hour/university require explicit lexical overrides).
"""
from functools import lru_cache
import re
from .admission import normalize_letters


def words(text):
    return tuple(normalize_letters(w) for w in text.split() if normalize_letters(w))


def article(word):
    return 'an' if word.lower().startswith(tuple('aeiou')) else 'a'


def copula(subject, tense='present'):
    w=words(subject) if isinstance(subject,str) else tuple(subject)
    plural=w in [('we',),('they',),('you',)] or (w and w[-1] in PLURAL)
    if tense=='past':return 'were' if plural else 'was'
    return 'am' if w==('i',) else 'are' if plural else 'is'


NAMES=('Liam','Leon','Noel','Tara','Nora','Tim','Sam','Dan','Ada','Eva')
COUNT=('rider','clerk','child','artist','man','nurse','pilot','baker','gardener',
       'sailor','writer','teacher','book','note','letter','report','diary','map',
       'boat','gem','pen','door','case','permit','carat','tray','pot','cave','room')
MASS=('mail','iron','evil','water','bread')
PLURAL=('children','artists','men','dogs','bees','birds','bats','pets','notes','letters')
ADJECTIVES=('old','new','red','blue','small','brief','large')
QUALITIES=('calm','tired','ill','ready','happy','asleep','late','safe','well')
VERBS={
 'see':('sees','saw',COUNT+MASS+PLURAL+NAMES),
 'read':('reads','read',('book','note','letter','report','diary','map','mail')),
 'write':('writes','wrote',('book','note','letter','report','diary')),
 'carry':('carries','carried',('book','note','letter','map','boat','gem','pen','pot','tray')),
 'hold':('holds','held',('book','note','letter','map','gem','pen','permit','pot','tray')),
 'move':('moves','moved',('boat','case','gem','pot','tray')),
 'open':('opens','opened',('book','door','case','letter')),
 'close':('closes','closed',('book','door','case')),
 'find':('finds','found',('book','note','letter','map','gem','pen')),
 'keep':('keeps','kept',('book','note','letter','map','gem','pen','permit','diary')),
 'weigh':('weighs','weighed',('gem','carat')),
 'place':('places','placed',('gem','book','pot','tray')),
}


def noun_phrases(nouns):
    out=[]
    for noun in nouns:
        if noun in NAMES or noun in MASS or noun in PLURAL:out.append(noun)
        if noun in COUNT or noun in MASS or noun in PLURAL:
            dets=('the','no','her','his','my')
            if noun in COUNT:dets+=(article(noun),)
            out.extend(d+' '+noun for d in dets)
            for adj in ADJECTIVES:
                ds=('the','no','her','his','my')+((article(adj),) if noun in COUNT else ())
                out.extend(d+' '+adj+' '+noun for d in ds)
                if noun in MASS or noun in PLURAL:out.append(adj+' '+noun)
    return tuple(dict.fromkeys(out))


class TypedGrammar:
    def __init__(self, vocabulary=None):
        vocab=set(vocabulary) if vocabulary is not None else None
        self.paths=[]
        self.path_metadata=[]
        def add(slots,label,tense='present'):
            normalized=[]
            for slot in slots:
                opts=tuple(dict.fromkeys(words(x) for x in slot if vocab is None or set(words(x))<=vocab))
                if not opts:return
                normalized.append(opts)
            self.paths.append((tuple(normalized),label))
            self.path_metadata.append({'tense':tense})
        sg=tuple(NAMES)+noun_phrases(COUNT+MASS)+('he','she')
        # Singular/plural features belong to the head noun, including modified NPs.
        plural=noun_phrases(PLURAL)+('we','they','you')
        nonthird=plural+('i',)
        allsub=sg+nonthird
        for base,(third,past,objects) in VERBS.items():
            obj=noun_phrases(objects)
            for subjects,predicate,tense in [(sg,third,'present'),(nonthird,base,'present'),(allsub,past,'past')]:
                add([subjects,(predicate,),obj],'transitive:'+base,tense)
            add([allsub,('can','will','could'),(base,),obj],'modal:'+base,'modal')
            add([allsub,('did',),(base,),obj],'emphatic:'+base,'past')
            if base=='place':
                for subjects,predicate,tense in [(sg,third,'present'),(nonthird,base,'present'),(allsub,past,'past')]:
                    add([subjects,(predicate,),obj,('inside','in'),noun_phrases(('case','room',))],'location:'+base,tense)
        for subjects,predicate in [(sg,'is'),(plural,'are'),(('i',),'am'),(sg+('i',),'was'),(plural,'were')]:
            tense='past' if predicate in ('was','were') else 'present'
            add([subjects,(predicate,),QUALITIES],'copular-quality',tense)
            add([subjects,(predicate,),noun_phrases(('nurse','writer','sailor','teacher','pilot'))],'copular-np',tense)
        for base,third,past in [('live','lives','lived'),('walk','walks','walked'),('run','runs','ran'),('wait','waits','waited')]:
            for subjects,predicate,tense in [(sg,third,'present'),(nonthird,base,'present'),(allsub,past,'past')]:
                add([subjects,(predicate,)],'intransitive:'+base,tense)
                if base=='live':add([subjects,(predicate,),('on',)],'particle:'+base,tense)
            if base=='live':add([allsub,('did',),(base,),('on',)],'emphatic-particle:'+base,'past')

    @staticmethod
    def _positions(slots, tape, reverse=False):
        # Return a fresh list: callers cannot mutate the cached result.
        return list(TypedGrammar._cached_positions(slots,tape,reverse))

    @staticmethod
    def positions_cache_info():
        return TypedGrammar._cached_positions.cache_info()

    @staticmethod
    @lru_cache(maxsize=8192, typed=True)
    def _cached_positions(slots,tape,reverse):
        return tuple(TypedGrammar._positions_uncached(slots,tape,reverse))

    @staticmethod
    def _positions_uncached(slots, tape, reverse=False):
        ss=slots[::-1] if reverse else slots
        remaining=tape[::-1] if reverse else tape
        states={(0,remaining)};results=[]
        while states:
            i,rest=states.pop()
            if not rest:
                # Boundary immediately before slot i.
                results.append((i,None,0));continue
            if i==len(ss):continue
            for alt in ss[i]:
                a=alt[::-1] if reverse else alt
                common=min(len(a),len(rest))
                if a[:common]!=rest[:common]:continue
                if len(rest)<len(a):results.append((i,alt,len(rest)))
                else:states.add((i+1,rest[len(a):]))
        return results

    @lru_cache(maxsize=4096)
    def complete(self,tape):
        return tuple(i for i,(slots,_) in enumerate(self.paths)
                     if any(p[0]==len(slots) and p[1] is None for p in self._positions(slots,tape)))

    @lru_cache(maxsize=4096)
    def compatible(self,left,right):
        matches=[]
        for i,(slots,_) in enumerate(self.paths):
            lp=self._positions(slots,left);rp=self._positions(slots,right,True)
            ok=False
            for li,la,ln in lp:
                for rev,ra,rn in rp:
                    ri=len(slots)-1-rev
                    if li<=ri:
                        if li<ri or la is None or ra is None or (la==ra and ln+rn<=len(la)):ok=True
                    elif li==ri+1 and la is None and ra is None:ok=True
            if ok:matches.append(i)
        return tuple(matches)

    @lru_cache(maxsize=4096)
    def paragraph(self,tape,max_sentences=8):
        """Segment complete clauses; tokens may come from arbitrary source pieces."""
        @lru_cache(maxsize=None)
        def visit(at,remaining):
            if at==len(tape):return ()
            if not remaining:return None
            for end in range(at+1,len(tape)+1):
                parses=self.complete(tape[at:end])
                if not parses:continue
                tail=visit(end,remaining-1)
                if tail is not None:return ((tape[at:end],parses),)+tail
            return None
        return visit(0,max_sentences) if tape else None

    @lru_cache(maxsize=4096, typed=True)
    def _cached_paragraph_frontier(self,left,right,max_sentences):
        return self._paragraph_frontier_uncached(left,right,max_sentences)

    def frontier_cache_info(self):
        return self._cached_paragraph_frontier.cache_info()

    def paragraph_frontier(self,left,right,max_sentences=3):
        # Exact word tapes, side order, budget and grammar identity are keys.
        # Compiled paths are immutable, as for complete/compatible caches.
        return self._cached_paragraph_frontier(left,right,max_sentences)

    def _paragraph_frontier_uncached(self,left,right,max_sentences=3):
        """Possible completion across sentence boundaries; no whole-block gate."""
        def splits(tape,reverse=False):
            # Strip any number of complete outer clauses, retaining one partial.
            out=[(tape,0)];seen=set(out)
            for rem,n in out:
                if n>=max_sentences:continue
                for size in range(1,len(rem)+1):
                    part=rem[-size:] if reverse else rem[:size]
                    if self.complete(part):
                        child=(rem[:-size] if reverse else rem[size:],n+1)
                        if child not in seen:seen.add(child);out.append(child)
            return out
        for l,ln in splits(left):
            for r,rn in splits(right,True):
                budget=max_sentences-ln-rn
                if not l and not r and 0<ln+rn<=max_sentences:return True
                if budget>=1 and self.compatible(l,r):return True
                if budget>=2 and self.compatible(l,()) and self.compatible((),r):return True
        return False

    def text_paragraph(self,text,max_sentences=8):
        """Explicit punctuation boundaries cannot be repaired by another source."""
        chunks=[words(x) for x in re.split(r'[.!?]+',text) if words(x)]
        if not chunks:return None
        result=[]
        for chunk in chunks:
            ids=self.complete(chunk)
            if not ids or len(result)>=max_sentences:return None
            result.append((chunk,ids))
        return tuple(result)

    def clause_features(self,tape,parse_ids):
        """Inspect one licensed derivation; entity overlap is not coherence proof."""
        derivations=[]
        for tid in parse_ids:
            slots,label=self.paths[tid]
            def match(i,at):
                if i==len(slots):return () if at==len(tape) else None
                for alt in slots[i]:
                    if tape[at:at+len(alt)]==alt:
                        tail=match(i+1,at+len(alt))
                        if tail is not None:return (alt,)+tail
                return None
            constituents=match(0,0)
            obj=constituents[2] if label.startswith(('transitive:','location:','copular-np')) else constituents[3] if label.startswith(('modal:','emphatic:')) else ()
            patient=obj if label.startswith(('transitive:','location:','modal:','emphatic:')) else ()
            location=constituents[-1] if label.startswith('location:') else ()
            pred=tuple(w for x in constituents[1:3] for w in x) if label.startswith(('modal:','emphatic:','emphatic-particle:')) else constituents[1]
            derivations.append({'parse_id':tid,'label':label,'subject':list(constituents[0]),
                'predicate':list(pred),'object':list(obj),'patient':list(patient),
                'location':list(location),'tense':self.path_metadata[tid]['tense']})
        primary=derivations[0];subject=tuple(primary['subject']);obj=tuple(primary['object']);location=tuple(primary['location'])
        names={str(x).lower() for x in NAMES}
        entities={w for w in subject+obj+location if w in set(COUNT+MASS+PLURAL)|names or w in {'i','we','you','he','she','they'}}
        tense_options=sorted({x['tense'] for x in derivations})
        return {**primary,'entities':sorted(entities),'event':primary['label'].split(':')[-1],
                'tense':tense_options[0] if len(tense_options)==1 else 'ambiguous',
                'tense_options':tense_options,'derivations':derivations,
                'grammar_scope':'limited typed grammar; meaning unreviewed'}


@lru_cache(maxsize=1)
def default_grammar():return TypedGrammar()
