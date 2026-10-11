"""Finite role grammar × palindrome product, without a full-sentence bank.

Each character arc carries its word, role and offset. Forward traversal and
reverse traversal meet inside words or between them. Shared slot endpoints
factor lexical combinations; no completed sentence lookup licenses a path.
"""
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
import time
from .admission import normalize_letters


@dataclass(frozen=True)
class Slot:
    role: str
    words: tuple[str, ...]


@dataclass(frozen=True)
class Frame:
    name: str
    slots: tuple[Slot, ...]
    style: str = 'statement'
    provenance: str = 'authored role grammar'
    known_scaffold: bool = False

    def render(self, words):
        words=list(words)
        if self.style=='quote':
            prefix=' '.join(words[:-1]);return prefix[0].upper()+prefix[1:]+', "'+words[-1]+'."'
        if self.style=='vocative':
            return words[0][0].upper()+words[0][1:]+', '+' '.join(words[1:])+'.'
        text=' '.join(words);return text[0].upper()+text[1:]+'.'


@dataclass(frozen=True)
class Arc:
    source: int
    target: int
    char: str
    sentence: int
    frame: int
    slot: int
    word: str
    role: str
    offset: int


class SearchBudgetExceeded(RuntimeError):
    pass


class GrammarDAG:
    def __init__(self,frames,sentences=1):
        if sentences<1:raise ValueError('positive sentence count required')
        self.frames=tuple(frames);self.sentences=sentences
        self.out=defaultdict(list);self.inc=defaultdict(list)
        self.eps=defaultdict(set);self.reverse_eps=defaultdict(set);self.arcs=[];self.nodes=0
        def node():
            n=self.nodes;self.nodes+=1;return n
        def epsilon(a,b):self.eps[a].add(b);self.reverse_eps[b].add(a)
        self.start=node();boundary=self.start
        for sentence in range(sentences):
            end=node()
            for fi,frame in enumerate(self.frames):
                if not frame.slots:raise ValueError('empty frame')
                current=node();epsilon(boundary,current)
                for si,slot in enumerate(frame.slots):
                    if not slot.words or not slot.role:raise ValueError('slot requires words and role')
                    next_slot=node()
                    for word in slot.words:
                        tape=normalize_letters(word)
                        if not tape:raise ValueError('empty word')
                        if ' ' in word:raise ValueError('slots contain lexical words, not phrases')
                        at=current
                        for offset,char in enumerate(tape):
                            target=node();aid=len(self.arcs)
                            self.arcs.append(Arc(at,target,char,sentence,fi,si,word,slot.role,offset))
                            self.out[at].append(aid);self.inc[target].append(aid);at=target
                        epsilon(at,next_slot)
                    current=next_slot
                epsilon(current,end)
            boundary=end
        self.accept=boundary
        # Reachability is a sound syntactic prune: a product pair must still
        # bound some full grammatical path. No quality or vocabulary pruning.
        successors={n:set(self.eps[n])|{self.arcs[a].target for a in self.out[n]} for n in range(self.nodes)}
        indegree=[0]*self.nodes
        for dests in successors.values():
            for d in dests:indegree[d]+=1
        queue=deque(n for n,d in enumerate(indegree) if d==0);order=[]
        while queue:
            n=queue.popleft();order.append(n)
            for d in successors[n]:
                indegree[d]-=1
                if indegree[d]==0:queue.append(d)
        if len(order)!=self.nodes:raise ValueError('grammar must be acyclic')
        self.reachable=[0]*self.nodes
        for n in reversed(order):
            mask=1<<n
            for d in successors[n]:mask|=self.reachable[d]
            self.reachable[n]=mask

    @lru_cache(maxsize=None)
    def closure(self,node,reverse=False):
        graph=self.reverse_eps if reverse else self.eps
        found={node};queue=[node]
        while queue:
            for next_node in graph[queue.pop()]:
                if next_node not in found:found.add(next_node);queue.append(next_node)
        return frozenset(found)

    @lru_cache(maxsize=None)
    def transitions(self,node,reverse=False):
        graph=self.inc if reverse else self.out
        return tuple(sorted({a for n in self.closure(node,reverse) for a in graph[n]}))

    def lexical_completions(self,prefix,suffix,role):
        """Intersect two partial lexical offsets in a role-licensed domain."""
        p=normalize_letters(prefix);s=normalize_letters(suffix);out=[]
        for word in sorted({a.word for a in self.arcs if a.role==role}):
            tape=normalize_letters(word)
            if tape.startswith(p) and tape.endswith(s) and len(p)+len(s)<=len(tape):
                out.append(dict(word=word,role=role,left_consumed=len(p),right_consumed=len(s),middle=tape[len(p):len(tape)-len(s) if s else len(tape)]))
        return out

    def accepts_words(self,words):
        # Independent token-level validation for completed paths; no sentence
        # bank. Words include punctuation-free lexical surfaces only.
        k=0
        @lru_cache(maxsize=None)
        def parse(sentence,at):
            if sentence==self.sentences:return at==len(words)
            for frame in self.frames:
                n=len(frame.slots)
                if at+n<=len(words) and all(words[at+i] in slot.words for i,slot in enumerate(frame.slots)):
                    if parse(sentence+1,at+n):return True
            return False
        return parse(k,0)

    def materialize(self,path):
        rows=[];all_words=[]
        for sentence in range(self.sentences):
            arcs=[self.arcs[a] for a in path if self.arcs[a].sentence==sentence]
            starts=[a for a in arcs if a.offset==0]
            if not starts:raise AssertionError('missing sentence')
            frame=starts[0].frame
            assert all(a.frame==frame for a in arcs)
            words=[a.word for a in starts];all_words.extend(words)
            spec=self.frames[frame]
            assert len(words)==len(spec.slots)
            assert all(word in slot.words for word,slot in zip(words,spec.slots))
            rows.append(dict(text=spec.render(words),frame=spec.name,frame_index=frame,words=words,roles=[s.role for s in spec.slots],known_scaffold=spec.known_scaffold,provenance=spec.provenance))
        text=' '.join(r['text'] for r in rows);tape=normalize_letters(text)
        assert tape and tape==tape[::-1]
        assert self.accepts_words(tuple(all_words))
        center=len(path)//2
        center_arc=self.arcs[path[center]]
        return dict(text=text,tape=tape,letters=len(tape),sentences=rows,midpoint=dict(word=center_arc.word,role=center_arc.role,offset=center_arc.offset,inside_word=center_arc.offset>0))


def exact_grammar_palindromes(grammar,*,max_work=20000000,max_paths=20000,seconds=30,trace=None):
    """Complete finite product, or raise with its exact incomplete receipt.

    Cached states preserve all path histories. Memoization never collapses
    distinct accepted derivations. Counts are algorithm operations, not a
    speedup comparison to baseline operations with a different cost model.
    """
    start=time.monotonic();deadline=start+seconds
    stats=dict(work=0,states=0,reachability_rejects=0,matched_arc_pairs=0,char_mismatch_pairs=0,partial_word_states=0)
    def budget():
        stats['work']+=1
        if stats['work']>max_work or time.monotonic()>deadline:
            raise SearchBudgetExceeded('product resource bound reached')
    @lru_cache(maxsize=None)
    def solve(left,right):
        budget();stats['states']+=1
        if not (grammar.reachable[left]>>right)&1:
            stats['reachability_rejects']+=1
            if trace:trace(dict(left=left,right=right,reason='no_grammar_path',accepted_derivations=0))
            return ()
        lc=grammar.closure(left);rc=grammar.closure(right,True)
        la=grammar.transitions(left);ra=grammar.transitions(right,True)
        if any(grammar.arcs[a].offset>0 for a in la) or any(grammar.arcs[a].offset>0 for a in ra):stats['partial_word_states']+=1
        results=[]
        if lc&rc:results.append(())
        for a in la:
            if grammar.arcs[a].target in rc:results.append((a,))
        rchars=defaultdict(list)
        for a in ra:rchars[grammar.arcs[a].char].append(a)
        matches=sum(len(rchars[grammar.arcs[a].char]) for a in la)
        stats['char_mismatch_pairs']+=len(la)*len(ra)-matches
        for a in la:
            aa=grammar.arcs[a]
            for b in rchars[aa.char]:
                budget();stats['matched_arc_pairs']+=1;bb=grammar.arcs[b]
                for middle in solve(aa.target,bb.source):
                    results.append((a,)+middle+(b,))
                    if len(results)>max_paths:raise SearchBudgetExceeded('product path bound reached')
        if trace:trace(dict(left=left,right=right,left_arcs=la,right_arcs=ra,char_mismatches=len(la)*len(ra)-matches,matched_arc_pairs=matches,accepted_derivations=len(results),reason='completed_state'))
        return tuple(results)
    try:
        paths=solve(grammar.start,grammar.accept)
    except SearchBudgetExceeded as exc:
        stats.update(complete=False,elapsed_seconds=time.monotonic()-start,reason=str(exc),cached_states=solve.cache_info().currsize)
        exc.receipt=stats;raise
    stats.update(complete=True,elapsed_seconds=time.monotonic()-start,accepted_derivations=len(paths),cached_states=solve.cache_info().currsize,nodes=grammar.nodes,character_arcs=len(grammar.arcs))
    return paths,stats
