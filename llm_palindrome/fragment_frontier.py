"""Finite grammatical sentences with source offsets and partial lexical states."""
from dataclasses import dataclass
from functools import lru_cache
from .admission import normalize_letters

@dataclass(frozen=True)
class LexicalPosition:
    frame:str
    token_index:int
    expected_word:str
    consumed_in_word:int
    remaining_in_word:str
    argument_role:str
    complete:bool

class FragmentFrontier:
    def __init__(self,entries):
        self.entries=tuple(entries)
        for entry in self.entries:
            if not entry['frame'] or not entry['tokens'] or len(entry['tokens'])!=len(entry['roles']):
                raise ValueError('frame, tokens and one role per token are required')
            if any(not normalize_letters(word) for word in entry['tokens']):
                raise ValueError('empty lexical token')
            if normalize_letters(entry['text'])!=''.join(normalize_letters(word) for word in entry['tokens']):
                raise ValueError('rendered text does not match licensed token tape')
        self.compiled=tuple((normalize_letters(e['text']),tuple(e['tokens']),tuple(e['roles']),e['frame']) for e in entries)

    @lru_cache(maxsize=4096)
    def positions(self,prefix):
        """Word prefixes stay live only when a full licensed path completes them."""
        tape=normalize_letters(prefix);out=set()
        for full,tokens,roles,frame in self.compiled:
            if not full.startswith(tape):continue
            if len(tape)==len(full):
                out.add(LexicalPosition(frame,len(tokens),'',0,'','complete',True));continue
            at=0
            for i,(word,role) in enumerate(zip(tokens,roles)):
                next_at=at+len(normalize_letters(word))
                if len(tape)<next_at:
                    offset=len(tape)-at;w=normalize_letters(word)
                    out.add(LexicalPosition(frame,i,w,offset,w[offset:],role,False));break
                at=next_at
        return tuple(sorted(out,key=lambda p:(p.frame,p.token_index,p.expected_word,p.consumed_in_word)))

    def complete(self,text):
        return any(p.complete for p in self.positions(text))
