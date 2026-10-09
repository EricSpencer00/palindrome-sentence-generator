"""Source-anchored full/partial palindromic islands, not arbitrary strings."""
from dataclasses import dataclass
import hashlib
from .admission import normalize_letters


def hash_text(text):return hashlib.sha256(text.encode()).hexdigest()


def full_island(block_id,text,source,grammar,entities=(),relation=None,known_control=True):
    tape=normalize_letters(text)
    if not tape or tape!=tape[::-1]:raise ValueError('full island must independently reverse exactly')
    return dict(id=block_id,status='full',text=text,letters=len(tape),text_sha256=hash_text(text),
      normalized_sha256=hash_text(tape),core=tape,core_span=[0,len(tape)],left_edge='',right_edge='',
      source=source,grammar=grammar,entities=list(entities),relation=relation,known_control=known_control,
      completion_requirements=[],human_readability_verified=False)


def partial_island(block_id,parent,prefix_or_span,source,grammar,requirements,min_core=4):
    """A contiguous original surface span covering the parent's mirror center.

    Its maximal parent-centered core must have >=min_core letters and occupy
    >=25% of the span. At least one nonempty unmatched edge is required.
    Source substring offsets and grammar completion requirements are retained.
    This definition cannot promote an arbitrary unrelated string as partial.
    """
    a,b=prefix_or_span
    if not 0<=a<b<=len(parent):raise ValueError('invalid surface span')
    parent_tape=normalize_letters(parent)
    if not parent_tape or parent_tape!=parent_tape[::-1]:raise ValueError('parent must be verified exact')
    text=parent[a:b];tape=normalize_letters(text)
    start=len(normalize_letters(parent[:a]));end=start+len(tape);n=len(parent_tape)
    mirrored_start=n-end;mirrored_end=n-start
    core_start=max(start,mirrored_start);core_end=min(end,mirrored_end)
    core=parent_tape[core_start:core_end]
    if core_end<=core_start or len(core)<min_core or 4*len(core)<len(tape):
        raise ValueError('no sufficiently large source-centered palindromic core')
    if core!=core[::-1]:raise AssertionError('source-centered core is not palindromic')
    left=parent_tape[start:core_start];right=parent_tape[core_end:end]
    if not left and not right:raise ValueError('this is a full island, not a partial')
    if not requirements:raise ValueError('partial needs explicit grammar/lexical completion requirements')
    return dict(id=block_id,status='partial',text=text,letters=len(tape),text_sha256=hash_text(text),
      normalized_sha256=hash_text(tape),core=core,core_span=[core_start-start,core_end-start],
      left_edge=left,right_edge=right,mirror_completion_left=right[::-1],mirror_completion_right=left[::-1],
      source=dict(locator=source,parent_text=parent,parent_text_sha256=hash_text(parent),
                  parent_normalized_sha256=hash_text(parent_tape),surface_span=[a,b],normalized_span=[start,end]),
      grammar=grammar,completion_requirements=requirements,known_control=True,human_readability_verified=False)


def lexical_atom(block_id,text,source,source_words,grammar_slot,requirements):
    """A real lexical unit may be entirely unmatched debt, with no palindromic core."""
    if not text.isascii() or not text.isalpha() or text.lower() not in source_words:
        raise ValueError('lexical atom must be a real word in the retained source')
    if not grammar_slot or not requirements:raise ValueError('lexical grammar interface required')
    tape=normalize_letters(text)
    return dict(id=block_id,status='lexical_atom',grain='word',text=text,letters=len(tape),
      text_sha256=hash_text(text),normalized_sha256=hash_text(tape),core='',left_edge=tape,right_edge='',
      source=source,grammar_slot=grammar_slot,completion_requirements=requirements,
      orientation_note='On left append N(text); on right prepend and match reverse(N(text)). Core is not required.',
      human_readability_verified=False)
