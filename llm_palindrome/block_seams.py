"""Finite grammar-block prototype with exact residual cancellation.

Let A=N(left) and B=reverse(N(right)). A viable state has A=P u,B=P
(owner left), or A=P,B=P u (owner right), or A=B (empty debt).
Appending a left fragment appends its tape to A. Prepending a right
fragment appends its reversed tape to B. Cancel their common prefix;
a differing character is irreversible under inner growth. At closure,
left+right is palindromic iff its unmatched tape u is palindromic.
Grammar and discourse checks are independent of this algebraic invariant.
"""
from dataclasses import dataclass
from .admission import normalize_letters


@dataclass(frozen=True)
class Piece:
    sentence_id: str
    index: int
    text: str


@dataclass(frozen=True)
class Seam:
    left: tuple[Piece, ...] = ()
    right: tuple[Piece, ...] = ()

    def tapes(self):
        return normalize_letters(' '.join(p.text for p in self.left)), normalize_letters(' '.join(p.text for p in self.right))[::-1]

    def debt(self):
        a,b=self.tapes()
        for i,(x,y) in enumerate(zip(a,b)):
            if x!=y:return dict(viable=False,mismatch_index=i,left_character=x,right_character=y,common_prefix=a[:i])
        shared=min(len(a),len(b));u=a[shared:] or b[shared:]
        return dict(viable=True,owner='left' if len(a)>len(b) else 'right' if len(b)>len(a) else None,
                    residual=u,common_prefix=a[:shared])

    def add(self,side,piece):
        if side not in {'left','right'}:raise ValueError('unknown side')
        child=Seam(self.left+(piece,),self.right) if side=='left' else Seam(self.left,(piece,)+self.right)
        if not child.debt()['viable']:return None
        return child

    def exact(self):
        d=self.debt()
        derived=d['viable'] and bool(normalize_letters(self.text())) and d['residual']==d['residual'][::-1]
        tape=normalize_letters(self.text())
        full_check=bool(tape) and tape==tape[::-1]
        if derived!=full_check:raise AssertionError('residual invariant disagrees with global reversal')
        return full_check

    def text(self):return ' '.join(p.text for p in self.left+self.right)


def legacy_source_bound_grammar_complete(pieces,bank):
    """Complete, contiguous inventory parses; no invented tokens or repairs."""
    i=0;ids=[]
    while i<len(pieces):
        sid=pieces[i].sentence_id
        if sid not in bank:return False,[]
        expected=bank[sid]['parts']
        chunk=pieces[i:i+len(expected)]
        if len(chunk)!=len(expected) or any(p.sentence_id!=sid or p.index!=j or p.text!=expected[j]
                                          for j,p in enumerate(chunk)):return False,[]
        ids.append(sid);i+=len(expected)
    return bool(ids),ids


def grammar_complete(pieces,bank):
    """Compositional text grammar; provenance is never an admission condition.

    Explicit bank declarations remain usable as legacy grammar controls by
    text alone. They are not inferred general English grammaticality.
    """
    from .typed_constituents import default_grammar,words
    tape=words(' '.join(p.text for p in pieces));g=default_grammar()
    parsed=g.text_paragraph(' '.join(p.text for p in pieces))
    if parsed is not None:
        ids=[]
        for clause,_ in parsed:
            sid=next((sid for sid,v in bank.items() if words(' '.join(v['parts']))==clause),None)
            ids.append(sid or 'typed:'+''.join(clause))
        return True,ids
    # Retain explicit historical declaration fixtures, independent of piece IDs.
    import re
    declared={words(' '.join(v['parts'])):sid for sid,v in bank.items()}
    chunks=[words(x) for x in re.split(r'[.!?]+',' '.join(p.text for p in pieces)) if words(x)]
    if not chunks or any(chunk not in declared for chunk in chunks):return False,[]
    return True,[declared[chunk] for chunk in chunks]


def discourse_gate(ids,bank):
    """Necessary topic linkage only, not a meaning/readability certificate."""
    if len(ids)<2:return False
    return all(set(bank.get(a,{}).get('entities',[]))&set(bank.get(b,{}).get('entities',[])) for a,b in zip(ids,ids[1:]))


def paragraph_gates(state,bank,*,require_nonpalindromic_blocks=False,require_unique_sentences=False):
    from .typed_constituents import default_grammar,words
    complete,ids=grammar_complete(state.left+state.right,bank)
    g=default_grammar();parsed=g.text_paragraph(state.text())
    clause_texts=[' '.join(clause) for clause,_ in parsed] if parsed is not None else [' '.join(bank[s]['parts']) for s in ids if s in bank]
    features=[g.clause_features(clause,tids) for clause,tids in parsed] if parsed is not None else []
    nonpalindromic=complete and all((t:=normalize_letters(text))!=t[::-1] for text in clause_texts)
    unique=complete and len(set(ids))==len(ids)
    return dict(exact_palindrome=state.exact(),sentence_grammar_complete=complete,
                experiment_options_pass=complete and (not require_nonpalindromic_blocks or nonpalindromic) and (not require_unique_sentences or unique),
                independent_nonpalindromic_blocks=nonpalindromic,
                no_duplicated_sentences=complete and len(set(ids))==len(ids),
                discourse_topic_linkage=complete and (discourse_gate(ids,bank) or (len(features)>=2 and all(set(a['entities'])&set(b['entities']) for a,b in zip(features,features[1:])))),
                clause_features=features,source_lineages=sorted({p.sentence_id for p in state.left+state.right}),
                coherence_assessment='unreviewed; entity linkage and tense metadata are necessary diagnostics only',
                human_coherence_verified=False)
