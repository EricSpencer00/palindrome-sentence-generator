"""Bounded constructive continuation of an existing palindrome frontier.
Run against the current owner repository with PYTHONPATH=. This adds no model
calls and changes no repository source. The candidate was human/model-authored
by debt solving; the grid is a correctness experiment, not a discovery rate.
"""
import argparse
import hashlib
import itertools
import json
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from llm_palindrome.bilateral_seams import (
    Chunk, AllChunkCentersGrammar, OwedSideBilateralGrammar,
    WholeChunkCenterGrammar, debt, norm,
)

TARGET = 'Do geese on a cedar trade canoe see God?'
MATERIALS = ['cedar', 'wooden', 'birch', 'narrow', 'sturdy', 'small']
USES = ['trade', 'cargo', 'touring', 'river', 'rescue', 'fishing']
SOURCE = 'Authored debt-conditioned extension of the existing Do geese / see God? control; no world-originality claim.'

def make_inventory():
    cs = [Chunk('L1', 'Do geese ', 'START', 'SUBJECT_PP',
                'plural subject under question auxiliary do; optional locative PP follows', SOURCE)]
    cs += [Chunk('L2-' + word, 'on a ' + word + ' ', 'SUBJECT_PP', 'CANOE_HEAD',
                 'locative PP opener: preposition on + singular determiner a + material noun or shape adjective', SOURCE)
           for word in MATERIALS]
    cs += [Chunk('R2-' + word, word + ' canoe ', 'CANOE_HEAD', 'BARE_VP',
                 'singular noun compound completing the PP object; modifier describes canoe use', SOURCE)
           for word in USES]
    cs += [Chunk('R1', 'see God?', 'BARE_VP', 'END',
                 'base-form transitive VP under do; object God', SOURCE)]
    return cs

def run(out, reference=None):
    out.mkdir(parents=True, exist_ok=True)
    cs = make_inventory()
    assert all(len(c.text.split()) >= 2 for c in cs)
    attempts = []
    for material, use in itertools.product(MATERIALS, USES):
        text = f'Do geese on a {material} {use} canoe see God?'
        tape = norm(text)
        attempts.append(dict(material=material, use=use, text=text, letters=len(tape), exact=tape == tape[::-1]))
    expected = {norm(a['text']) for a in attempts if a['exact']}
    assert expected == {norm(TARGET)}
    records = {}
    for name, cls in [('whole_chunk_centers', AllChunkCentersGrammar), ('owed_outside_in', OwedSideBilateralGrammar)]:
        result = cls(cs).search(max_steps=4, max_states=10000, max_outputs=100, seconds=2)
        assert result['receipt']['status'] == 'complete_bounded_depth'
        assert {o['tape'] for o in result['outputs']} == expected
        records[name] = result
    by = {c.id: c for c in cs}
    center = WholeChunkCenterGrammar(cs, 'R2-trade', 0, 1)
    s = center.initial()
    trace = [dict(chunk='R2-trade', text=by['R2-trade'].text, side='anchor',
                  center_offset=[0, 1], center_letter='t', debt=s['debt'])]
    for cid, side in [('L2-cedar', 'left'), ('L1', 'left'), ('R1', 'right')]:
        s, error = center.extend(s, by[cid], side)
        assert error is None
        trace.append(s['trace'][-1])
    assert center.closed(s) and center.render(s) == TARGET
    words = re.findall('[a-z]+', TARGET.lower())
    reference_result = None
    if reference:
        raw = json.loads(Path(reference).read_text())
        known = {o['tape'] for run in raw for o in run['outputs']}
        reference_result = dict(reference=str(reference), prior_unique_tapes=len(known), target_present=norm(TARGET) in known,
                                scope='Supplied 018 outputs only. Owner must check the full local corpus before a broader new-to-corpus claim.')
        assert norm(TARGET) not in known
    result = dict(
        target=TARGET, letters=len(norm(TARGET)), words=len(words), distinct_words=len(set(words)),
        repeated_words={w:n for w,n in Counter(words).items() if n>1},
        whole_word_chunks=['Do geese', 'on a cedar', 'trade canoe', 'see God?'],
        source='Known question scaffold: Do geese see God? New authored continuation: on a cedar trade canoe.',
        inherited_scaffold_letters=13, added_letters=18, phrase_is_itself_palindromic=False,
        construction='The frontier leaves debt e. The new PP is onacedartradecanoe; e+PP is palindromic. Its t center lies inside the intact phrase trade canoe.',
        parse='Do [SUBJECT geese [PP on [NP a [MOD cedar] [N-COMPOUND trade canoe]]]] [VP see God]?',
        meaning='A whimsical question about geese perched on a cedar-built canoe used for trade.',
        quality_caution='The compound cedar trade canoe is interpretable but marked and uncommon; no claim that this exact term is conventional or that readers prefer this to the short control. Philosophical whimsy is inherited from the scaffold.',
        novelty_caution='This is a known-scaffold extension, not a new independent palindrome family. No world-originality claim. No unseen full-corpus claim.',
        lexical_evidence=[dict(url='https://repository.si.edu/bitstream/10088/9991/1/USNMB_2301964_unit.pdf',
                              support='Smithsonian historical monograph documents trade canoes and cedar in canoe construction. This supports the component concepts, not attestation of the exact compound.')],
        experiment=dict(kind='one bounded authored grid', path_count=36, inventory_chunks=len(cs),
                        exact_paths=1, all_words_intact=True, paid_calls=0,
                        scoring_note='Authored around a found candidate; not an unbiased yield estimate or a readability evaluation.'),
        reference_comparison=reference_result, center_trace=trace,
        receipts={name:r['receipt'] for name,r in records.items()},
    )
    for filename, value in [('candidate.json', result), ('inventory.json', [asdict(c) for c in cs]),
                            ('attempts.json', attempts), ('engine-results.json', records)]:
        (out / filename).write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='constructive-geese-output')
    parser.add_argument('--reference-018')
    args = parser.parse_args()
    run(Path(args.output), args.reference_018)
