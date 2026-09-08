"""Exact finite-language mirror intersection, without word search or ranking.

Templates are proposal provenance, never a claim of semantic quality. Controls
are evaluated separately and cannot enter the generated inventory. Run with
python -m experiments.sentence_intersection --out runs/sentence_intersection.json
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import product
import json
from pathlib import Path
import re
import time


def letters(text):
    return ''.join(c for c in text.lower() if c.isalpha())


def repetition_ok(text):
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())
    return (bool(words) and all(a != b for a, b in zip(words, words[1:]))
            and max(Counter(words).values()) <= 3
            and max(Counter(zip(words, words[1:])).values(), default=0) <= 2)


def intersect(rows):
    """Exhaustive lookup preserves different spacings of the same letter run.

    Self-palindromes are reported separately, not admitted as distinct pairs.
    Each unordered pair occurs once. Input rows carry text and source fields.
    """
    index = defaultdict(dict)
    rejected = 0
    for row in rows:
        if not repetition_ok(row['text']):
            rejected += 1
            continue
        key = letters(row['text'])
        index[key][row['text'].lower()] = row
    pairs, centres = [], []
    for key in sorted(index):
        if key == key[::-1]:
            centres.extend(index[key].values())
        elif key < key[::-1] and key[::-1] in index:
            for left, right in product(index[key].values(), index[key[::-1]].values()):
                if repetition_ok(left['text'] + ' ' + right['text']):
                    assert letters(left['text']) == letters(right['text'])[::-1]
                    pairs.append({'left': left, 'right': right})
    # Necessary boundary compatibility, not an estimate of quality.
    boundary = {}
    for size in (1, 2, 3, 4, 8):
        starts = {key[:size] for key in index if len(key) >= size}
        boundary[str(size)] = sum(
            len(key) >= size and key[-size:][::-1] in starts
            for key in index)
    return {'unique_letters': len(index), 'unique_sentences': sum(map(len, index.values())),
            'repetition_rejected': rejected, 'pairs': pairs, 'centres': centres,
            'reverse_prefix_compatible': boundary}


def inventory():
    """Finite, typed clauses: agreement and verb complements chosen explicitly.

    Slots are intentionally inspectable. No proper names, acronym escape hatches,
    POS model, language model, corpus attestation or learned score is involved.
    """
    people = ['i', 'we', 'they', 'the men', 'the women', 'the children',
              'the sailors', 'the workers', 'the guards', 'the visitors']
    things = ['the map', 'a note', 'the door', 'the room', 'the boat', 'a rope',
              'the road', 'a red car', 'the river', 'the gate', 'the wall',
              'the old house', 'the food', 'the water', 'the light', 'the bag']
    locations = ['at home', 'on the road', 'by the river', 'in the room',
                 'near the gate', 'on a boat', 'in a garden', 'in the rain',
                 'at sea', 'in town', 'on land', 'under a tree']
    times = ['', 'today', 'again', 'at dawn', 'last night', 'before noon']
    specs = [
        ('past_transitive', [people + ['he', 'she', 'the man', 'the woman'],
          ['saw', 'found', 'lost', 'moved', 'left', 'wanted', 'needed', 'remembered'], things, times]),
        ('past_motion', [people + ['he', 'she'],
          ['waited', 'sat', 'stood', 'slept', 'stayed', 'rested'], locations, times]),
        ('plural_present', [people[1:], ['wait', 'sit', 'stand', 'sleep', 'stay', 'rest'], locations, times]),
        ('singular_present', [['he', 'she', 'the man', 'the woman', 'the child'],
          ['waits', 'sits', 'stands', 'sleeps', 'stays', 'rests'], locations, times]),
        ('imperative', [['find', 'move', 'leave', 'remember', 'bring', 'take'], things, times]),
        ('modal_transitive', [people + ['he', 'she'], ['can', 'must', 'will', 'may'],
          ['find', 'move', 'leave', 'remember', 'bring', 'take'], things, times]),
        ('negative_motion', [people + ['he', 'she'], ['did not', 'could not'],
          ['wait', 'sit', 'stand', 'sleep', 'stay', 'rest'], locations, times]),
    ]
    for source, slots in specs:
        for parts in product(*slots):
            yield {'text': ' '.join(p for p in parts if p), 'source': source}


def controls():
    positive = intersect([{'text': t, 'source': 'catalogue_control'} for t in
                          ['Go hang a salami', "I'm a lasagna hog"]])
    centre = intersect([{'text': 'Items draw award', 'source': 'compressed_control'}])
    cycle = intersect([{'text': 'do do do', 'source': 'cycle_control'}])
    negative = intersect([{'text': t, 'source': 'negative_control'} for t in
                          ['Go hang a salami', "I'm a lasagna dog"]])
    passed = (len(positive['pairs']) == 1 and centre['unique_sentences'] == 1
              and cycle['repetition_rejected'] == 1 and not negative['pairs'])
    return {'passed': passed, 'known_pair_recovered': len(positive['pairs']),
            'compressed_retained': centre['unique_sentences'],
            'cycle_rejected': cycle['repetition_rejected'],
            'one_letter_negative_pairs': len(negative['pairs'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    calibration = controls()
    if not calibration['passed']:
        raise RuntimeError('mechanical controls failed')
    rows = list(inventory())
    result = intersect(rows)
    result.update(controls=calibration, proposed=len(rows),
                  by_template=dict(Counter(r['source'] for r in rows)),
                  examples=[next(r for r in rows if r['source'] == source)
                            for source in dict.fromkeys(r['source'] for r in rows)],
                  elapsed_seconds=time.monotonic() - start,
                  promoted=False,
                  limitation='Finite template coverage only; no semantic calibration or novelty claim.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
