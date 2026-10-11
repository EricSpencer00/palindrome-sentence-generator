"""Necessary character-boundary feasibility for a declared finite grammar."""
from itertools import product

from .admission import normalize_letters

BOUNDARY_INDEX_VERSION = 'finite-grammar-boundaries-v1'


def prefix_compatible(a, b):
    return a.startswith(b) or b.startswith(a)


class GrammarBoundaryIndex:
    """First/last clause slot boundaries, independent of source or target outputs.

    A complete paragraph must begin with one compiled first-clause prefix and
    end with one compiled last-clause suffix. Their letter streams must agree
    under reversal. The condition is necessary, not sufficient. Comparing
    characters preserves partial lexical seams, including prefixes inside a
    word; no corpus lookup or full-word admission happens here.
    """
    def __init__(self, grammar):
        prefixes, suffixes = {}, {}
        for path_id, (slots, label) in enumerate(grammar.paths):
            if not slots:
                raise ValueError('nonempty finite grammar paths required')
            for target, selected in ((prefixes, slots[:2]), (suffixes, slots[-2:])):
                for alternatives in product(*selected):
                    text = ' '.join(w for alternative in alternatives for w in alternative)
                    letters = normalize_letters(text)
                    if not letters:
                        raise ValueError('nonempty ASCII-letter constituent boundaries required')
                    target.setdefault(letters, []).append({'path_id': path_id,
                                                          'label': label, 'text': text})
        self.pairs = []
        for prefix in sorted(prefixes):
            for suffix in sorted(suffixes):
                reversed_suffix = suffix[::-1]
                if prefix_compatible(prefix, reversed_suffix):
                    self.pairs.append({'family_id': 'boundary-' + str(len(self.pairs)),
                        'prefix': prefix, 'suffix': suffix,
                        'reversed_suffix': reversed_suffix,
                        'prefix_records': prefixes[prefix], 'suffix_records': suffixes[suffix]})
        self.grammar_path_count = len(grammar.paths)
        self.prefix_count, self.suffix_count = len(prefixes), len(suffixes)

    def matching_families(self, left, right):
        l = normalize_letters(left)
        r = normalize_letters(right)[::-1]
        return tuple(pair['family_id'] for pair in self.pairs
                     if prefix_compatible(l, pair['prefix'])
                     and prefix_compatible(r, pair['reversed_suffix']))

    def allows_text(self, left, right):
        return bool(self.matching_families(left, right))

    def allows_state(self, state):
        return self.allows_text(' '.join(p.text for p in state.left),
                                ' '.join(p.text for p in state.right))

    def receipt(self):
        return {'version': BOUNDARY_INDEX_VERSION,
                'scope': 'Necessary outer-character condition for the declared compiled grammar only',
                'grammar_paths': self.grammar_path_count,
                'prefixes': self.prefix_count, 'suffixes': self.suffix_count,
                'families': self.pairs}
