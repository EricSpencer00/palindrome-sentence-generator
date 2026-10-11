"""Finite-grammar boundary certificate; no search or model invocation."""
import collections
import hashlib
import itertools
import json
from pathlib import Path

from experiments.block_seam_comparison_20261009 import build_config
from llm_palindrome.admission import normalize_letters
from llm_palindrome.typed_constituents import TypedGrammar, words


def boundary_certificate(grammar):
    prefixes, suffixes = {}, {}
    for path_id, (slots, label) in enumerate(grammar.paths):
        if len(slots) < 2:
            raise ValueError('certificate requires at least two constituent slots per path')
        for target, selected in ((prefixes, slots[:2]), (suffixes, slots[-2:])):
            for alternatives in itertools.product(*selected):
                text = ' '.join(w for alternative in alternatives for w in alternative)
                target.setdefault(text, []).append({'path_id': path_id, 'label': label})
    compatible = []
    mismatch = collections.Counter()
    for left, left_paths in prefixes.items():
        a = normalize_letters(left)
        for right, right_paths in suffixes.items():
            b = normalize_letters(right)[::-1]
            if a.startswith(b) or b.startswith(a):
                compatible.append({'left_prefix': left, 'right_suffix': right,
                    'prefix_paths': left_paths, 'suffix_paths': right_paths})
            else:
                mismatch[next(i + 1 for i, pair in enumerate(zip(a, b))
                              if pair[0] != pair[1])] += 1
    return {'grammar_productions': len(grammar.paths),
        'distinct_prefixes': len(prefixes), 'distinct_suffixes': len(suffixes),
        'pairs_checked': len(prefixes) * len(suffixes),
        'compatible_pairs': compatible, 'first_mismatch_position_counts': dict(mismatch),
        'prefix_catalogue': prefixes, 'suffix_catalogue': suffixes}


def run():
    controls = []
    for text in ('Liam sees mail.', 'No rider sees red iron.'):
        # Positive controls test orientation and grammar parsing; they are not pilot discoveries.
        grammar = TypedGrammar(set(words(text)))
        assert grammar.paragraph(words(text), 4) is not None
        assert normalize_letters(text) == normalize_letters(text)[::-1]
        check = boundary_certificate(grammar)
        assert check['compatible_pairs']
        controls.append({'text': text, 'compatible_pairs': len(check['compatible_pairs'])})
    config, grammar = build_config()
    report = boundary_certificate(grammar)
    report.update(schema_version=1, vocabulary=config['words'],
        source_hashes=config['source_hashes'],
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        positive_controls=controls,
        method='Exhaustive first-two-slot prefix versus reversed final-two-slot suffix check across all compiled grammar paths.',
        proof='Every complete paragraph begins with a first-clause prefix and ends with a last-clause suffix in these catalogues. A global palindrome requires equality of their normalized streams over their common length. Zero compatible pairs excludes every complete paragraph in this finite compiled grammar, for any number of middle clauses.',
        scope='The declared 38-word, nine-production grammar only; no statement about general English or expanded lexicons.')
    out = Path(__file__).resolve().parents[1] / 'research/block-seams/depth-diagnostic-001-boundary-gate.json'
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('grammar_productions', 'distinct_prefixes',
        'distinct_suffixes', 'pairs_checked', 'compatible_pairs', 'positive_controls')}, indent=2))


if __name__ == '__main__':
    run()
