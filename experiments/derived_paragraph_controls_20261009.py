"""Parent-authored cross-block controls; representation checks, not discovery."""
import hashlib
import json
from pathlib import Path

from experiments.block_boundary_gate_20261009 import boundary_certificate
from experiments.block_seam_comparison_20261009 import build_config, render_paragraph, WordAdditiveScorer
from llm_palindrome.admission import normalize_letters
from llm_palindrome.block_seams import Piece, Seam
from llm_palindrome.block_search import BlockUnit, block_beam_search, compatible_actions
from llm_palindrome.typed_constituents import TypedGrammar, words

CANDIDATES = (
    'No rider sees mail. Liam sees red iron.',
    'No rider sees mail. Leon sees red iron. No rider sees Noel. Liam sees red iron.',
)
UNITS = ('no rider', 'sees', 'mail', 'leon', 'red iron', 'noel', 'liam')
PATH = (('left', 'no rider'), ('right', 'red iron'),
        ('left', 'sees'), ('right', 'sees'), ('left', 'mail'), ('right', 'liam'),
        ('left', 'leon'), ('right', 'noel'), ('left', 'sees'), ('right', 'sees'),
        ('left', 'red iron'), ('right', 'no rider'))


def tapes(state):
    return (tuple(w for p in state.left for w in words(p.text)),
            tuple(w for p in state.right for w in words(p.text)))


def check_representation(grammar):
    inventory = tuple(BlockUnit(f'control-unit-{i}', unit,
        ({'source': 'parent-authored derived controls', 'text': unit,
          'role': 'supplied construction ingredient'},)) for i, unit in enumerate(UNITS))
    by_text = {u.text: u for u in inventory}
    def frontier(state):
        left, right = tapes(state)
        return grammar.paragraph_frontier(left, right, 4)
    def linguistic_gate(state):
        left, right = tapes(state)
        if not frontier(state):
            return False
        return (grammar.paragraph(left + right, 4) is not None or bool(
            compatible_actions(state, inventory, grammar_accept=frontier,
                               max_letters=119, max_words=64)))
    state = Seam()
    states = [state]
    trace = []
    for depth, (side, text) in enumerate(PATH, 1):
        menu = compatible_actions(state, inventory, grammar_accept=linguistic_gate,
                                  max_letters=119, max_words=64)
        selected = next((a for a in menu if a.side == side and a.unit.text == text), None)
        assert selected is not None, (depth, side, text)
        state = selected.child
        states.append(state)
        trace.append({'depth': depth, 'side': side, 'unit': text,
            'actual_linguistically_valid_menu': [(a.side, a.unit.text) for a in menu],
            'left': [p.text for p in state.left], 'right': [p.text for p in state.right],
            'debt': state.debt(), 'letters': len(normalize_letters(state.text()))})
    assert normalize_letters(state.text()) == normalize_letters(CANDIDATES[1])
    assert state.exact()
    allowed = set(states[1:])
    def closure(state):
        left, right = tapes(state)
        parsed = grammar.paragraph(left + right, 4)
        rendered = render_paragraph(parsed) if parsed else None
        n = normalize_letters(state.text())
        return (state.exact() and 60 <= len(n) <= 119 and parsed is not None
            and 2 <= len(parsed) <= 4
            and len({tuple(w) for w, _ in parsed}) == len(parsed)
            and grammar.text_paragraph(rendered, 4) is not None)
    # Stronger path-membership restriction isolates representation. It does not relax grammar.
    result = block_beam_search(inventory, WordAdditiveScorer(UNITS),
        grammar_accept=lambda state: state in allowed and linguistic_gate(state),
        allow_closed=closure, beam_width=64, max_steps=12, max_actions=10000,
        min_letters=60, max_letters=119, max_words=64, seed=921, diversity=0)
    eligible = [t for t in result['terminals'] if t['eligible']]
    assert len(eligible) == 1
    assert not result['truncated']
    assert result['attempted_actions'] == 168
    left, right = tapes(eligible[0]['state'])
    rendered = render_paragraph(grammar.paragraph(left + right, 4))
    assert rendered == CANDIDATES[1]
    return {'scope': 'Known construction path with an explicit stronger path-membership gate; not autonomous discovery or general search coverage.',
        'path_steps': 12, 'inventory_units': list(UNITS), 'beam_width': 64,
        'max_actions': 10000, 'attempted_actions': result['attempted_actions'],
        'truncated': result['truncated'], 'bounded_horizon_status': result['status'],
        'rendered_terminal': rendered, 'eligible_terminals': len(eligible), 'trace': trace}


def run():
    config, restricted = build_config()
    full = TypedGrammar()
    declared_vocabulary = set(config['words']) | set(words(CANDIDATES[1]))
    expanded = TypedGrammar(declared_vocabulary)
    rows = []
    for text in CANDIDATES:
        n = normalize_letters(text)
        parsed = full.text_paragraph(text, 4)
        assert parsed is not None
        assert restricted.text_paragraph(text, 4) is None
        assert expanded.text_paragraph(text, 4) is not None
        assert n == n[::-1]
        rows.append({'exact_text': text, 'letters': len(n),
            'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'normalized_sha256': hashlib.sha256(n.encode()).hexdigest(),
            'global_exact': True, 'default_grammar_clause_count': len(parsed),
            'distinct_clauses': len({tuple(w) for w, _ in parsed}) == len(parsed),
            'pilot_grammar_accepted': False,
            'missing_pilot_vocabulary': sorted(set(words(text)) - set(config['words'])),
            'declared_union_grammar_accepted': True,
            'clause_features': [full.clause_features(w, ids) for w, ids in parsed],
            'provenance': 'Parent-authored derived control, cross-matching approved noun-phrase pairs and reversing paired clause order.',
            'novelty': 'unverified', 'coherence': 'unreviewed; repetitive seeing predicates and weak discourse connections',
            'human_rating': None, 'held_out_evidence': False})
    for negative in ('No rider see mail.', 'Liam sees a mail.'):
        assert full.text_paragraph(negative, 4) is None
    restricted_gate = boundary_certificate(restricted)
    expanded_gate = boundary_certificate(expanded)
    assert not restricted_gate['compatible_pairs']
    assert expanded_gate['compatible_pairs']
    report = {'schema_version': 1,
        'base_commit': 'd3d92dc132eeb9181e7f5f059b8aa572322ea090',
        'role': 'Constructive representation and restricted-inventory coverage regression',
        'candidates': rows,
        'pair_recipe': [{'left_np': a, 'right_np': b,
                         'normalized_reverse_pair': normalize_letters(a) == normalize_letters(b)[::-1]}
                        for a, b in [('No rider', 'red iron'), ('Liam', 'mail'), ('Leon', 'Noel')]],
        'restricted_boundary_gate': {k: restricted_gate[k] for k in ('grammar_productions',
            'distinct_prefixes', 'distinct_suffixes', 'pairs_checked', 'compatible_pairs')},
        'declared_union_boundary_gate': {k: expanded_gate[k] for k in ('grammar_productions',
            'distinct_prefixes', 'distinct_suffixes', 'pairs_checked', 'compatible_pairs')},
        'vocabulary_additions': sorted(declared_vocabulary - set(config['words'])),
        'grammar_rule_changes': [],
        'linguistic_basis': 'Existing singular no+noun NP agreement, named singular subjects/objects, and bare or modified mass-noun objects already license these clauses.',
        'representation': check_representation(expanded),
        'interpretation': 'The boundary certificate correctly excludes the old 38-word grammar. These supplied controls are admitted by existing rules once missing vocabulary and NP block actions are declared. This is a vocabulary/inventory coverage limitation, not a general English impossibility result.',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'core_source_hashes': config['source_hashes']}
    out = Path(__file__).resolve().parents[1] / 'research/block-seams/derived-paragraph-controls-001.json'
    if out.exists():
        raise RuntimeError('preserve existing regression receipt')
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'candidates': [{k: x[k] for k in ('letters',
        'default_grammar_clause_count', 'distinct_clauses', 'pilot_grammar_accepted',
        'missing_pilot_vocabulary')} for x in rows],
        'restricted_boundary_pairs': len(restricted_gate['compatible_pairs']),
        'declared_union_boundary_pairs': len(expanded_gate['compatible_pairs']),
        'representation': {k: report['representation'][k] for k in ('path_steps',
            'attempted_actions', 'truncated', 'rendered_terminal')}}, indent=2))


if __name__ == '__main__':
    run()
