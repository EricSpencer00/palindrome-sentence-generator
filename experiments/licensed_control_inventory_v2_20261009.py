"""Systematic licensed control components; frozen grammar rules unchanged."""
import hashlib
import json
from pathlib import Path

from experiments import block_seam_comparison_20261009 as pilot
from experiments.block_boundary_gate_20261009 import boundary_certificate
from experiments.derived_paragraph_controls_20261009 import CANDIDATES, PATH, UNITS, tapes
from llm_palindrome.admission import normalize_letters
from llm_palindrome.block_search import BlockUnit, block_beam_search, compatible_actions
from llm_palindrome.block_seams import Seam
from llm_palindrome.typed_constituents import TypedGrammar, words

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/block-seams'
VERSION = 'licensed-control-components-v2'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_inventory():
    original, _ = pilot.build_config()
    unrestricted = TypedGrammar()
    licensed_phrases = {alternative for slots, _ in unrestricted.paths
                        for slot in slots for alternative in slot}
    licensed_words = {w for phrase in licensed_phrases for w in phrase}
    sources = []
    def add(record_id, text, source, **extra):
        sources.append({'id': record_id, 'text': text, 'source': source,
                        'words': list(words(text)), **extra})
    bank_path = pilot.BASE / 'palindrome-island-bank-001.json'
    bank = json.loads(bank_path.read_text())
    for kind in ('full_islands', 'partial_islands'):
        for record in bank[kind]:
            add('bank:' + record['id'], record['text'], record['source'],
                known_control=kind == 'full_islands', source_anchored_partial=kind == 'partial_islands')
    readable_path = ROOT / 'data/readable_palindrome_centres.json'
    for record in json.loads(readable_path.read_text()):
        add('catalogue:' + record['id'], record['text'],
            'data/readable_palindrome_centres.json#' + record['id'], known_control=True)
    calibration_path = OUT / 'human-calibration.json'
    for i, record in enumerate(json.loads(calibration_path.read_text())['ratings']):
        extra = {'known_control': True, 'human_calibration': True}
        if record['text'] == 'No rider sees red iron.':
            extra['verified_reference'] = 'https://mockok.com/n/'
        add('calibration:' + str(i), record['text'],
            'research/block-seams/human-calibration.json#ratings/' + str(i), **extra)
    add('known:leon-noel', 'Leon sees Noel.', 'https://mockok.com/i-l/',
        known_control=True, verified_reference='https://mockok.com/i-l/',
        human_calibration=False)
    vocabulary = set(original['words'])
    blocks = set(original['blocks'])
    provenance = {unit: list(records) for unit, records in original['unit_provenance'].items()}
    for source in sources:
        tokens = tuple(source['words'])
        vocabulary.update(set(tokens) & licensed_words)
        for start in range(len(tokens)):
            for stop in range(start + 1, len(tokens) + 1):
                phrase = tokens[start:stop]
                if not set(phrase) <= licensed_words:
                    continue
                if len(phrase) != 1 and phrase not in licensed_phrases:
                    continue
                unit = ' '.join(phrase)
                blocks.add(unit)
                provenance.setdefault(unit, []).append({'source_id': source['id'],
                    'source': source['source'], 'source_text_sha256':
                    hashlib.sha256(source['text'].encode()).hexdigest(),
                    'word_span': [start, stop], 'text': unit,
                    'component_kind': 'licensed word' if len(phrase) == 1 else 'licensed constituent',
                    'known_control_source': source.get('known_control', False)})
    # Add every newly licensed token to the action inventory, rather than relying on whole phrases.
    assert vocabulary - set(original['words']) <= blocks
    exclusions = set(original['excluded_normalized'])
    exclusions.update(normalize_letters(s['text']) for s in sources if s.get('known_control'))
    exclusions.update(normalize_letters(text) for text in CANDIDATES)
    grammar = TypedGrammar(vocabulary)
    inventory = {'schema_version': 2, 'version': VERSION,
        'derivation': 'Existing pilot inventory plus every licensed word and contiguous typed-slot constituent from all textual full/partial bank records, readable catalogue, human calibration controls and externally verified Leon/Noel control.',
        'grammar_rule_changes': [], 'original_word_count': len(original['words']),
        'original_block_count': len(original['blocks']), 'words': sorted(vocabulary),
        'blocks': sorted(blocks), 'added_words': sorted(vocabulary - set(original['words'])),
        'added_blocks': sorted(blocks - set(original['blocks'])),
        'source_tokens_not_added_as_new_licensed_components': sorted(
            {w for source in sources for w in source['words']} - licensed_words),
        'unit_provenance': provenance, 'source_records': sources,
        'excluded_normalized': sorted(exclusions),
        'normalized_only_catalogue_policy': 'Existing normalized catalogue remains an exclusion list; no guessed word segmentation.',
        'derived_controls_policy': 'Supplied derived paragraphs are regressions and exclusions, not inventory source texts or novel outcomes.',
        'source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in
            (bank_path, readable_path, calibration_path, Path(__file__),
             ROOT / 'llm_palindrome/typed_constituents.py',
             ROOT / 'llm_palindrome/block_search.py', ROOT / 'data/known_palindromes.json')}}
    return inventory, grammar, original


def represent(inventory, grammar):
    units = tuple(BlockUnit('v2-unit-' + str(i), text,
        tuple(inventory['unit_provenance'].get(text, []))) for i, text in enumerate(inventory['blocks']))
    assert set(UNITS) <= set(inventory['blocks'])
    def frontier(state):
        left, right = tapes(state)
        return grammar.paragraph_frontier(left, right, 4)
    def linguistic_gate(state):
        left, right = tapes(state)
        return frontier(state) and (grammar.paragraph(left + right, 4) is not None
            or bool(compatible_actions(state, units, grammar_accept=frontier,
                                       max_letters=119, max_words=64)))
    state = Seam()
    allowed = []
    trace = []
    for depth, (side, text) in enumerate(PATH, 1):
        menu = compatible_actions(state, units, grammar_accept=linguistic_gate,
                                  max_letters=119, max_words=64)
        action = next((a for a in menu if a.side == side and a.unit.text == text), None)
        assert action is not None, (depth, side, text)
        state = action.child
        allowed.append(state)
        trace.append({'depth': depth, 'side': side, 'unit': text,
                      'debt': state.debt(), 'source_records': list(action.unit.provenance)})
    permitted = set(allowed)
    def closure(state):
        left, right = tapes(state)
        parsed = grammar.paragraph(left + right, 4)
        return (state.exact() and parsed is not None and len(parsed) == 4
            and len({tuple(w) for w, _ in parsed}) == 4
            and grammar.text_paragraph(pilot.render_paragraph(parsed), 4) is not None)
    result = block_beam_search(units, pilot.WordAdditiveScorer(inventory['words']),
        grammar_accept=lambda state: state in permitted and linguistic_gate(state),
        allow_closed=closure, beam_width=64, max_steps=12, max_actions=10000,
        min_letters=60, max_letters=119, max_words=64, seed=921, diversity=0)
    terminals = [t for t in result['terminals'] if t['eligible']]
    assert len(terminals) == 1 and not result['truncated']
    left, right = tapes(terminals[0]['state'])
    rendered = pilot.render_paragraph(grammar.paragraph(left + right, 4))
    assert rendered == CANDIDATES[1]
    return {'scope': 'Predeclared parent-supplied path under a stronger path-membership gate; representation only.',
        'rendered_text': rendered, 'global_exact': True, 'distinct_complete_clauses': 4,
        'novelty_eligible': normalize_letters(rendered) not in inventory['excluded_normalized'],
        'inventory_units': len(units), 'path_steps': 12, 'trace': trace,
        'max_actions': 10000, 'attempted_actions': result['attempted_actions'],
        'truncated': result['truncated'], 'coherence': 'unreviewed; repetitive seeing predicates'}


def run():
    output_path = OUT / 'licensed-control-inventory-v2-001.json'
    if output_path.exists() or (pilot.BASE / 'block-seam-comparison-run-007-config.json').exists():
        raise RuntimeError('preserve immutable inventory and pilot IDs')
    plan = {'schema_version': 1, 'version': VERSION, 'run_id': '007',
        'seeds': [921, 922, 923], 'band': [60, 119], 'beam_width': 1,
        'max_actions': 2000, 'max_steps': 64, 'max_words': 64, 'diversity': .4,
        'clause_count': [2, 4], 'distinct_clauses': True, 'workers': 1,
        'maximum_cell_seconds': 5, 'aggregate_outer_seconds': 60,
        'derivation_policy': 'Systematic licensed components from the bank, not a target-success whitelist.',
        'search_policy': 'Unguided native two-sided beam; no supplied paragraph path restriction.',
        'source_sha256': sha(Path(__file__))}
    (OUT / 'licensed-control-inventory-v2-001-plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    inventory, grammar, original = build_inventory()
    for text in CANDIDATES:
        assert grammar.text_paragraph(text, 4) is not None
    inventory['boundary_certificate'] = boundary_certificate(grammar)
    inventory['representation'] = represent(inventory, grammar)
    output_path.write_text(json.dumps(inventory, indent=2) + '\n')
    original_config, original_search = pilot.build_config, pilot.block_beam_search
    def configured():
        cfg = dict(original)
        cfg['words'], cfg['blocks'] = inventory['words'], inventory['blocks']
        cfg['unit_provenance'] = inventory['unit_provenance']
        cfg['excluded_normalized'] = inventory['excluded_normalized']
        cfg['inventory_version'] = VERSION
        cfg['grammar_production_count'] = len(grammar.paths)
        cfg['cells'] = [c for c in original['cells'] if c['arm'] == 'block_seam' and c['band'] == [60, 119]]
        cfg['budget'] = dict(original['budget'], beam_width=1, cells=3)
        cfg['source_hashes'] = {**original['source_hashes'],
            str(output_path.relative_to(ROOT)): sha(output_path),
            str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}
        cfg['preflight'] = {**original['preflight'],
            'vocabulary_sources': {w: sorted({str(r.get('source', r.get('source_record', {}).get('source', 'existing frozen bank')))
                for r in inventory['unit_provenance'].get(w, [])}) for w in inventory['words']},
            'component_policy': inventory['derivation'],
            'lexical_membership_basis': 'Existing frozen pilot vocabulary plus source-bound components licensed by the unchanged default typed grammar; proper names use its declared name lexicon.',
            'source_tokens_not_added_as_new_licensed_components': inventory['source_tokens_not_added_as_new_licensed_components']}
        return cfg, grammar
    def searched(*args, **kwargs):
        kwargs['beam_width'] = 1
        return original_search(*args, **kwargs)
    try:
        pilot.build_config, pilot.block_beam_search = configured, searched
        pilot.run('007')
    finally:
        pilot.build_config, pilot.block_beam_search = original_config, original_search


if __name__ == '__main__':
    run()
