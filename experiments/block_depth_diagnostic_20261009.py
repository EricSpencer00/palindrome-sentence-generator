"""One predeclared local width sweep; reuse frozen pilot callbacks unchanged."""
import collections
import hashlib
import json
from pathlib import Path
import time

from experiments import block_seam_comparison_20261009 as pilot

WIDTHS = (1, 2, 4)
RUN_IDS = ('004', '005', '006')
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/block-seams'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(path):
    data = json.loads(path.read_text())
    result = []
    for cell in data['cells']:
        d = cell['block_search_diagnostics']
        layers = collections.defaultdict(lambda: {'actions': 0, 'parents': set(),
                                                  'retained': 0, 'pruned': 0,
                                                  'max_parent_letters': 0,
                                                  'max_retained_letters': 0})
        for a in d['action_log']:
            left, right = a['parent_left'], a['parent_right']
            depth = len(left) + len(right)
            layer = layers[depth]
            layer['actions'] += 1
            layer['parents'].add((tuple(left), tuple(right)))
            letters = len(pilot.normalize_letters(' '.join(left + right)))
            layer['max_parent_letters'] = max(layer['max_parent_letters'], letters)
            if a['status'] in ('retained_proposal', 'beam_pruned'):
                layer['retained'] += 1
                layer['max_retained_letters'] = max(layer['max_retained_letters'],
                    letters + len(pilot.normalize_letters(a['text'])))
            layer['pruned'] += a['status'] == 'beam_pruned'
        frontier = []
        for depth, layer in sorted(layers.items()):
            parent_count = len(layer.pop('parents'))
            frontier.append({'depth': depth, 'expanded_parents': parent_count,
                **layer, 'all_logged_parents_fully_enumerated':
                layer['actions'] == 108 * parent_count})
        last_depth = max(layers, default=-1)
        full_depths = [z['depth'] for z in frontier
                      if z['all_logged_parents_fully_enumerated']
                      and (z['depth'] < last_depth or d['status'] == 'completed')]
        eligible = [z for z in cell['closures'] if z.get('mechanically_eligible_for_review')]
        result.append({k: cell[k] for k in ('cell_id', 'seed', 'band', 'status',
                                           'elapsed_seconds', 'expanded_actions')} | {
            'attempted_side_unit_actions': d['attempted_actions'],
            'visited_states': d['visited_states'], 'truncated': d['truncated'],
            'completed_expansion_depths': full_depths,
            'maximum_expanded_depth': last_depth, 'layers': frontier,
            'max_visited_letters': max((z['max_parent_letters'] for z in frontier), default=0),
            'max_retained_proposal_letters': max((z['max_retained_letters'] for z in frontier), default=0),
            'raw_closures': len(cell['closures']), 'eligible_closures': len(eligible),
            'closure_records': cell['closures'],
            'action_rejection_counts': dict(collections.Counter(
                a.get('reason') for a in d['action_log'] if a.get('reason'))),
            'callback_rejection_counts': dict(collections.Counter(
                a.get('rejection') for a in cell['proposed_actions'] if a.get('rejection'))),
            'remaining_frontier': d['remaining_beam'],
            'partial_next_frontier_discarded': sum(
                a['status'] == 'retained_proposal' and 'beam_selected' not in a
                for a in d['action_log'])})
    return result


def run():
    plan_path = OUT / 'depth-diagnostic-001-plan.json'
    if plan_path.exists():
        raise RuntimeError('diagnostic already declared; no duplicate sweep')
    for run_id in RUN_IDS:
        if (pilot.BASE / f'block-seam-comparison-run-{run_id}-config.json').exists():
            raise RuntimeError('immutable run ID already used')
    plan = {'schema_version': 1, 'base_commit': 'd3d92dc132eeb9181e7f5f059b8aa572322ea090',
        'purpose': 'Measure depth and target-band reachability at fixed action cap',
        'widths': list(WIDTHS), 'run_ids': list(RUN_IDS), 'seeds': [921, 922, 923],
        'band': [60, 119], 'arm': 'typed-two-sided-block-beam-v1',
        'aggregate_outer_seconds': 60, 'maximum_cell_seconds': 5, 'workers': 1,
        'max_actions': 2000, 'max_steps': 64, 'max_words': 64, 'diversity': .4,
        'branch_cap': None, 'enumeration': 'All 54 units on both sides; 108 attempts per parent',
        'closure': 'Unchanged global exactness, 2–4 distinct rendered clauses, known-control exclusion',
        'comparison': 'Width alone varies. Reference measurements remain frozen in pilot 003.',
        'coverage_tradeoff': 'Narrow beams prune more compatible proposals; cap discards partial next frontier.',
        'settings_order': [1, 2, 4], 'outcome_tuning': False,
        'source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in
            (Path(__file__), Path(pilot.__file__), ROOT / 'llm_palindrome/block_search.py',
             ROOT / 'llm_palindrome/typed_constituents.py')}}
    plan_path.write_text(json.dumps(plan, indent=2) + '\n')
    original_config, original_search = pilot.build_config, pilot.block_beam_search
    summaries = []
    try:
        for width, run_id in zip(WIDTHS, RUN_IDS):
            def configured(width=width):
                cfg, grammar = original_config()
                cfg['cells'] = [c for c in cfg['cells'] if c['arm'] == 'block_seam' and c['band'] == [60, 119]]
                cfg['budget']['beam_width'] = width
                cfg['budget']['cells'] = len(cfg['cells'])
                cfg['diagnostic_plan_sha256'] = sha(plan_path)
                cfg['source_hashes'][str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
                return cfg, grammar
            def searched(*args, width=width, **kwargs):
                kwargs['beam_width'] = width
                return original_search(*args, **kwargs)
            pilot.build_config, pilot.block_beam_search = configured, searched
            pilot.run(run_id)
            result_path = pilot.BASE / f'block-seam-comparison-run-{run_id}-results.json'
            summaries.append({'width': width, 'run_id': run_id,
                'results_sha256': sha(result_path), 'cells': summarize(result_path)})
            (OUT / 'depth-diagnostic-001-summary.json').write_text(
                json.dumps({'plan_sha256': sha(plan_path), 'settings': summaries}, indent=2) + '\n')
    finally:
        pilot.build_config, pilot.block_beam_search = original_config, original_search


if __name__ == '__main__':
    run()
