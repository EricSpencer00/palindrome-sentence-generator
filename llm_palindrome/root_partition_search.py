"""Deterministic bounded quotas for all viable root actions; one worker."""
import time

from .admission import normalize_letters
from .block_seams import Seam
from .block_search import block_beam_search, compatible_actions

ROOT_SCHEDULE_VERSION = 'root-quotas-v1'


def root_partition_search(inventory, scorer, boundary_index, *, grammar_accept,
                          allow_closed, beam_width=1, max_steps=64,
                          max_actions_per_root=2000, min_letters=60,
                          max_letters=119, max_words=64, seeds=(921,922,923),
                          diversity=.4, deadline=None, seconds_per_root=2.0,
                          output_reserve_seconds=1.0):
    if not seeds or max_steps < 1 or seconds_per_root <= 0:
        raise ValueError('positive root horizon, seeds and time quota required')
    roots = compatible_actions(Seam(), inventory, grammar_accept=grammar_accept,
        boundary_accept=boundary_index.allows_state, deadline=deadline,
        max_letters=max_letters, max_words=max_words)
    roots.sort(key=lambda action:(action.side, action.unit.text, action.unit.id))
    lanes = []
    for number, action in enumerate(roots):
        now = time.monotonic()
        remaining = len(roots) - number
        quota = seconds_per_root if deadline is None else min(seconds_per_root,
            max(0.0, deadline-now-output_reserve_seconds)/remaining)
        seed = seeds[number % len(seeds)]
        family_ids = boundary_index.matching_families(
            ' '.join(p.text for p in action.child.left),
            ' '.join(p.text for p in action.child.right))
        lane = {'root_number':number,'root_side':action.side,
                'root_unit':action.unit.text,'root_id':action.unit.id,
                'boundary_families':list(family_ids),'seed':seed,
                'allocated_seconds':quota,'beam_width':beam_width,
                'max_actions':max_actions_per_root,'max_added_steps':max_steps-1}
        if quota<=0:
            lane.update(status='aggregate_deadline_before_root',result=None)
        else:
            left = tuple(p.text for p in action.child.left)
            right = tuple(p.text for p in action.child.right)
            root_delta = 0.0 if scorer is None else scorer.word_delta(left,right,
                'L' if action.side=='left' else 'R',action.unit.text,
                'append' if action.side=='left' else 'prepend')
            result = block_beam_search(inventory,scorer,initial_state=action.child,
                grammar_accept=grammar_accept,allow_closed=allow_closed,
                boundary_accept=boundary_index.allows_state,
                beam_width=beam_width,max_steps=max_steps-1,
                max_actions=max_actions_per_root,min_letters=min_letters,
                max_letters=max_letters,max_words=max_words,seed=seed,
                diversity=diversity,deadline=now+quota)
            lane.update(status=result['status'],root_score_delta=root_delta,
                        result=result,elapsed_seconds=time.monotonic()-now)
        lanes.append(lane)
    all_families={pair['family_id'] for pair in boundary_index.pairs}
    root_families={family for lane in lanes for family in lane['boundary_families']}
    attempted_families={family for lane in lanes if lane['result'] is not None
                       for family in lane['boundary_families']}
    return {'version':ROOT_SCHEDULE_VERSION,'root_count':len(roots),'lanes':lanes,
        'family_count':len(all_families),'families_with_viable_root':sorted(root_families),
        'families_with_started_root':sorted(attempted_families),
        'families_without_inventory_root':sorted(all_families-root_families),
        'root_order':'lexicographic (side, text, ID), independent of score',
        'seed_assignment':'seeds[root_number modulo len(seeds)]',
        'score_policy':'Original scorer and diversity ranking within each lane. All root actions retained. Root score is a lane-constant omitted by native seeded search and restored when reporting total terminal scores.',
        'coverage_policy':'One fixed-width beam per viable root action; equal bounded time/action quotas. Narrow-lane beam pruning and partial-frontier loss remain logged.',
        'grammar_scope':'Boundary feasibility is necessary for the declared finite grammar, not a corpus-match or general-English completeness claim.'}
