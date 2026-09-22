"""Move one event-scoped temporal fact through the packed paragraph graph."""
from collections import deque
import hashlib
import json
from pathlib import Path

from experiments.packed_discourse_automaton_20260927 import EVENTS, accessible, event_slots
from experiments.packed_seam_grammar_20260927 import Grammar, SEED, audit, intersect, norm
from llm_palindrome.validator import is_palindrome

ROOT = Path(__file__).resolve().parents[1]
ID = 'packed-temporal-paragraph-20260927'
TIMES = ('now', 'today', 'before dawn', 'after work', 'in the morning')


def temporal_slots(slots, placement):
    if placement == 'bare':
        return slots
    if placement == 'front':
        return (('time:fact0:event0', tuple(t+',' for t in TIMES)),) + slots
    final_role, final_words = slots[-1]
    return slots[:-1] + ((final_role, tuple(w.rstrip('.') for w in final_words)),
        ('time:fact0:event0', tuple(t+'.' for t in TIMES)))


def compile_temporal(mode='movable', max_events=3):
    g = Grammar()
    accepting = g.new()
    boundaries = {(0, False): g.start}
    pending = deque([(0, False)])
    trace = []
    while pending:
        mask, time_used = pending.popleft()
        source = boundaries[mask, time_used]
        if mask & 3 == 3 and (time_used or mode == 'bare'):
            g.epsilon[source].append(accepting)
        if mask.bit_count() >= max_events:
            continue
        for event in range(len(EVENTS)):
            if mask & (1 << event):
                continue
            slots = event_slots(event, mask)
            if slots is None:
                continue
            placements = ('bare',)
            if event == 0 and mode != 'bare':
                placements = ('front', 'post') if mode == 'movable' else (mode,)
            for placement in placements:
                new_used = time_used or placement != 'bare'
                target = (mask | (1 << event), new_used)
                if target not in boundaries:
                    boundaries[target] = g.new()
                    pending.append(target)
                g.finish = source
                for role, alternatives in temporal_slots(slots, placement):
                    g.slot(alternatives, role)
                g.epsilon[g.finish].append(boundaries[target])
                trace.append(dict(source=[mask, time_used], target=list(target),
                    event=EVENTS[event], accessible=sorted(accessible(mask)),
                    temporal_placement=placement, time_fact_event=0 if placement != 'bare' else None))
    g.finish = accepting
    return g, trace


def render(text):
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    return ' '.join(s[0].upper()+s[1:]+'.' for s in sentences)


def controls():
    rows = []
    for placement in ('front', 'post'):
        parts = []
        for event, mask in ((0, 0), (1, 1), (2, 3)):
            slots = temporal_slots(event_slots(event, mask), placement if event == 0 else 'bare')
            parts.extend(words[0] for _, words in slots)
        text = render(' '.join(parts))
        rows.append(dict(rendered=text, audit=audit(text), temporal_placement=placement,
            time_fact_count=1, event_scope='tear_documents', generated_by_grammar=True))
    return rows


def run():
    conditions = []
    for mode in ('bare', 'front', 'post', 'movable'):
        g, trace = compile_temporal(mode)
        result = intersect(g, max_letters=160, cap=50000)
        for row in result['candidates']:
            row['rendered'] = render(row['rendered'])
            row['audit']['independent_validator_exact'] = is_palindrome(row['rendered'])
            row['seed_rotation'] = (len(norm(row['rendered'])) == len(norm(SEED))
                                    and norm(row['rendered']) in norm(SEED)*2)
            row['new_content'] = not row['seed_rotation']
            time_emissions = [g.edges[i][3] for i in row['accepting_path']
                              if g.edges[i][4] == 'time:fact0:event0' and g.edges[i][3]]
            row['temporal_fact_emissions'] = time_emissions
            assert len(time_emissions) == (0 if mode == 'bare' else 1)
            row['human_readability_evidence'] = 'not collected'
        result.update(mode=mode, discourse_transitions=trace)
        conditions.append(result)
    return dict(experiment_id=ID, conditions=conditions, complete_prose_controls=controls(),
        provenance=dict(core_event_lexicon_unchanged=True,
            topology_change='one event-scoped fact may precede or follow its clause, exactly once',
            temporal_vocabulary=TIMES, complete_sentence_enumeration=False,
            seed_and_rotation_are_controls=True, catalogue_replay=False,
            generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        novelty_preflight=dict(novel_algorithm_claim=False,
            distinct_from='packed discourse event-order ablation: temporal placement changes clause endpoint languages'),
        reader_test=dict(status='not collected', next='Novel exact output goes into randomized blind intact/shuffled comparison.'),
        next_repair=dict(operator='Reopen the Diana noun-phrase / next-clause boundary as one grammatical window.',
            concrete='Allow Diana to be the subject of a finite follow-on predicate rather than forcing it to finish the preceding object phrase; retain the same event ledger and exact-character product.',
            constraint='This is a change to argument sharing and sentence attachment, not another temporal word list.'))


if __name__ == '__main__':
    result = run()
    (ROOT/'runs'/f'{ID}.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps([dict(mode=r['mode'], states=r['states'], exact=len(r['candidates']),
        new_content=sum(x['new_content'] for x in r['candidates'])) for r in result['conditions']]))
