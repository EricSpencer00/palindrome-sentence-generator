"""Ablate synchronized clause closures against one paragraph automaton.

Event completion, rather than a paired sentence index, controls topology.
An event occurs once; reference forms require introduced entities. Sentence
punctuation emits no letters, so the two character fronts cross sentence
boundaries independently. No complete-sentence bank is materialized.
"""
from collections import deque
import hashlib
import json
from pathlib import Path

from experiments.packed_seam_grammar_20260927 import Grammar, SEED, audit, intersect, norm
from llm_palindrome.validator import is_palindrome

ROOT = Path(__file__).resolve().parents[1]
ID = 'packed-discourse-automaton-20260927'
EVENTS = ('tear_documents', 'inspire_person', 'read_documents')


def accessible(mask):
    entities = set()
    if mask & 1:
        entities.update(('aide', 'memos'))
    if mask & 2:
        entities.update(('men', 'Diana'))
    return entities


def event_slots(event, mask):
    if event == 0:
        return (('subject:introduce:aide', ('an aide', 'a clerk', 'a tired aide')),
                ('verb:singular:documents', ('rips', 'shreds', 'tears')),
                ('object:introduce:memos', ('nine memos.', 'two memos.', 'some memos.')))
    if event == 1:
        return (('subject:introduce:men', ('some men', 'two men', 'the writers')),
                ('verb:plural:human', ('inspire', 'encourage', 'help')),
                ('object:introduce:Diana', ('Diana.',)))
    if event == 2 and {'Diana', 'memos'} <= accessible(mask):
        return (('subject:reference:Diana', ('Diana', 'she')),
                ('verb:singular:documents', ('reads', 'saves', 'files')),
                ('object:reference:memos', ('the memos.', 'those memos.', 'them.')))
    return None


def compile_paragraph(max_events=3):
    g = Grammar()
    accepting = g.new()
    boundaries = {0: g.start}
    pending = deque([0])
    trace = []
    while pending:
        mask = pending.popleft()
        source = boundaries[mask]
        if mask & 3 == 3:
            g.epsilon[source].append(accepting)
        if mask.bit_count() >= max_events:
            continue
        for event in range(len(EVENTS)):
            if mask & (1 << event):
                continue
            slots = event_slots(event, mask)
            if slots is None:
                continue
            target_mask = mask | (1 << event)
            if target_mask not in boundaries:
                boundaries[target_mask] = g.new()
                pending.append(target_mask)
            # Alternative event orders branch from the same packed discourse
            # state and merge at the same completed-event state.
            g.finish = source
            for role, alternatives in slots:
                g.slot(alternatives, role)
            g.epsilon[g.finish].append(boundaries[target_mask])
            trace.append(dict(source_mask=mask, target_mask=target_mask,
                event=EVENTS[event], introduced=sorted(accessible(mask)),
                boundary_is_epsilon=True))
    g.finish = accepting
    return g, trace


def controls():
    rows = []
    for order in ((0, 1), (1, 0), (0, 1, 2), (1, 0, 2)):
        mask, clauses = 0, []
        for event in order:
            slots = event_slots(event, mask)
            assert slots is not None
            clause = ' '.join(alternatives[0] for _, alternatives in slots)
            clauses.append(clause[0].upper() + clause[1:])
            mask |= 1 << event
        rendered = ' '.join(clauses)
        rows.append(dict(rendered=rendered, events=[EVENTS[i] for i in order],
                         audit=audit(rendered), generated_by_grammar=True))
    return rows


def run():
    conditions = []
    for size in (2, 3):
        g, trace = compile_paragraph(size)
        result = intersect(g, max_letters=160, cap=50000)
        for row in result['candidates']:
            original_tape = norm(row['rendered'])
            sentences = [s.strip() for s in row['rendered'].split('.') if s.strip()]
            row['rendered'] = ' '.join(s[0].upper()+s[1:]+'.' for s in sentences)
            assert norm(row['rendered']) == original_tape
            row['audit']['independent_validator_exact'] = is_palindrome(row['rendered'])
            row['novel_relative_to_seed'] = norm(row['rendered']) != norm(SEED)
            row['seed_rotation'] = (len(original_tape) == len(norm(SEED))
                                    and original_tape in norm(SEED)*2)
            row['new_content'] = not row['seed_rotation']
            row['human_readability_evidence'] = 'not collected'
        result.update(max_events=size, discourse_transitions=trace)
        conditions.append(result)
    return dict(experiment_id=ID, conditions=conditions,
        controls=controls(),
        positive_control=dict(rendered=SEED, audit=audit(SEED),
            recovered_in_both_conditions=all(any(row['audit']['normalized'] == norm(SEED)
                for row in condition['candidates']) for condition in conditions)),
        provenance=dict(complete_sentence_enumeration=False,
            source='fresh authored typed event slots; seed recovery is inherited',
            catalogue_replay=False, per_candidate_rlaif=False,
            event_repetition=False, semordnilap_bank=False,
            generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        novelty_preflight=dict(novel_algorithm_claim=False,
            representation_change='whole forward paragraph with event-mask merges and reference accessibility; no independently closed clause pairs',
            boundaries='epsilon punctuation; asynchronous character fronts'),
        reader_test=dict(status='not collected',
            next='For each novel exact paragraph, create randomized blinded ratings with intact controls and shuffled versions. Recovery controls are not new outputs.'),
        next_repair=dict(operator='Move a postposed temporal adjunct across the sentence boundary while retaining its event attachment.',
            reason='Event ordering changes middle topology but leaves current outer subject/object character sets narrow; moving an existing adjunct changes endpoint possibilities structurally.',
            concrete='Add a single time fact with before/after-clause realization alternatives, shared semantic identity, and exactly-once usage; compare frontier survival using unchanged core event vocabulary.'))


if __name__ == '__main__':
    result = run()
    (ROOT/'runs'/f'{ID}.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps([dict(max_events=c['max_events'], states=c['states'],
        exact=len(c['candidates']), new_content=sum(x['new_content'] for x in c['candidates']))
        for c in result['conditions']]))
