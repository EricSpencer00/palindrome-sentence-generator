"""Exact intersection of an acyclic lexical NFA with the palindrome language.

The generation object is a pair of automaton states, not a completed sentence.
Every transition consumes the SAME character at opposite ends of one accepting
path. Surface words are recovered only after the two paths meet. No LM runs.
"""
from collections import defaultdict, deque
from itertools import product
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]


def normalize(text):
    return re.sub('[^a-z]', '', text.lower())


def independent_audit(text):
    tape = ''.join(c.lower() for c in text if 'a' <= c.lower() <= 'z')
    i, j = 0, len(tape) - 1
    exact = True
    while i < j:
        if tape[i] != tape[j]:
            exact = False
        i += 1
        j -= 1
    return dict(letters=len(tape), two_pointer_exact=exact,
                forward_sha256=hashlib.sha256(tape.encode()).hexdigest(),
                reverse_sha256=hashlib.sha256(tape[::-1].encode()).hexdigest(),
                hash_equal=hashlib.sha256(tape.encode()).digest() ==
                           hashlib.sha256(tape[::-1].encode()).digest())


def compile_slots(slots):
    """Union within each lexical slot, concatenation between slots."""
    edges = []
    next_state = 1
    start = 0
    boundary = start
    for options in slots:
        paths = []
        for surface in options:
            tape = normalize(surface)
            assert tape
            inner = list(range(next_state, next_state + len(tape) - 1))
            next_state += len(inner)
            paths.append((surface, tape, inner))
        end = next_state
        next_state += 1
        for surface, tape, inner in paths:
            states = [boundary] + inner + [end]
            for k, char in enumerate(tape):
                edges.append((states[k], states[k+1], char,
                              surface if k == 0 else ''))
        boundary = end
    return start, boundary, edges


def intersect(slots):
    start, finish, edges = compile_slots(slots)
    forward, backward = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    for index, (u, v, char, _) in enumerate(edges):
        forward[u][char].append(index)
        backward[v][char].append(index)
    queue = deque([(start, finish, (), ())])
    seen = set()
    results = []
    expanded = matched = dead = 0
    deepest = 0
    trace = []
    while queue:
        left, right, lp, rp = queue.popleft()
        key = (left, right, len(lp))
        if key in seen:
            continue
        seen.add(key)
        expanded += 1
        deepest = max(deepest, len(lp))
        centers = [()] if left == right else []
        centers += [(eid,) for ids in forward[left].values() for eid in ids
                    if edges[eid][1] == right]
        for center in centers:
            path = lp + center + rp[::-1]
            # Verify accepting path independently from its string audit.
            cursor = start
            for eid in path:
                assert edges[eid][0] == cursor
                cursor = edges[eid][1]
            assert cursor == finish
            text = ''.join(edges[eid][3] for eid in path)
            results.append(dict(text=text, audit=independent_audit(text),
                                lexical_path=list(path)))
        common = forward[left].keys() & backward[right].keys()
        advanced = False
        for char in sorted(common):
            for le in forward[left][char]:
                for re_ in backward[right][char]:
                    lnext, rnext = edges[le][1], edges[re_][0]
                    # State IDs increase strictly along all accepting paths.
                    if lnext <= rnext:
                        assert edges[le][2] == edges[re_][2]
                        queue.append((lnext, rnext, lp + (le,), rp + (re_,)))
                        matched += 1
                        advanced = True
        if not advanced and not centers:
            dead += 1
            trace.append(dict(depth=len(lp), left_state=left, right_state=right,
                              left_consumed=''.join(edges[e][2] for e in lp),
                              right_consumed=''.join(edges[e][2] for e in rp[::-1]),
                              left_started_words=''.join(edges[e][3] for e in lp),
                              right_started_words=''.join(edges[e][3] for e in rp[::-1]),
                              required_from_left=sorted(forward[left]),
                              available_from_right=sorted(backward[right])))
    return dict(candidates=results, stats=dict(states=expanded, matched_transitions=matched,
                dead_frontiers=dead, deepest_matched_pairs=deepest, nfa_edges=len(edges)),
                dead_frontier_examples=sorted(trace, key=lambda r: -r['depth'])[:20],
                enumeration='One witness per (left state, right state, depth); complete for existence, not all surface variants.')


def self_test():
    # Exhaustive oracle on a tiny language checks even/odd centers and merges.
    for slots in [[['a', 'ab'], ['a', 'ba', 'b']], [['ab'], ['ba']], [['ab'], ['a']]]:
        oracle = {''.join(p) for p in product(*slots)
                  if normalize(''.join(p)) == normalize(''.join(p))[::-1]}
        found = {r['text'] for r in intersect(slots)['candidates']}
        assert bool(found) == bool(oracle)
        assert found <= oracle
    seed = 'An aide rips nine memos; some men inspire Diana.'
    # Declared regression only: not part of the new generation inventory.
    recovered = intersect([[seed]])['candidates']
    assert len(recovered) == 1 and recovered[0]['audit']['two_pointer_exact']
    return dict(tiny_exhaustive_oracles=3, declared_seed_regression=True)


def main():
    slots = [
        ['Nora ', 'Diana ', 'Leon ', 'Noel '],
        ['asks ', 'assists ', 'admires ', 'helps '],
        ['a careful clerk; ', 'an honest sailor; ', 'a quiet aide; '],
        ['some clerks ', 'the sailors ', 'several aides '],
        ['ask ', 'assist ', 'admire ', 'help '],
        ['Aaron.', 'Diana.', 'Leon.', 'Noel.'],
    ]
    result = intersect(slots)
    # The observed depth-five conflict requires s on the right, while all
    # plural finite verbs end e/k/p/t. Change the grammatical subject-number
    # construction, carrying agreement in the same lexical edge.
    singular_slots = slots[:3] + [[
        'a careful clerk asks ', 'an honest sailor assists ',
        'a quiet aide admires ', 'a careful clerk helps ',
    ], slots[-1]]
    singular = intersect(singular_slots)
    controls = [''.join(p) for p in list(product(*slots))[:3]]
    result.update(experiment_id='online-regular-language-palindrome-20260921',
        method='Online equal-character product of forward and reverse lexical NFA paths',
        invariant='Every queued path consumes equal characters at opposing positions before rendering; a candidate requires a connected even or odd center.',
        slots=slots, language_size=4*4*3*3*4*4,
        controls=[dict(text=t, audit=independent_audit(t)) for t in controls],
        provenance=dict(source='Fresh hand-authored finite clause grammar; no catalogue sentence or seed in generation inventory.',
                        seed_used_only_for_regression=True,
                        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        novelty_preflight=dict(status='implementation correction, not a new theoretical family',
            compared_sources=['shared_tape_finite_automata_role_ledger_20260921.py',
                              'semantic_slot_lattice_smt.py',
                              'cfg_earley_character_intersection_20260916.py'],
            distinction='Those inspected scripts evaluate pre-rendered prose; this program only reaches rendering through live equal-character edges and an accepting path.'),
        verification=self_test(), reader_evidence=None,
        construction_followup=dict(
            reason='The depth-five conflict is caused by plural verb endings. A singular subject plus its agreeing verb is now one atomic alternative.',
            slots=singular_slots, result=singular,
            control=dict(text='Nora asks a careful clerk; a careful clerk asks Aaron.',
                         audit=independent_audit('Nora asks a careful clerk; a careful clerk asks Aaron.'))),
        reader_gate='Closed until a longer exact candidate passes provenance/shortcut review and blinded readers.',
        next_construction='Use the recorded dead-frontier letters to jointly synthesize additional inflected verb/argument alternatives with semantic compatibility; rerun the identical automaton product. No completed candidate is repaired.')
    out = ROOT / 'runs/online-regular-language-palindrome-20260921.json'
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(stats=result['stats'], exact=len(result['candidates']),
                         agreement_followup=singular['stats'], controls=controls)))


if __name__ == '__main__':
    main()
