"""Exact all-length palindrome reachability for a finite typed clause grammar.

Unlike a bounded decoder, the finite pair graph exhausts the language. Each
character edge retains its lexical path; no lexeme can change halfway through.
Productive cycles certify unbounded exact tapes, NOT readable or admissible prose.
"""
from collections import defaultdict, deque
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def audit(text):
    tape = ''.join(c.lower() for c in text if 'a' <= c.lower() <= 'z')
    i, j = 0, len(tape) - 1
    while i < j and tape[i] == tape[j]:
        i, j = i + 1, j - 1
    return dict(letters=len(tape), exact=bool(tape) and i >= j,
                sha256_forward=hashlib.sha256(tape.encode()).hexdigest(),
                sha256_reverse=hashlib.sha256(tape[::-1].encode()).hexdigest())


class Grammar:
    def __init__(self):
        self.out = defaultdict(list)
        self.inc = defaultdict(list)
        self.size = 1
        self.start = self.end = 0

    def node(self):
        n = self.size
        self.size += 1
        return n

    def phrase(self, a, b, text):
        chars = re.sub('[^a-z]', '', text.lower())
        for i, char in enumerate(chars):
            target = b if i == len(chars) - 1 else self.node()
            edge = (a, target, char, text if i == 0 else '')
            self.out[a].append(edge)
            self.inc[target].append(edge)
            a = target


def make_grammar(repair=False, holdout_seed=False):
    g = Grammar()
    # Selectional roles and grammatical number are part of state identity.
    inventories = [
        (['an aide', 'a reader', 'Diana', 'Nora'], ['rips', 'reads', 'files'],
         ['nine memos', 'some notes', 'a memo']),
        (['some men', 'the aides', 'the readers'], ['inspire', 'admire'],
         ['Diana', 'Nora', 'an aide']),
    ]
    if repair:
        inventories += [
            (['I', 'we'], ['saw', 'met', 'admired'], ['Leon', 'an aide', 'Diana']),
            (['Leon', 'Nora'], ['saw', 'met'], ['some men', 'an aide']),
        ]
    for subjects, verbs, objects in inventories:
        after_subject, after_verb = g.node(), g.node()
        for phrase in subjects:
            g.phrase(0, after_subject, phrase)
        for phrase in verbs:
            g.phrase(after_subject, after_verb, phrase)
        for phrase in objects:
            if holdout_seed and phrase == 'nine memos':
                continue
            g.phrase(after_verb, 0, phrase + ';')
    return g


def analyze(g):
    root = (g.start, g.end)
    todo = deque([root])
    parent = {root: None}
    arcs, reverse = defaultdict(list), defaultdict(list)
    centers = {}
    while todo:
        pair = todo.popleft()
        a, b = pair
        if a == b:
            centers[pair] = None
        else:
            for edge in g.out[a]:
                if edge[1] == b:
                    centers[pair] = edge
                    break
        right = defaultdict(list)
        for edge in g.inc[b]:
            right[edge[2]].append(edge)
        for left_edge in g.out[a]:
            for right_edge in right[left_edge[2]]:
                target = (left_edge[1], right_edge[0])
                arcs[pair].append((target, left_edge, right_edge))
                reverse[target].append(pair)
                if target not in parent:
                    parent[target] = (pair, left_edge, right_edge)
                    todo.append(target)
    productive = set(centers)
    todo = deque(centers)
    while todo:
        for previous in reverse[todo.popleft()]:
            if previous not in productive:
                productive.add(previous)
                todo.append(previous)
    # Kahn elimination detects any cycle on a path from start to a midpoint.
    indegree = {p: 0 for p in productive}
    for p in productive:
        for target, _, _ in arcs[p]:
            if target in productive:
                indegree[target] += 1
    todo = deque(p for p, degree in indegree.items() if not degree)
    removed = 0
    while todo:
        p = todo.popleft()
        removed += 1
        for target, _, _ in arcs[p]:
            if target in indegree:
                indegree[target] -= 1
                if not indegree[target]:
                    todo.append(target)
    witnesses = []
    closing_paths = [(p, target, le, re_) for p in arcs
                     for target, le, re_ in arcs[p] if target in centers]
    for previous, center, last_left, last_right in closing_paths:
        middle = centers[center]
        left, right, p = [last_left], [last_right], previous
        while parent[p] is not None:
            p, le, re_ = parent[p]
            left.append(le)
            right.append(re_)
        edges = list(reversed(left)) + ([middle] if middle else []) + right
        text = ' '.join(e[3] for e in edges if e[3]).rstrip(';') + '.'
        if audit(text)['exact']:
            witnesses.append(dict(text=text, audit=audit(text)))
        else:
            raise AssertionError(('witness reconstruction failure', text))
    return dict(states=g.size, pair_states=len(parent),
                pair_edges=sum(map(len, arcs.values())),
                productive_pairs=len(productive),
                nonempty_exact_language=bool(closing_paths),
                unbounded_exact_language=removed < len(productive),
                witnesses=sorted(witnesses, key=lambda x: x['audit']['letters'], reverse=True))


def run():
    results = [dict(variant='typed_core', **analyze(make_grammar())),
               dict(variant='past_tense_pronoun_repair', **analyze(make_grammar(True))),
               dict(variant='seed_object_withheld', **analyze(make_grammar(True, True)))]
    return dict(experiment_id='palindrome-grammar-reachability-20260918',
                novelty='Exhaustive finite pair-graph reachability plus productive-cycle test; no beam or maximum length.',
                provenance='Fresh authored typed subject/verb/object inventories; user seed vocabulary included as calibration, never claimed original.',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                results=results,
                intact_controls=[dict(text=t, audit=audit(t)) for t in [
                    'A reader files some notes; the aides admire Nora.',
                    'Nora met some men; we admired Diana.']],
                reader_evidence=False,
                admission='All exact witnesses are seed calibration or clause-order rearrangements. None is a new generated result. Unboundedness allows repetition and is not success.',
                next_repair='Use dead productive-pair frontier to request role-compatible lexical paths spanning multiple word boundaries; grammar cycles alone do not provide long prose.')


if __name__ == '__main__':
    result = run()
    path = ROOT / 'runs/palindrome-grammar-reachability-20260918.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
