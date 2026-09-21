"""Root-supported CFG chart and global palindrome-domain propagation.

No strings are enumerated before constraint search. Spaces are lexical metadata;
one sentence derivation spans the entire shared tape, including its center.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ALPHABET = frozenset('abcdefghijklmnopqrstuvwxyz')
LEXICON = {
    'DET': ('a', 'the'), 'PERSON': ('guard', 'poet', 'scribe', 'sailor', 'writer'),
    'NP': ('Diana', 'Leon', 'Mira', 'she', 'he'),
    'READ': ('reads', 'studies'), 'WRITE': ('writes', 'copies'),
    'TEXT': ('letter', 'map', 'poem', 'note', 'memo'),
    'CONJ': ('and', 'while'),
}
# Separate verb frames make each finite VP take a text object. All subjects and
# finite verbs are singular. Coordination/subordination is recursive, bounded
# by target character length, not an enumerated bank of complete frames.
BINARY = (('NP', 'DET', 'PERSON'), ('OBJ', 'DET', 'TEXT'),
          ('VP', 'READ', 'OBJ'), ('VP', 'WRITE', 'OBJ'),
          ('C', 'NP', 'VP'), ('S', 'C', 'TAIL'), ('TAIL', 'CONJ', 'S'))
UNARY = (('S', 'C'),)

def normalize(text):
    return ''.join(c for c in text.lower() if 'a' <= c <= 'z')

def audit(text):
    tape = normalize(text)
    mismatch = next((i for i in range(len(tape)//2) if tape[i] != tape[-1-i]), None)
    return {'letters': len(tape), 'exact': bool(tape) and mismatch is None,
            'first_mismatch': mismatch,
            'sha256_forward': hashlib.sha256(tape.encode()).hexdigest(),
            'sha256_reverse': hashlib.sha256(tape[::-1].encode()).hexdigest()}

def chart(domains, lexicon=LEXICON, binary=BINARY, unary=UNARY):
    """Build a packed forest; each edge is lexical text or child chart keys."""
    n = len(domains)
    forest = {}
    def add(key, edge):
        edges = forest.setdefault(key, [])
        if edge not in edges:
            edges.append(edge)
    for category, words in lexicon.items():
        for word in words:
            letters = normalize(word)
            for i in range(n-len(letters)+1):
                if all(c in domains[i+k] for k, c in enumerate(letters)):
                    add((category, i, i+len(letters)), word)
    for width in range(1, n+1):
        for i in range(n-width+1):
            j = i+width
            for parent, left, right in binary:
                for k in range(i+1, j):
                    a, b = (left, i, k), (right, k, j)
                    if a in forest and b in forest:
                        add((parent, i, j), (a, b))
            # Unary closure permits grammars whose production order varies.
            changed = True
            while changed:
                before = len(forest)
                for parent, child in unary:
                    child_key = (child, i, j)
                    if child_key in forest:
                        add((parent, i, j), (child_key,))
                changed = len(forest) != before
    return forest

def supported_domains(forest, n, root='S'):
    """Only letters participating in at least one full parse retain support."""
    start = (root, 0, n)
    if start not in forest:
        return None
    supports = [set() for _ in range(n)]
    pending, seen = [start], set()
    while pending:
        key = pending.pop()
        if key in seen:
            continue
        seen.add(key)
        for edge in forest[key]:
            if isinstance(edge, str):
                for k, ch in enumerate(normalize(edge)):
                    supports[key[1]+k].add(ch)
            else:
                pending.extend(edge)
    return supports

def propagate(domains, lexicon=LEXICON, binary=BINARY, unary=UNARY, root='S', trace=None):
    domains = [set(x) for x in domains]
    rounds = 0
    while True:
        rounds += 1
        forest = chart(domains, lexicon, binary, unary)
        support = supported_domains(forest, len(domains), root)
        if trace is not None:
            trace.append({'round': rounds, 'root_exists': support is not None,
                          'chart_items': len(forest),
                          'outer_supported_domains': None if support is None else
                          [{'left': ''.join(sorted(support[i])),
                            'right': ''.join(sorted(support[-1-i])),
                            'shared': ''.join(sorted(support[i] & support[-1-i]))}
                           for i in range(min(6, len(domains)//2))]})
        if support is None:
            return None, forest, rounds
        reduced = [domains[i] & support[i] & domains[-1-i] & support[-1-i]
                   for i in range(len(domains))]
        if any(not x for x in reduced):
            return None, forest, rounds
        if reduced == domains:
            return reduced, forest, rounds
        domains = reduced

def yields(forest, key):
    for edge in forest[key]:
        if isinstance(edge, str):
            tape = normalize(edge)
            if len(tape) == 1 or tape != tape[::-1]:
                yield (edge,)
        elif len(edge) == 1:
            yield from yields(forest, edge[0])
        else:
            for left in yields(forest, edge[0]):
                for right in yields(forest, edge[1]):
                    words = left + right
                    if len({normalize(w) for w in words}) == len(words):
                        yield words

def search(n, max_nodes=100, lexicon=LEXICON, binary=BINARY, unary=UNARY, root='S'):
    stats = {'nodes': 0, 'propagation_rounds': 0, 'contradictions': 0,
             'complete_tapes': 0, 'truncated': False}
    outputs, root_trace = [], []
    def visit(domains):
        if stats['nodes'] >= max_nodes:
            stats['truncated'] = True
            return
        stats['nodes'] += 1
        domains, forest, rounds = propagate(domains, lexicon, binary, unary, root,
                                           root_trace if stats['nodes'] == 1 else None)
        stats['propagation_rounds'] += rounds
        if domains is None:
            stats['contradictions'] += 1
            return
        choices = [(len(domains[i]), i) for i in range((n+1)//2) if len(domains[i]) > 1]
        if not choices:
            stats['complete_tapes'] += 1
            for index, words in enumerate(yields(forest, (root, 0, n))):
                if index >= 200:
                    stats['truncated'] = True
                    break
                text = ' '.join(words) + '.'
                check = audit(text)
                assert check['exact']
                outputs.append({'rendered': text, 'audit': check, 'words': words})
            return
        _, i = min(choices)
        for ch in sorted(domains[i]):
            child = [set(x) for x in domains]
            child[i] = child[-1-i] = {ch}
            visit(child)
    visit([set(ALPHABET) for _ in range(n)])
    stats['status'] = 'SAT' if outputs else ('UNKNOWN' if stats['truncated'] else 'UNSAT')
    return {'target_letters': n, 'stats': stats, 'root_propagation': root_trace,
            'exact_candidates': outputs}

def run():
    controls = ['The guard reads a letter.', 'Diana writes a poem while Leon studies a map.']
    control_rows = []
    for text in controls:
        tape = normalize(text)
        forest = chart([{c} for c in tape])
        control_rows.append({'rendered': text, 'audit': audit(text),
                             'grammar_accepts': ('S', 0, len(tape)) in forest})
    return {'experiment_id': 'shared-tape-support-chart-20260920',
            'method': 'single whole-sentence packed CFG forest; root support and mirror-domain fixpoint; most-constrained character-orbit branching',
            'results': [search(n) for n in (39, 40, 44, 48, 52, 60, 72, 100)],
            'diagnostic_controls': control_rows,
            'novelty': {'claim': 'implementation topology, not a new theoretical solver family',
                        'difference': 'root-supported parse alternatives propagate globally before any lexical or boundary commitment; no fixed clause split or slot order',
                        'prior_gap': 'bounded Earley enumerates completed controls; earlier character CSP branches fixed slots and does not propagate full-parse support'},
            'provenance': {'catalogue_text': False, 'fixed_tape_reversal': False,
                           'per_search_rlaif': False, 'readability_established': False},
            'next_repair': 'The only compatible first/last letter is m, forcing subject Mira and terminal poem; their second letters i/e conflict. Add a typed person-object predicate frame with named objects, allowing the final argument to change grammatical category. This is an argument-realization repair, not another text-noun sweep.',
            'reader_facing_candidates': []}

if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
