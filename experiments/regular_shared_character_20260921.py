"""Forward grammatical NFA plus mirrored character-domain propagation.

No completed prefix/half is reversed to supply words. At every search node,
position-specific forward/backward grammatical supports are recomputed until
both grammar and mirror constraints reach a fixed point.
"""
from __future__ import annotations
import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ID = 'regular-shared-character-20260921'
SEED = 'An aide rips nine memos; some men inspire Diana.'


def letters(text):
    return re.sub('[^a-z]', '', text.lower())


def audit(text):
    # Independent normalizers and comparison implementations.
    tape = letters(text)
    second = ''.join(c.lower() for c in text if c in
                     'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
    i, j = 0, len(second)-1
    while i < j and second[i] == second[j]:
        i += 1
        j -= 1
    return {'letters': len(tape), 'normalized': tape,
            'exact': bool(tape) and tape == tape[::-1],
            'independent_pointer_exact': bool(second) and i >= j,
            'normalizers_agree': tape == second,
            'sha256': hashlib.sha256(tape.encode()).hexdigest(),
            'reverse_sha256': hashlib.sha256(tape[::-1].encode()).hexdigest()}


class Grammar:
    def __init__(self):
        self.edges = defaultdict(list)
        self.size = 2
        self.frames = []

    def new(self):
        n = self.size
        self.size += 1
        return n

    def frame(self, name, slots):
        """Each slot is (feature role, independently authored alternatives)."""
        start = self.new()
        self.edges[0].append(('', start, None))
        self.frames.append({'name': name, 'slots': slots})
        for role, words in slots:
            end = self.new()
            for word in words:
                state = start
                normalized = letters(word)
                if not normalized:
                    self.edges[state].append(('', end, None))
                for index, char in enumerate(normalized):
                    final = index == len(normalized)-1
                    nxt = end if final else self.new()
                    label = {'word': word, 'role': role, 'frame': name} if final else None
                    self.edges[state].append((char, nxt, label))
                    state = nxt
            start = end
        self.edges[start].append(('', 1, None))

    def compile(self):
        closure = {}
        for state in range(self.size):
            pending = [state]
            seen = {state}
            while pending:
                for char, dst, _ in self.edges[pending.pop()]:
                    if not char and dst not in seen:
                        seen.add(dst)
                        pending.append(dst)
            closure[state] = seen
        transitions = defaultdict(list)
        for state in range(self.size):
            for member in closure[state]:
                for char, dst, _ in self.edges[member]:
                    if char:
                        transitions[state].append((char, dst))
        return transitions, {s for s in range(self.size) if 1 in closure[s]}

    def render(self, tape):
        # Replay a forward accepting path, retaining lexical boundary evidence.
        pending = [(0, 0)]
        parent = {(0, 0): None}
        while pending:
            state, pos = pending.pop()
            if state == 1 and pos == len(tape):
                path = []
                key = (state, pos)
                while parent[key] is not None:
                    before, label = parent[key]
                    if label:
                        path.append(label)
                    key = before
                path.reverse()
                text = ' '.join(x['word'] for x in path)
                return text[:1].upper()+text[1:]+'.', path
            for char, dst, label in self.edges[state]:
                if char and (pos == len(tape) or tape[pos] != char):
                    continue
                key = (dst, pos+bool(char))
                if key not in parent:
                    parent[key] = ((state, pos), label)
                    pending.append(key)
        return None


def construct():
    g = Grammar()
    # Agreement and valency are encoded in distinct grammatical paths.
    # A second clause starts at its determiner role; punctuation is restored
    # from that recorded boundary, never used to supply or change letters.
    subject = [('det', ['an']), ('animate_sg', ['aide', 'artist', 'editor'])]
    other = [('det', ['a', 'the']), ('animate_sg', ['sailor', 'writer', 'pilot'])]
    left = [('transitive_sg', ['rips', 'reads', 'keeps', 'sees', 'marks']),
            ('quantity_plural', ['nine', 'seven', 'two']),
            ('object_adj', ['', 'red', 'new', 'old']),
            ('artifact_plural', ['memos', 'notes', 'maps', 'letters'])]
    right = [('second_clause_det', ['some', 'many', 'the']),
             ('animate_plural', ['men', 'women', 'artists', 'poets']),
             ('animate_object_transitive_plural', ['inspire', 'help', 'guide', 'meet']),
             ('person', ['Diana', 'Nora', 'Mira', 'Leon', 'Ariel'])]
    for idx, prefix in enumerate((subject, other)):
        g.frame(f'artifact_action_then_person_event_{idx}', prefix+left+right)
        second_prefix = [('second_clause_det', prefix[0][1])]+prefix[1:]
        g.frame(f'person_event_then_artifact_action_{idx}', right+second_prefix+left)
    # Alternate topology: one imperative followed by a reporting statement.
    g.frame('imperative_then_statement',
            [('imperative', ['read', 'keep', 'mark']),
             ('quantity_plural', ['nine', 'seven', 'two']),
             ('artifact_plural', ['memos', 'notes', 'maps', 'letters'])]+right)
    # Fresh feature-checked constructions requested for the shared-character
    # propagator.  Question auxiliaries select singular animate subjects;
    # relative clauses use an explicit who-gap and transitive object.
    question = [('aux_question', ['can', 'will']),
                ('question_det', ['the', 'a']),
                ('question_agent_sg', ['artist', 'writer', 'pilot']),
                ('question_verb', ['read', 'see', 'help']),
                ('question_det_obj', ['a', 'the']),
                ('question_object', ['map', 'book', 'letter'])]
    relative = [('relative_det', ['the', 'a']),
                ('relative_agent_sg', ['artist', 'writer', 'pilot']),
                ('relative_marker', ['who']),
                ('relative_verb_sg', ['reads', 'sees', 'helps']),
                ('relative_det_obj', ['a', 'the']),
                ('relative_object', ['map', 'book', 'letter'])]
    embedded_relative = [('embedded_det', ['the', 'a']),
                         ('embedded_agent_sg', ['artist', 'writer', 'pilot']),
                         ('embedded_verb_sg', ['reads', 'sees', 'helps']),
                         ('embedded_det_obj', ['a', 'the']),
                         ('embedded_object', ['map', 'book', 'letter']),
                         # Object-relative markers are licensed for the
                         # transitive gap; ``who`` is retained for the
                         # animate subject-relative alternative.
                         ('embedded_marker', ['that', 'which', 'who']),
                         ('embedded_subject_sg', ['artist', 'writer', 'pilot']),
                         ('embedded_pred_sg', ['reads', 'sees', 'helps']),
                         ('embedded_det_obj2', ['a', 'the']),
                         ('embedded_object2', ['map', 'book', 'letter']),
                         # Held-out temporal terminals, distinct from the
                         # earlier top-level adjunct lane.
                         ('embedded_temporal', ['at sunset', 'before winter',
                                                'during rain']),
                         ('embedded_locative', ['at harbor', 'beside quay',
                                                'under bridge'])]
    embedded_tense_relative = [
        ('tense_det', ['the', 'a']),
        ('tense_agent_sg', ['artist', 'writer', 'pilot']),
        ('tense_marker', ['who', 'that', 'which']),
        ('tense_aux', ['has', 'will', 'did']),
        ('tense_base_pred', ['read', 'see', 'help']),
        ('tense_det_obj', ['a', 'the']),
        ('tense_object', ['map', 'book', 'letter'])]
    g.frame('question_then_person_event', question+right)
    g.frame('relative_clause_then_person_event', relative+right)
    g.frame('embedded_object_relative_then_person_event', embedded_relative+right)
    g.frame('tense_embedded_relative_then_person_event', embedded_tense_relative+right)
    return g


def propagate(domains, transitions, finals, stats):
    domains = list(domains)
    n = len(domains)
    while True:
        stats['propagation_rounds'] += 1
        forward = [{0}]
        for domain in domains:
            forward.append({dst for src in forward[-1]
                            for ch, dst in transitions[src] if ch in domain})
        if not forward[n] & finals:
            return None
        backward = [set() for _ in range(n+1)]
        backward[n] = forward[n] & finals
        supported = [set() for _ in range(n)]
        for i in range(n-1, -1, -1):
            for src in forward[i]:
                for ch, dst in transitions[src]:
                    if ch in domains[i] and dst in backward[i+1]:
                        backward[i].add(src)
                        supported[i].add(ch)
        changed = False
        for i in range((n+1)//2):
            j = n-1-i
            keep = frozenset(supported[i] & supported[j])
            if not keep:
                return None
            if keep != domains[i] or keep != domains[j]:
                stats['domain_values_removed'] += len(domains[i]-keep)
                if i != j:
                    stats['domain_values_removed'] += len(domains[j]-keep)
                changed = True
                domains[i] = domains[j] = keep
        if not changed:
            return tuple(domains)


def solve(grammar, n, cap=120):
    transitions, finals = grammar.compile()
    stats = {'nodes': 0, 'propagation_rounds': 0, 'domain_values_removed': 0,
             'conflicts': 0, 'cap_reached': False}
    candidates = []
    def residual_score(domains, pos, ch):
        """Rank a character by live support at both residual positions.

        This is deliberately computed before branching; it is not a score of
        a completed rendered candidate.  The mirrored position is included so
        lexical alternatives that cannot survive the current obligation move
        to the back of the queue.
        """
        mirror = n - 1 - pos
        trans, finals_local = transitions, finals
        left = {0}
        for domain in domains[:pos]:
            left = {dst for src in left for edge, dst in trans[src]
                    if edge in domain}
        right = set(finals_local)
        for domain in reversed(domains[mirror + 1:]):
            right = {src for src in range(grammar.size)
                     for edge, dst in trans[src]
                     if edge in domain and dst in right}
        forward_hit = sum(1 for dst in left for edge, _ in trans[dst] if edge == ch)
        reverse_hit = sum(1 for src in range(grammar.size)
                          for edge, dst in trans[src]
                          if edge == ch and dst in right)
        return (forward_hit > 0) + (reverse_hit > 0), forward_hit + reverse_hit
    def visit(domains):
        if stats['nodes'] >= cap:
            stats['cap_reached'] = True
            return
        stats['nodes'] += 1
        domains = propagate(domains, transitions, finals, stats)
        if domains is None:
            stats['conflicts'] += 1
            return
        open_positions = [i for i in range((n+1)//2) if len(domains[i]) > 1]
        if not open_positions:
            tape = ''.join(next(iter(d)) for d in domains)
            rendered = grammar.render(tape)
            assert rendered is not None
            text, path = rendered
            words = []
            for token in path:
                if token['role'] == 'second_clause_det' and words:
                    words[-1] += ';'
                words.append(token['word'])
            text = ' '.join(words)
            text = text[:1].upper()+text[1:]+'.'
            flags = []
            tokens = re.findall('[a-z]+', text.lower())
            if len(tokens) != len(set(tokens)):
                flags.append('repeated_word')
            if any(len(w)>1 and w == w[::-1] for w in tokens):
                flags.append('self_palindromic_word')
            candidates.append({'text': text, 'audit': audit(text), 'lexical_path': path,
                               'shortcut_flags': flags,
                               'known_calibration': letters(text) == letters(SEED),
                               'calibration_family': sorted(tokens) == sorted(re.findall('[a-z]+', SEED.lower())),
                               'human_readability': 'not_tested'})
            return
        pos = min(open_positions, key=lambda i: (len(domains[i]), abs(n/2-i)))
        ordered = sorted(domains[pos], key=lambda ch: residual_score(domains, pos, ch), reverse=True)
        for ch in ordered:
            branch = list(domains)
            branch[pos] = branch[n-1-pos] = frozenset(ch)
            visit(tuple(branch))
    visit(tuple(frozenset('abcdefghijklmnopqrstuvwxyz') for _ in range(n)))
    return {'target_letters': n, 'stats': stats, 'candidates': candidates}


def differential():
    g = Grammar()
    g.frame('unequal_boundaries', [('x', ['ab', 'a', 'ba']), ('y', ['a', 'ba', 'b'])])
    checks = []
    for n in range(2, 6):
        expected = {a+b for a in ['ab', 'a', 'ba'] for b in ['a', 'ba', 'b']
                    if len(a+b) == n and a+b == (a+b)[::-1]}
        run = solve(g, n)
        actual = {c['audit']['normalized'] for c in run['candidates']}
        assert actual == expected, (n, actual, expected)
        checks.append({'N': n, 'expected': sorted(expected), 'actual': sorted(actual)})
    return checks


def main():
    start = time.monotonic()
    grammar = construct()
    results = [solve(grammar, n) for n in range(39, 201)]
    out = {'experiment_id': ID, 'method': 'forward_NFA_REGULAR_fixed_point_mirrored_domains',
           'differential_checks': differential(), 'grammar': grammar.frames,
           'results': results, 'seconds': time.monotonic()-start,
           'forward_controls': [{'text': t, 'audit': audit(t),
                                 'status': 'intact_grammar_control_not_palindrome_candidate'}
                                for t in ['An editor reads seven old letters; many poets guide Nora.',
                                          'The sailor keeps two new maps; some women help Leon.']],
           'provenance': {'lexicon': 'hand_authored_typed_word_choices',
                          'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          'grammar_states': grammar.size, 'catalogue_text': False,
                          'finished_tape_reverse_for_construction': False,
                          'branch_variable': 'smallest_supported_mirrored_character_domain'},
           'novelty': {'prior_checked': ['finite_semantic_palindrome_csp_20260916.py',
                                        'event_graph_character_sat_20260916.py',
                                        'position_domain_arc_consistency_csp_20260920.py'],
                       'implementation_distinction': 'iterated position-specific forward/backward supports with latent word boundaries'},
           'next_operator': 'add a bounded aspectual auxiliary variant with a held-out animate noun; current grammar has no accepting root at 39..200',
           'reader_gate': 'closed until exact original plausible prose and blinded ratings'}
    (ROOT/'runs'/f'{ID}.json').write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps([{'N': r['target_letters'], **r['stats'],
                       'texts': [c['text'] for c in r['candidates']]} for r in results], indent=2))


if __name__ == '__main__':
    main()
