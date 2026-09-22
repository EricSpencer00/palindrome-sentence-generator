"""Packed character intersection around one fixed, meaningful clause seam.

Each grammar slot is a shared NFA boundary, never a list of complete strings.
Equal-character transitions advance opposing NFA states before lexical paths
are rendered. The fixed middle is inherited from the readable incumbent;
both surrounding clause fragments remain variable, including their lengths.
"""
from collections import defaultdict, deque
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SEED = "An aide rips nine memos; some men inspire Diana."


def norm(text):
    return re.sub('[^a-z]', '', text.lower())


def audit(text):
    tape = norm(text)
    mismatches = [(i, tape[i], tape[-i-1]) for i in range(len(tape)//2)
                  if tape[i] != tape[-i-1]]
    return dict(letters=len(tape), exact=bool(tape) and not mismatches,
                first_mismatch=mismatches[:1], normalized=tape,
                sha256=hashlib.sha256(tape.encode()).hexdigest(),
                reverse_sha256=hashlib.sha256(tape[::-1].encode()).hexdigest())


class Grammar:
    def __init__(self):
        self.count = 1
        self.edges = []
        self.epsilon = defaultdict(list)
        self.start = self.finish = 0

    def new(self):
        result = self.count
        self.count += 1
        return result

    def slot(self, alternatives, role):
        before, after = self.finish, self.new()
        for phrase in alternatives:
            if not phrase:
                self.epsilon[before].append(after)
                continue
            current = before
            tape = norm(phrase)
            for index, char in enumerate(tape):
                following = after if index == len(tape)-1 else self.new()
                self.edges.append((current, following, char,
                                   phrase if index == 0 else '', role))
                current = following
        self.finish = after


def grammar():
    g = Grammar()
    # Independent, ordinary English alternatives; the incumbent is a recovery
    # control, not a novel candidate. Optional slots vary both window lengths.
    g.slot(('an aide', 'a nurse', 'a clerk', 'the aide', 'the nurse',
            'an old aide', 'a tired aide', 'a quiet aide', 'one aide',
            'a senior aide', 'another aide', 'a male aide'), 'subject:singular')
    g.slot(('', 'now', 'often', 'still', 'quietly', 'slowly', 'carefully'), 'adverb')
    g.slot(('rips', 'reads', 'writes', 'sends', 'files', 'copies', 'shreds',
            'signs', 'sorts', 'edits', 'saves', 'keeps'), 'verb:singular:document')
    for word in ('nine', 'memos;', 'some', 'men'):
        g.slot((word,), 'fixed:incumbent-seam')
    g.slot(('', 'now', 'often', 'still', 'quietly', 'greatly', 'truly'), 'adverb')
    g.slot(('inspire', 'admire', 'praise', 'encourage', 'assist', 'help',
            'guide', 'support', 'surprise', 'impress', 'thank'), 'verb:plural:human')
    g.slot(('Diana', 'Anna', 'Nora', 'Leon', 'the aide', 'a nurse', 'the clerk',
            'one writer', 'a poet', 'the writer', 'our aide', 'the old aide'), 'object:human')
    return g


def intersect(g, max_letters=120, cap=100000):
    @lru_cache(None)
    def closure(node):
        reached = {node}
        for nxt in g.epsilon[node]:
            reached.update(closure(nxt))
        return frozenset(reached)

    forward, backward = defaultdict(list), defaultdict(list)
    for edge_id, (left, right, char, _, _) in enumerate(g.edges):
        forward[left].append(edge_id)
        backward[right].append(edge_id)
    reverse_epsilon = defaultdict(list)
    for left, rights in list(g.epsilon.items()):
        for right in rights:
            reverse_epsilon[right].append(left)

    @lru_cache(None)
    def backclosure(node):
        reached = {node}
        for nxt in reverse_epsilon[node]:
            reached.update(backclosure(nxt))
        return frozenset(reached)

    @lru_cache(None)
    def outgoing(node):
        result = defaultdict(list)
        for src in closure(node):
            for edge_id in forward[src]:
                result[g.edges[edge_id][2]].append(edge_id)
        return result

    @lru_cache(None)
    def incoming(node):
        result = defaultdict(list)
        for dst in backclosure(node):
            for edge_id in backward[dst]:
                result[g.edges[edge_id][2]].append(edge_id)
        return result

    queue = deque([(g.start, g.finish, (), ())])
    seen, candidates, dead = set(), {}, []
    transitions = 0
    while queue and len(seen) < cap:
        left, right, lp, rp = queue.popleft()
        key = (left, right, len(lp))
        if key in seen:
            continue
        seen.add(key)
        middle_paths = [()] if right in closure(left) else []
        for ids in outgoing(left).values():
            middle_paths.extend((edge_id,) for edge_id in ids
                                if right in closure(g.edges[edge_id][1]))
        for middle in middle_paths:
            ids = lp + middle + rp[::-1]
            text = ' '.join(g.edges[i][3] for i in ids if g.edges[i][3])
            text = text[:1].upper() + text[1:] + '.'
            check = audit(text)
            # Audit connectivity separately from palindrome equality.
            node = g.start
            connected = True
            for i in ids:
                src, dst, _, _, _ = g.edges[i]
                connected &= src in closure(node)
                node = dst
            connected &= g.finish in closure(node)
            assert connected and check['exact']
            candidates[check['normalized']] = dict(rendered=text, audit=check,
                accepting_path=list(ids), connected=connected,
                novel_relative_to_seed=norm(text) != norm(SEED))
        if 2*len(lp)+2 > max_letters:
            continue
        fw, bw = outgoing(left), incoming(right)
        common = fw.keys() & bw.keys()
        if not common:
            dead.append(dict(matched_pairs=len(lp), left_state=left,
                right_state=right, left_next=sorted(fw), right_next=sorted(bw),
                left_partial=' '.join(g.edges[i][3] for i in lp if g.edges[i][3]),
                right_partial=' '.join(g.edges[i][3] for i in rp[::-1] if g.edges[i][3])))
            dead.sort(key=lambda row:-row['matched_pairs'])
            del dead[30:]
        for char in sorted(common):
            for a in fw[char]:
                for b in bw[char]:
                    queue.append((g.edges[a][1], g.edges[b][0], lp+(a,), rp+(b,)))
                    transitions += 1
    return dict(candidates=sorted(candidates.values(), key=lambda x:-x['audit']['letters']),
                states=len(seen), transitions=transitions, cap_reached=bool(queue),
                dead_frontiers=dead, grammar_states=g.count,
                grammar_character_edges=len(g.edges))


def run():
    result = intersect(grammar())
    result.update(experiment_id='packed-seam-grammar-20260927',
        method='shared-slot NFA with live equal-character state-pair intersection',
        fixed_seam='nine memos; some men',
        incumbent=SEED,
        represented_surface_paths=12*7*12*7*11*12,
        witness_policy='One accepting witness per state pair and matched length; exact existence search, not exhaustive distinct-text enumeration.',
        provenance=dict(complete_sentence_enumeration=False, inherited_seed=True,
            catalogue_text=False, per_candidate_rlaif=False,
            lexical_alternatives='fresh typed ordinary-English banks in this file',
            generated_sentence_count_before_intersection=0,
            generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        novelty_preflight=dict(novel_algorithm_claim=False,
            distinction='packed shared boundaries and symmetric editable incumbent windows; prior capped complete-derivation banks are not used'),
        reader_test=dict(status='not collected',
            protocol='Any novel closure: randomized blinded English/coherence ratings alongside length-matched intact prose and shuffled controls.'),
        next_expansion='If only the incumbent survives, move the fixed seam outward to allow object quantity and plural subject alternatives jointly; preserve this same packed solver.')
    return result


if __name__ == '__main__':
    result = run()
    path = ROOT/'runs/packed-seam-grammar-20260927.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
