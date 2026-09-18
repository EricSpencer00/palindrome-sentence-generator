"""Bounded title search using the project's existing overhang transitions.

This is a title-writing experiment, not a rerun of the paper's benchmark.
It permits short titles and centers inside words, and records raw candidates
for editorial review. Frequency and bigram scores do not establish meaning.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import heapq
import itertools
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wordfreq import zipf_frequency
from llm_palindrome.bigram import BigramModel
from llm_palindrome.generate import build_vocab
from llm_palindrome.lexicon import load_lexicon
from llm_palindrome.pairs import pair_vocabulary
from llm_palindrome.search import State, WordTries, _expand
from llm_palindrome.validator import is_palindrome, normalize

TERMS = """palindrome palindromes sentence sentences search searching meaning
meaningful words word text pattern patterns grammar syntax parse parsing pair
pairs join joins mirror mirrored reverse reversal build building grow growing
draw drawn onward outward middle center centre sides letters letter read reads
reading write writing edit edits test tests set sets fit fits match matches
prune pruning trim cut cuts stack step steps part parts form forms level levels
order orders loop loops nested nesting generate generation constrain constraints
rule rules name names trace scan seed seeds expand expansion language prose
compose composition emit time plan plans net nets limit limits check checks
filter filters select selection sum sums keep keeps add adds can we find make
made do stop start state states rate rates node nodes graph path paths debt
overhang walk walks tries trie bind bound bounds wordplay say says see sees
reuse use turn turns string strings run runs pass passes avoid valid void
""".split()


def substring_possible(text, tries):
    """Can text occur anywhere in a concatenation of vocabulary words?

    Start at every trie node to permit a partial word at the first boundary;
    restart at the root at every terminal to permit intervening word breaks.
    Every surviving prefix can be completed by a vocabulary word.
    """
    root = tries._fwd.root
    nodes, todo = [], [root]
    while todo:
        node = todo.pop()
        nodes.append(node)
        todo.extend(node.children.values())
    active = set(nodes)
    for ch in text:
        if any(node.words for node in active):
            active.add(root)
        active = {node.children[ch] for node in active if ch in node.children}
        if not active:
            return False
    return bool(active)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=ROOT / 'runs/title_hunt/2026-09-10.json')
    ap.add_argument('--seconds-per-anchor', type=float, default=2)
    ap.add_argument('--nodes-per-anchor', type=int, default=12000)
    ap.add_argument('--max-letters', type=int, default=60)
    ap.add_argument('--max-words', type=int, default=12)
    ap.add_argument('--min-zipf', type=float, default=3.2)
    ap.add_argument('--anchors-file', type=Path)
    args = ap.parse_args()
    vocab = pair_vocabulary(build_vocab(60000), lambda w: zipf_frequency(w, 'en'),
                            load_lexicon(str(ROOT / 'data/lexicon.txt')),
                            min_zipf=args.min_zipf)
    # Search terms remain available even when rarer than the frequency floor.
    vocab = list(dict.fromkeys(vocab + TERMS))
    tries = WordTries(vocab)
    bigram = BigramModel.from_file(str(ROOT / 'data/count_2w.txt'), vocab)
    freq = {w: zipf_frequency(w, 'en') for w in vocab}
    known = set(json.loads((ROOT / 'data/known_palindromes.json').read_text()))
    audit = {term: {'reverse': term[::-1],
                    'reverse_fits_vocabulary': substring_possible(term[::-1], tries)}
             for term in TERMS}
    print('Vocabulary:', len(vocab), flush=True)
    print('Mirror-compatible terms:', ' '.join(w for w in TERMS
          if audit[w]['reverse_fits_vocabulary']), flush=True)
    anchors = (args.anchors_file.read_text().splitlines() if args.anchors_file else TERMS)
    anchors = [a.strip().lower() for a in anchors if a.strip()]
    counters, found = {}, {}

    @lru_cache(maxsize=30000)
    def transitions(overhang, side):
        state = State(0, (), (), overhang, side)
        return tuple(_expand(state, tries, limit=10**6))

    def word_cost(w):
        return max(.25, 7 - freq.get(w, 2))

    def join_cost(a, b):
        return 0 if bigram.observed(a, b) else 3

    for anchor in anchors:
        words = tuple(anchor.split())
        if any(not audit.get(w, {'reverse_fits_vocabulary': True})['reverse_fits_vocabulary']
               for w in words):
            counters[anchor] = {'nodes': 0, 'reason': 'reversed substring absent from vocabulary language'}
            continue
        letters = normalize(anchor)
        serial = itertools.count()
        cost = sum(word_cost(w) for w in words)
        cost += sum(join_cost(a, b) for a, b in zip(words, words[1:]))
        heap = [(cost, next(serial), words, (), letters, 'L', cost)]
        seen = set()
        begin = time.monotonic()
        nodes, hits = 0, 0
        while heap and nodes < args.nodes_per_anchor:
            if nodes % 128 == 0 and time.monotonic() - begin >= args.seconds_per_anchor:
                break
            _, _, left, right, over, side, cost = heapq.heappop(heap)
            nodes += 1
            state_key = (left, right)
            if state_key in seen:
                continue
            seen.add(state_key)
            all_words = left + right
            nletters = sum(map(len, all_words))
            if over == over[::-1] and len(all_words) >= 2:
                text = ' '.join(all_words)
                assert is_palindrome(text), text
                hits += 1
                missed = sum(not bigram.observed(a, b) for a, b in zip(all_words, all_words[1:]))
                row = {'text': text, 'letters': nletters, 'words': len(all_words),
                       'unattested_joins': missed,
                       'mean_zipf': round(sum(freq.get(w, 2) for w in all_words) / len(all_words), 3),
                       'known_catalogue_entry': normalize(text) in known,
                       'anchor': anchor, 'cost': round(cost, 3)}
                found[text] = row
            if nletters >= args.max_letters or len(all_words) >= args.max_words:
                continue
            # A closed branch can extend through its center, but its empty
            # debt exposes the whole vocabulary. Keep the search bounded.
            expansions = transitions(over, side)
            if not over:
                expansions = expansions[:400]
            for place, w, new_over, new_side in expansions:
                if nletters + len(w) > args.max_letters:
                    continue
                if all_words.count(w) >= (2 if w in {'a', 'i'} else 1):
                    continue
                nl = left + (w,) if place == 'L' else left
                nr = (w,) + right if place == 'R' else right
                extra = word_cost(w)
                if place == 'L' and left:
                    extra += join_cost(left[-1], w)
                elif place == 'R' and right:
                    extra += join_cost(w, right[0])
                new_cost = cost + extra
                heapq.heappush(heap, (new_cost + .15 * len(new_over), next(serial),
                                     nl, nr, new_over, new_side, new_cost))
            if len(heap) > 60000:
                heap = heapq.nsmallest(30000, heap)
                heapq.heapify(heap)
        counters[anchor] = {'nodes': nodes, 'hits': hits,
                            'seconds': round(time.monotonic() - begin, 3),
                            'frontier_remaining': len(heap)}
        print(anchor, counters[anchor], flush=True)
    rows = sorted(found.values(), key=lambda r: (r['unattested_joins'], r['cost'], r['letters'], r['text']))
    payload = {'description': __doc__, 'config': vars(args) | {'out': str(args.out),
               'anchors_file': str(args.anchors_file) if args.anchors_file else None},
               'vocabulary_size': len(vocab),
               'vocabulary_sha256': hashlib.sha256('\n'.join(vocab).encode()).hexdigest(),
               'anchors': anchors, 'topic_terms': TERMS,
               'term_audit': audit, 'searches': counters, 'candidates': rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + '\n')
    print('Saved', len(rows), 'verified candidates to', args.out, flush=True)
    for row in rows[:60]:
        print(row, flush=True)


if __name__ == '__main__':
    main()
