"""Bounded Norvig/Hoey inward DFS, maximizing letters with no reused phrase.

Source algorithm: https://norvig.com/pal-alg.html . Dictionary:
https://norvig.com/npdict.txt . This generates a noun-list palindrome, not prose.
"""
from __future__ import annotations
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import re
import time

from llm_palindrome.search import WordTries, consume


def letters(text):
    return ''.join(c for c in text.lower() if 'a' <= c <= 'z')


def load_phrases(path):
    phrases = {}
    for line in path.read_text().splitlines():
        phrase = ' '.join(re.findall('[a-z]+', line.lower()))
        words = phrase.split()
        if words and all(a != b for a, b in zip(words, words[1:])):
            phrases.setdefault(letters(phrase), phrase)
    return list(phrases.values())


def render(left, right, paragraph_words=100):
    units = left + list(reversed(right))
    paragraphs, chunk, count = [], [], 0
    for phrase in units:
        chunk.append(phrase)
        count += len(phrase.split())
        if count >= paragraph_words:
            paragraphs.append(', '.join(chunk))
            chunk, count = [], 0
    if chunk:
        paragraphs.append(', '.join(chunk))
    text = ',\n\n'.join(paragraphs) + '.'
    return text[0].upper() + text[1:]


FUNCTIONS = set('a an the and or of to in on at for with by'.split())


def search(phrases, seconds, seed, out, max_content_uses=3, stagnation_nodes=2000):
    started = time.monotonic()
    rng = random.Random(seed)
    tries = WordTries(phrases)
    keys = {p: letters(p) for p in tries.words}
    counts = {p: Counter(p.split()) for p in tries.words}
    left, right = ['a man', 'a plan'], ['panama', 'a canal']
    used = {letters(p) for p in left + right}
    word_counts = Counter(' '.join(left + right).split())
    total = sum(map(len, used))
    best = total
    nodes = backtracks = 0
    last_improvement = 0
    backjumps = 0
    out.mkdir(parents=True, exist_ok=True)

    @lru_cache(maxsize=8192)
    def edges(over, side):
        menu = (tries.right_candidates(over, len(phrases))
                if side == 'L' or not over else
                tries.left_candidates(over, len(phrases)))
        placement = 'R' if side == 'L' or not over else 'L'
        result = []
        for p in menu:
            match = consume(keys[p][::-1] if placement == 'R' else keys[p], over)
            if match is None:
                continue
            debt, flipped = match
            owner = placement if flipped else ('L' if placement == 'R' else 'R')
            result.append((p, placement, debt, owner))
        return result

    def options(over, side):
        result = list(edges(over, side))
        rng.shuffle(result)
        return iter(result)

    def save():
        text = render(left, right)
        norm = letters(text)
        assert norm == norm[::-1], 'broken palindrome'
        units = left + list(reversed(right))
        assert len({letters(p) for p in units}) == len(units)
        tokens = re.findall('[a-z]+', text.lower())
        assert all(a != b for a, b in zip(tokens, tokens[1:]))
        metadata = dict(letters=len(norm), words=len(tokens), phrases=len(units),
                        paragraphs=text.count('\n\n') + 1, seed=seed,
                        elapsed_seconds=time.monotonic()-started, nodes=nodes,
                        backtracks=backtracks, backjumps=backjumps,
                        stagnation_nodes=stagnation_nodes, distinct_phrases=len(units),
                        maximum_content_word_uses=max(word_counts[w] for w in word_counts if w not in FUNCTIONS),
                        max_content_uses=max_content_uses,
                        sha256=hashlib.sha256((text+'\n').encode()).hexdigest(),
                        source='Norvig npdict phrase inventory; newly searched arrangement',
                        algorithm='Norvig/Hoey inward overhang DFS with backtracking',
                        claim='Longest closed result found in this bounded run; not a proven maximum or coherent prose')
        (out/'palindrome.txt').write_text(text+'\n')
        (out/'result.json').write_text(json.dumps(metadata,indent=2)+'\n')
        (out/'phrases.json').write_text(json.dumps(units,indent=2)+'\n')
        print(json.dumps(metadata), flush=True)

    save()
    # Frames store iterator and the move to undo when this frame is exhausted.
    stack = [(options('aca', 'R'), None)]
    while stack and time.monotonic()-started < seconds:
        # A bounded longest-path heuristic: abandon a stagnant local subtree.
        # This sacrifices exhaustiveness, never the saved best or invariants.
        if stagnation_nodes and nodes - last_improvement >= stagnation_nodes:
            for _ in range(min(100, len(stack)-1)):
                _, (p, placement) = stack.pop()
                (left if placement == 'L' else right).pop()
                used.remove(keys[p]); word_counts.subtract(counts[p]); total -= len(keys[p])
                backtracks += 1
            backjumps += 1
            last_improvement = nodes
        move = next(stack[-1][0], None)
        if move is None:
            _, undo = stack.pop()
            if undo:
                p, placement = undo
                (left if placement == 'L' else right).pop()
                used.remove(keys[p]); word_counts.subtract(counts[p]); total -= len(keys[p])
                backtracks += 1
            continue
        p, placement, debt, owner = move
        if keys[p] in used:
            continue
        if any(word_counts[w] + n > max_content_uses for w,n in counts[p].items() if w not in FUNCTIONS):
            continue
        words = p.split()
        if placement == 'L' and left[-1].split()[-1] == words[0]:
            continue
        if placement == 'R' and words[-1] == right[-1].split()[0]:
            continue
        closed = debt == debt[::-1]
        # At closure the two inward edges become adjacent in the final text.
        llast = words[-1] if placement == 'L' else left[-1].split()[-1]
        rfirst = words[0] if placement == 'R' else right[-1].split()[0]
        if not closed and not edges(debt, owner):
            continue
        (left if placement == 'L' else right).append(p)
        used.add(keys[p]); word_counts.update(counts[p]); total += len(keys[p]); nodes += 1
        if closed and llast != rfirst and total > best:
            best = total
            # Save each improvement: interruption always leaves a verified artifact.
            save()
            last_improvement = nodes
        stack.append((options(debt, owner), (p, placement)))
    return json.loads((out/'result.json').read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dictionary', type=Path, default=Path('runs/norvig/npdict.txt'))
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--stagnation-nodes', type=int, default=2000)
    parser.add_argument('--max-content-uses', type=int, default=3)
    parser.add_argument('--out', type=Path, default=Path('runs/norvig-long'))
    args = parser.parse_args()
    search(load_phrases(args.dictionary), args.seconds, args.seed, args.out, args.max_content_uses, args.stagnation_nodes)


if __name__ == '__main__':
    main()
