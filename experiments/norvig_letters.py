"""Controlled Norvig v3 reproduction and remaining-inventory scoring ablation.

Requires unmodified pal3.py from norvig/pytudes in runs/norvig. Original
algorithm attribution: Peter Norvig, https://norvig.com/pal-alg.html .
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import time

from experiments.norvig_long import FUNCTIONS

ROOT = Path(__file__).resolve().parents[1]


def load_original():
    path = ROOT/'runs/norvig/pal3.py'
    spec = importlib.util.spec_from_file_location('norvig_original', path)
    module = importlib.util.module_from_spec(spec)
    previous = Path.cwd()
    try:
        os.chdir(path.parent)
        spec.loader.exec_module(module)
    finally:
        os.chdir(previous)
    assert module.test1() == module.test2() == 'ok'
    return module


def run(seconds, dynamic, out, max_content_uses=3, feasible=False, resume=None):
    original = load_original()
    cap = max_content_uses or float("inf")
    seed_left, seed_right, seed_L, seed_R = ['aman','aplan'], ['acanal','panama'], 'aca', ''
    if resume:
        units = json.loads((resume/'phrases.json').read_text())
        keys = [original.letters(p) for p in units]
        norm = ''.join(keys)
        assert norm == norm[::-1] and len(keys) == len(set(keys))
        assert all(key in original.DICT for key in keys)
        at = 0
        boundaries = []
        for cut_index, key in enumerate(keys[:-1], 1):
            at += len(key)
            boundaries.append((abs(2 * at - len(norm)), cut_index))
        # Choose the nearest boundary, not always the first after the centre:
        # otherwise the overlap can span multiple completed phrases and fall
        # outside the reference implementation's local closure predicate.
        cut = min(boundaries)[1]
        seed_left, seed_right = keys[:cut], keys[cut:]
        left, mirror_right = ''.join(seed_left), ''.join(seed_right)[::-1]
        if len(left) >= len(mirror_right):
            assert left.startswith(mirror_right)
            seed_L, seed_R = '', left[len(mirror_right):][::-1]
        else:
            assert mirror_right.startswith(left)
            seed_L, seed_R = mirror_right[len(left):], ''

    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)

    class StopSearch(Exception):
        pass

    class Search(original.Panama):
        def __init__(self):
            self.total = sum(map(len, seed_left + seed_right))
            self.best_letters = 0
            self.words = Counter(re.findall('[a-z]+', ' '.join(original.DICT[k] for k in seed_left + seed_right).lower()))
            self.tokens = {k: tuple(re.findall('[a-z]+', p.lower())) for k,p in original.DICT.items()}
            self.last_save = started
            self.closed = 0
            super().__init__(left=seed_left, L=seed_L, R=seed_R, right=seed_right)
            if feasible:
                self.byword = {}
                for key, words in self.tokens.items():
                    for word in set(words) - FUNCTIONS:
                        self.byword.setdefault(word, set()).add(key)
                self.active = set()
                for key in self.dict:
                    if self.inventory_allowed(key):
                        self.active.add(key)
                    else:
                        self.adjust(key, -1)
            elif dynamic:
                for key in self.set:
                    self.adjust(key, -1)

        def adjust(self, key, delta):
            for prefix in original.prefixes(key):
                self.dict.prefixes[prefix] += delta
            for suffix in original.suffixes(key):
                self.dict.suffixes[suffix] += delta

        def inventory_allowed(self, key):
            words = self.tokens[key]
            return (key not in self.set and key.isascii() and key.isalpha()
                    and bool(words) and all(a != b for a,b in zip(words, words[1:]))
                    and all(self.words[w] + n <= cap
                            for w,n in Counter(words).items() if w not in FUNCTIONS))

        def refresh(self, key):
            affected = {key}
            for word in set(self.tokens[key]) - FUNCTIONS:
                affected.update(self.byword.get(word, ()))
            for candidate in affected:
                allowed = self.inventory_allowed(candidate)
                if allowed != (candidate in self.active):
                    self.adjust(candidate, 1 if allowed else -1)
                    if allowed:
                        self.active.add(candidate)
                    else:
                        self.active.remove(candidate)

        def do(self, action):
            key = self.L if action == ',' else self.R if action == ';' else None
            super().do(action)
            if key is not None:
                self.total += len(key)
                self.words.update(self.tokens[key])
                if feasible:
                    self.refresh(key)
                elif dynamic:
                    self.adjust(key, -1)

        def undo(self, action):
            key = self.left[-1] if action == ',' else self.right[0] if action == ';' else None
            super().undo(action)
            if key is not None:
                self.total -= len(key)
                self.words.subtract(self.tokens[key])
                if feasible:
                    self.refresh(key)
                elif dynamic:
                    self.adjust(key, 1)

        def allowed_side(self, key, side):
            if not self.is_allowed(key) or not key.isascii() or not key.isalpha():
                return False
            words = self.tokens[key]
            if not words or any(a == b for a,b in zip(words,words[1:])):
                return False
            if any(self.words[w] + n > cap for w,n in Counter(words).items() if w not in FUNCTIONS):
                return False
            if side == 'L':
                return self.tokens[self.left[-1]][-1] != words[0]
            return words[-1] != self.tokens[self.right[0]][0]

        def applicable_actions(self):
            if time.monotonic() - started >= seconds:
                raise StopSearch
            actions = super().applicable_actions()
            if ',' in actions and not self.allowed_side(self.L, 'L'):
                actions.remove(',')
            if ';' in actions and not self.allowed_side(self.R, 'R'):
                actions.remove(';')
            return actions

        def check(self):
            if not self.is_palindrome():
                return
            self.closed += 1
            if self.total > self.best_letters and self.tokens[self.left[-1]][-1] != self.tokens[self.right[0]][0]:
                self.best_letters = self.total
                self.best = self.left + list(self.right)
                if time.monotonic() - self.last_save > 5:
                    self.save()

        def save(self):
            units = [self.dict[k] for k in self.best]
            text = ', '.join(units) + '.\n'
            normalized = re.sub('[^a-z]', '', text.lower())
            tokens = re.findall('[a-z]+', text.lower())
            counts = Counter(tokens)
            assert normalized == normalized[::-1]
            assert len(normalized) == self.best_letters
            assert len(self.best) == len(set(self.best))
            assert all(a != b for a,b in zip(tokens,tokens[1:]))
            assert max(n for w,n in counts.items() if w not in FUNCTIONS) <= cap
            data = dict(letters=len(normalized), words=len(tokens), phrases=len(units),
                        resume=str(resume) if resume else None, dynamic_inventory=dynamic, feasible_inventory=feasible, seconds=time.monotonic()-started,
                        steps=self.i, closures=self.closed, max_content_uses=max_content_uses,
                        sha256=hashlib.sha256(text.encode()).hexdigest(),
                        dictionary_sha256=hashlib.sha256((ROOT/'runs/norvig/npdict.txt').read_bytes()).hexdigest(),
                        original_code_sha256=hashlib.sha256((ROOT/'runs/norvig/pal3.py').read_bytes()).hexdigest(),
                        beats_published_norvig_letters=len(normalized)>90439)
            (out/'palindrome.txt').write_text(text)
            (out/'phrases.json').write_text(json.dumps(units,indent=2)+'\n')
            (out/'result.json').write_text(json.dumps(data,indent=2)+'\n')
            self.last_save = time.monotonic()
            print(json.dumps(data), flush=True)

    search = Search()
    try:
        search.search(steps=10**9)
    except StopSearch:
        pass
    search.save()
    return search


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds', type=float, default=60)
    p.add_argument('--dynamic', action='store_true')
    p.add_argument('--feasible', action='store_true')
    p.add_argument('--max-content-uses', type=int, default=3, help='0 uses Norvig phrase uniqueness without a word-count cap')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--resume', type=Path)
    args = p.parse_args()
    run(args.seconds, args.dynamic, args.out, max_content_uses=args.max_content_uses, feasible=args.feasible, resume=args.resume)
