from collections import Counter, defaultdict
import re
from wordfreq import zipf_frequency, top_n_list

from nltk.corpus import brown

pos = defaultdict(Counter)
for sent in brown.tagged_sents(tagset='universal'):
    for word, tag in sent:
        word = word.lower()
        if word.isascii() and word.isalpha(): pos[tag][word] += 1
stops = set('a an the and or but if while as of to in on at by for from with into over under after before is are was were be been being has have had do did does i you he she we they it my our his her their this that these those no not'.split())

def words(tag, limit):
    out = []
    for word, count in pos[tag].items():
        if len(word) < 2 or word in stops: continue
        z = zipf_frequency(word, 'en')
        if z < 2.5: continue
        out.append((word, count, z))
    out.sort(key=lambda row: (-row[2], -row[1], row[0]))
    return [word for word, _, _ in out[:limit]]

B = {
    'DET': 'a an the some one our my no this that his her'.split(),
    'SUBJ': words('NOUN', 1800),
    'VERB': words('VERB', 1500),
    'MOD': words('ADJ', 900) + 'a an one two three four five six seven eight nine ten'.split(),
    'NOUN': words('NOUN', 1800),
    'PROPN': 'ada anna diana eve iris lena lisa maria maya nora noah olivia otto sara uma alan eric emil ella ian ava art leo susan ruth james jane john mary paul peter rose mark anne clara david nina tara tina lori lara sam ben tom ray eli'.split(),
}
for key in B: B[key] = list(dict.fromkeys(B[key]))
T = ['DET', 'SUBJ', 'VERB', 'MOD', 'NOUN', 'DET', 'SUBJ', 'VERB', 'PROPN']

def tape(word): return ''.join(c for c in word if c.isalpha())

def search(limit=3_000_000):
    nodes, solutions = 0, []
    def dfs(li, ri, left, right, assigned, score):
        nonlocal nodes
        nodes += 1
        if nodes > limit: return
        reverse_right = right[::-1]
        match = min(len(left), len(reverse_right))
        if left[:match] != reverse_right[:match]: return
        if li > ri:
            if left == reverse_right: solutions.append((' '.join(assigned), score))
            return
        sides = []
        if len(left) <= len(reverse_right): sides.append('L')
        if len(reverse_right) <= len(left): sides.append('R')
        if len(sides) == 2:
            sides.sort(key=lambda side: len(B[T[li if side == 'L' else ri]]))
        for side in sides:
            kind = T[li if side == 'L' else ri]
            for word in B[kind]:
                if kind in {'SUBJ', 'VERB', 'NOUN', 'PROPN'} and word in assigned: continue
                if side == 'L':
                    new_left = left + tape(word)
                    match = min(len(new_left), len(reverse_right))
                    if new_left[:match] != reverse_right[:match]: continue
                    dfs(li + 1, ri, new_left, right, assigned + [word], score + zipf_frequency(word, 'en'))
                else:
                    new_right = tape(word) + right
                    new_reverse = new_right[::-1]
                    match = min(len(left), len(new_reverse))
                    if left[:match] != new_reverse[:match]: continue
                    dfs(li, ri - 1, left, new_right, [word] + assigned, score + zipf_frequency(word, 'en'))
    dfs(0, len(T) - 1, '', '', [], 0.)
    return nodes, solutions

if __name__ == '__main__':
    n, sol = search()
    print('nodes', n, 'solutions', len(sol))
    for text, score in sorted(sol, key=lambda row: (len(tape(row[0])), row[1]), reverse=True)[:100]:
        print(len(tape(text)), round(score, 2), text)
