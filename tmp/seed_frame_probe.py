import re
from wordfreq import zipf_frequency

B = {
    "DET": "a an the some one our my no this that".split(),
    "SUBJ": "aide artist baker carpenter captain child clerk doctor driver farmer friend gardener guide keeper man master men nurse pilot poet porter reader sailor scholar singer smith soldier student teacher woman writer agent angel author actor elder monk mother father sister brother crew team people girls boys birds dogs cats diana maria nora lena anna iris otto ada eve noah olivia uma sara maya".split(),
    "VERB": "aids asks bakes calls carries charts checks cleans cooks cuts draws finds folds gives guides guards helps holds inspires keeps leads likes maps marks mends meets moves needs opens packs paints plants reads repairs rides rips sends sets shares shows sings speaks spins takes teaches tells trains uses visits walks watches waters writes inspire review reviews sees knows loves makes brings builds saves names".split(),
    "MOD": "a an one two three four five six seven eight nine ten old new red blue green small large quiet fresh warm cool kind clear young wise bright dark long short soft hard fine fair true good last first many some more each odd even".split(),
    "ADJ": "old new red blue green small large quiet fresh warm cool kind clear young wise bright dark long short soft hard fine fair true good patient gentle silent brave".split(),
    "OBJ": "aid aide apple atlas boat book bread bridge chart chair candle car case cave coin cove dawn deer desk door dream drum flag flower gate garden glass horse island key lamp lantern letter line list map meal memo memos note notes page paper path pen poem pond rope road room sail seal ship sign song stone story table tent tool toy train tree vase wall water wheel window yard action answer idea image message mission number plan report signal task truth word words".split(),
    "NOUN": "aid aide apple atlas boat book bread bridge chart chair candle car case cave coin cove dawn deer desk door dream drum flag flower gate garden glass horse island key lamp lantern letter line list map meal memo memos note notes page paper path pen poem pond rope road room sail seal ship sign song stone story table tent tool toy train tree vase wall water wheel window yard action answer idea image message mission number plan report signal task truth word words".split(),
    "PROPN": "ada anna diana eve iris lena lisa maria maya nora noah olivia otto sara uma alan eric emil ella ian ava art leo susan ruth".split(),
    "ADP": "at by in on near under over after before beside toward from with".split(),
}
for k in B:
    B[k] = list(dict.fromkeys(B[k]))
TEMPLATES = [
    ["DET", "SUBJ", "VERB", "MOD", "OBJ", "DET", "SUBJ", "VERB", "PROPN"],
    ["DET", "SUBJ", "VERB", "MOD", "ADJ", "OBJ", "DET", "SUBJ", "VERB", "PROPN"],
    ["DET", "SUBJ", "VERB", "MOD", "OBJ", "DET", "SUBJ", "VERB", "ADJ", "PROPN"],
    ["DET", "SUBJ", "VERB", "MOD", "OBJ", "ADP", "NOUN", "DET", "SUBJ", "VERB", "PROPN"],
    ["DET", "SUBJ", "VERB", "MOD", "OBJ", "DET", "SUBJ", "VERB", "PROPN", "ADP", "NOUN"],
    ["DET", "SUBJ", "VERB", "MOD", "ADJ", "OBJ", "DET", "SUBJ", "VERB", "ADJ", "PROPN"],
]

def tape(word):
    return "".join(c for c in word if c.isalpha())

def search(template, limit=2_000_000):
    nodes = 0
    solutions = []

    def dfs(li, ri, left, right, words, score):
        nonlocal nodes
        nodes += 1
        if nodes > limit:
            return
        reverse_right = right[::-1]
        matched = min(len(left), len(reverse_right))
        if left[:matched] != reverse_right[:matched]:
            return
        if li > ri:
            if left == reverse_right:
                solutions.append((" ".join(words), score))
            return
        sides = []
        if len(left) <= len(reverse_right):
            sides.append("L")
        if len(reverse_right) <= len(left):
            sides.append("R")
        if len(sides) == 2:
            sides.sort(key=lambda side: len(B[template[li if side == "L" else ri]]))
        for side in sides:
            kind = template[li if side == "L" else ri]
            for word in B[kind]:
                if kind not in {"DET", "MOD"} and word in words:
                    continue
                if side == "L":
                    new_left = left + tape(word)
                    matched = min(len(new_left), len(reverse_right))
                    if new_left[:matched] != reverse_right[:matched]:
                        continue
                    dfs(li + 1, ri, new_left, right, words + [word], score + zipf_frequency(word, "en"))
                else:
                    new_right = tape(word) + right
                    new_reverse_right = new_right[::-1]
                    matched = min(len(left), len(new_reverse_right))
                    if left[:matched] != new_reverse_right[:matched]:
                        continue
                    dfs(li, ri - 1, left, new_right, [word] + words, score + zipf_frequency(word, "en"))

    dfs(0, len(template) - 1, "", "", [], 0.0)
    return nodes, solutions

if __name__ == "__main__":
    for template in TEMPLATES:
        nodes, solutions = search(template)
        print("template", template, "nodes", nodes, "solutions", len(solutions))
        for text, score in sorted(solutions, key=lambda row: (len(tape(row[0])), row[1]), reverse=True)[:20]:
            print(len(tape(text)), round(score, 1), text)
