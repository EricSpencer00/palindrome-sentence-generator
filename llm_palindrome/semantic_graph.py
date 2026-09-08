"""Build a sentence-pair path using semantic similarity as a hard edge gate."""
from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence

from .hierarchy import CONNECTIVES

_EDGE_CACHE: dict[tuple[int, int, float], dict[int, list[int]]] = {}


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    aa = math.sqrt(sum(x * x for x in a))
    bb = math.sqrt(sum(y * y for y in b))
    return dot / (aa * bb) if aa and bb else 0.0


def paired_path(pairs: Sequence[dict], centre: str,
                embeddings: Mapping[str, Sequence[float]], threshold: float,
                want: int, seed: int = 0, restarts: int = 200,
                max_bigram_uses: int = 2,
                max_template_uses: int = 2,
                max_content_word_uses: int = 3) -> list[dict]:
    """Find an outer-to-inner path whose two reading directions stay related.

    For adjacent pair nodes A,B in outer-to-inner order, the paragraph reads
    ``A.left -> B.left`` on the way in and ``B.right -> A.right`` on the way
    out. Both similarities must clear ``threshold``. Similarity never ranks
    complete paragraphs; below the threshold an edge does not exist, above it
    all edges are equal and seeded random exploration decides.
    """
    rng = random.Random(seed)
    centre_vec = embeddings[centre]
    inner = [index for index, pair in enumerate(pairs)
             if cosine(embeddings[pair["left"]], centre_vec) >= threshold
             and cosine(centre_vec, embeddings[pair["right"]]) >= threshold]
    # Edges depend only on the threshold and pair texts. Cache them once per
    # search instead of recomputing 768-dimensional cosines on every restart.
    cache_key = (id(pairs), id(embeddings), threshold)
    outer_for = _EDGE_CACHE.get(cache_key)
    if outer_for is None:
        outer_for = {}
        for current_index, current in enumerate(pairs):
            outer_for[current_index] = [
                index for index, pair in enumerate(pairs) if index != current_index
                and cosine(embeddings[pair["left"]],
                           embeddings[current["left"]]) >= threshold
                and cosine(embeddings[current["right"]],
                           embeddings[pair["right"]]) >= threshold]
        _EDGE_CACHE[cache_key] = outer_for
    best: list[dict] = []
    for _ in range(restarts):
        if not inner:
            break
        centre_words = centre.split()
        bigrams = Counter(zip(centre_words, centre_words[1:]))
        templates = Counter({tuple(map(len, centre_words)): 1})
        sentences = Counter({centre: 1})
        content_words = Counter(word for word in centre_words
                                if word not in CONNECTIVES)

        def features(index):
            sentences = (pairs[index]["left"].split(), pairs[index]["right"].split())
            bg = [item for words in sentences for item in zip(words, words[1:])]
            tp = [tuple(map(len, words)) for words in sentences]
            return bg, tp

        def safe(index):
            bg, tp = features(index)
            texts = (pairs[index]["left"], pairs[index]["right"])
            content = [word for text in texts for word in text.split()
                       if word not in CONNECTIVES]
            return (all(a != b for a, b in bg)
                    and all(not sentences[text] for text in texts)
                    and texts[0] != texts[1]
                    and all(bigrams[item] + bg.count(item) <= max_bigram_uses
                            for item in set(bg))
                    and all(templates[item] + tp.count(item) <= max_template_uses
                            for item in set(tp))
                    and all(content_words[word] + content.count(word)
                            <= max_content_word_uses for word in set(content)))

        possible_inner = [index for index in inner if safe(index)]
        if not possible_inner:
            continue
        path = [rng.choice(possible_inner)]
        bg, tp = features(path[0])
        bigrams.update(bg); templates.update(tp)
        sentences.update((pairs[path[0]]["left"], pairs[path[0]]["right"]))
        content_words.update(word for text in (pairs[path[0]]["left"],
                                                pairs[path[0]]["right"])
                             for word in text.split() if word not in CONNECTIVES)
        unused = set(range(len(pairs))) - set(path)
        while len(path) < want:
            eligible = [index for index in unused.intersection(outer_for[path[0]])
                        if safe(index)]
            if not eligible:
                break
            outer = rng.choice(eligible)
            path.insert(0, outer)
            unused.discard(outer)
            bg, tp = features(outer)
            bigrams.update(bg); templates.update(tp)
            sentences.update((pairs[outer]["left"], pairs[outer]["right"]))
            content_words.update(word for text in (pairs[outer]["left"],
                                                    pairs[outer]["right"])
                                 for word in text.split() if word not in CONNECTIVES)
        if len(path) > len(best):
            best = [pairs[index] for index in path]
        if len(best) >= want:
            break
    return best
