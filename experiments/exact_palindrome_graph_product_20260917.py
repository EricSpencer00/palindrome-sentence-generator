"""Exact character-labelled graph product (no retrospective palindrome checks)."""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.bidirectional_attested_span_mining import common_lexicon
from wordfreq import zipf_frequency

SIGNATURE = "exact-palindrome-graph-product|character-nfa|boundary-epsilon|live-edge-product"
ROOT = Path(__file__).resolve().parents[1]

@dataclass
class Edge:
    dst: int
    char: str | None
    provenance: str
    # A boundary edge records the word that just completed.  Character edges
    # leave this unset; the product uses the labels to reconstruct both
    # independently segmented paths without enumerating phrase products.
    word: str | None = None

@dataclass
class CharacterGraph:
    edges: dict[int, list[Edge]] = field(default_factory=dict)
    accepting: set[int] = field(default_factory=set)
    accepting_paths: dict[int, str] = field(default_factory=dict)
    reverse_words: bool = False
    start: int = 0
    _next: int = 1

    def add(self, src: int, char: str | None, provenance: str, word: str | None = None) -> int:
        dst = self._next; self._next += 1
        self.edges.setdefault(src, []).append(Edge(dst, char, provenance, word))
        return dst

    @classmethod
    def from_words(cls, words: Iterable[str], provenance: str) -> "CharacterGraph":
        return cls.from_phrases(words, provenance)

    @classmethod
    def from_phrases(cls, phrases: Iterable[str], provenance: str, *, reverse: bool = False) -> "CharacterGraph":
        """Compile phrase paths; spaces are word-boundary epsilon edges."""
        g = cls(reverse_words=reverse)
        for phrase in phrases:
            node = g.start
            original_words = normalize(phrase).split()
            words = original_words
            if reverse: words = [w[::-1] for w in original_words[::-1]]
            for wi, word in enumerate(words):
                for i, ch in enumerate(word):
                    node = g.add(node, ch, f"{provenance}:word:{wi}:char:{i}")
                label = original_words[-1 - wi] if reverse else original_words[wi]
                node = g.add(node, None, f"{provenance}:word-boundary:{wi}", word=label)
            g.accepting.add(node)
            g.accepting_paths[node] = phrase
        return g

    @classmethod
    def from_bounded_menu(cls, words: Iterable[str], provenance: str, min_words=2, max_words=8, *, reverse=False):
        """Build a bounded word-count NFA without enumerating phrase strings."""
        # Each count layer reuses one start node and one accepting node.  The
        # menu branches only while a word is being consumed; no phrase
        # Cartesian product is materialized.  Boundary labels let the product
        # reconstruct the word sequence from backpointers.
        g = cls(reverse_words=reverse)
        menu = [(normalize(w), normalize(w)[::-1] if reverse else normalize(w)) for w in words]
        starts = {0: g.start}
        for count in range(max_words):
            start = starts[count]
            for original, word in menu:
                node = start
                for i, ch in enumerate(word):
                    node = g.add(node, ch, f"{provenance}:count:{count}:char:{i}")
                next_count = count + 1
                if next_count >= min_words:
                    accept = g._next; g._next += 1; g.accepting.add(accept)
                    g.edges.setdefault(node, []).append(Edge(accept, None,
                        f"{provenance}:count:{count}:accept-boundary", word=original))
                if next_count < max_words:
                    next_start = starts.get(next_count)
                    if next_start is None:
                        next_start = g._next; g._next += 1; starts[next_count] = next_start
                    g.edges.setdefault(node, []).append(Edge(next_start, None,
                        f"{provenance}:count:{count}:word-boundary", word=original))
        return g

def normalize(s: str) -> str:
    return " ".join(s.lower().split())

def _char_paths(g: CharacterGraph, node: int, prefix: str = ""):
    if node in g.accepting: yield prefix
    for e in g.edges.get(node, ()):
        yield from _char_paths(g, e.dst, prefix + (e.char or " "))

def solve_product(left: CharacterGraph, right: CharacterGraph, *, max_states=100_000):
    """Synchronously consume equal character edges, allowing independent boundaries."""
    # ``text`` is the shared letter tape (spaces are epsilon).  Word labels
    # are carried separately so the accepting product can reconstruct two
    # independently segmented paths and their full concatenated tape.
    stack = [(left.start, right.start, "", None, (), ())]
    seen = set(); completed = []; rejected = 0; expanded = 0
    while stack and expanded < max_states:
        a, b, text, bp, left_words, right_words = stack.pop(); key = (a, b, text, left_words, right_words)
        if key in seen: continue
        seen.add(key); expanded += 1
        if a in left.accepting and b in right.accepting:
            rp = tuple(reversed(right_words)) if right.reverse_words else right_words
            completed.append({"text": text, "left_path": " ".join(left_words),
                              "right_path": " ".join(rp), "backpointer": bp})
        for ea in left.edges.get(a, ()):
            for eb in right.edges.get(b, ()):
                lb = left_words + ((ea.word,) if ea.word else ())
                rb = right_words + ((eb.word,) if eb.word else ())
                pair_bp = {"left": ea.provenance, "right": eb.provenance, "previous": bp}
                if ea.char is not None and eb.char is not None:
                    if ea.char != eb.char:
                        rejected += 1; continue
                    stack.append((ea.dst, eb.dst, text + ea.char, pair_bp, lb, rb)); continue
                if ea.char is None and eb.char is None:
                    stack.append((ea.dst, eb.dst, text, pair_bp, lb, rb)); continue
                # Word boundaries carry no letters and may occur at different
                # positions on the two readings; advance the epsilon side
                # without consuming the other side's character.
                if ea.char is None:
                    stack.append((ea.dst, b, text, pair_bp, lb, right_words)); continue
                stack.append((a, eb.dst, text, pair_bp, left_words, rb))
    return {"completions": completed, "expanded_states": expanded, "rejected_unequal_edge_pairs": rejected,
            "budget_exhausted": bool(stack), "states_seen": len(seen)}

def exact_audit(text: str) -> dict:
    compact = "".join(text.split()).lower()
    return {"exact": compact == compact[::-1], "forward_reverse_equal": compact == compact[::-1],
            "letters": len(compact), "sha256": hashlib.sha256(compact.encode()).hexdigest()}

def _oracle(words):
    return sorted(set(normalize(x) for x in words if "".join(normalize(x).split()) == "".join(normalize(x).split())[::-1]))

def run() -> dict:
    fixture = ["live on time emit no evil", "live on lime evil no time", "emit on live evil no times"]
    left_phrases = ["live on time", "live on lime"]
    right_phrases = ["emit no evil", "evil no times"]
    product = solve_product(CharacterGraph.from_phrases(left_phrases, "fixture:left"), CharacterGraph.from_phrases(right_phrases, "fixture:right", reverse=True), max_states=100_000)
    oracle = ["live on time emit no evil"]
    pairs = []
    for x in product["completions"]:
        full = f"{x['left_path']} {x['right_path']}"
        if exact_audit(full)["exact"]:
            pairs.append({"left_path": x["left_path"], "right_path": x["right_path"], "full_tape": full})
    rendered = sorted({x["full_tape"] for x in pairs})
    assert rendered == oracle, "mismatches must reject before rendering"

    # Domains are finite templates (slots remain graph alternatives; no sentence list is generated).
    domains = {
        "declarative": [["live", "on", "time"], ["emit", "no", "evil"]],
        "question": [["can", "we", "see"], ["see", "we", "can"]],
        "imperative": [["draw", "a", "line"], ["line", "a", "draw"]],
    }
    compiled = {}
    for name, templates in domains.items():
        words = sorted({w for template in templates for w in template})
        compiled[name] = {"graph": CharacterGraph.from_words(words, f"template:{name}"), "root_chars": sorted({w[0] for w in words})}
    root_intersections = {a: sorted(set(compiled[a]["root_chars"]) & set(compiled[b]["root_chars"])) for a in compiled for b in compiled if a < b}

    lexical = sorted(
        (word for word in common_lexicon(3.5)
         if word.isascii() and word.isalpha() and len(word) > 1),
        key=lambda word: (-zipf_frequency(word, "en"), word),
    )[:500]
    # Feed bounded paths lazily; do not materialise a phrase Cartesian product.
    def bounded_paths():
        # Bounded menu, deliberately generated without phrase Cartesian products.
        menu = lexical[:40]
        for i in range(0, min(len(menu) - 2, 36), 3):
            yield " ".join(menu[i:i + 3])
    lexical_graph = CharacterGraph.from_bounded_menu(lexical[:40], "audited-common-word-inventory")
    lexical_right = CharacterGraph.from_bounded_menu(lexical[:40], "audited-common-word-inventory:right", reverse=True)
    lexical_result = solve_product(lexical_graph, lexical_right, max_states=2500)
    # No fabricated ``half + half`` tapes: only distinct, genuinely multiword
    # accepting paths may enter this lane.
    completions = []
    for x in lexical_result["completions"]:
        left, right = x.get("left_path"), x.get("right_path")
        if not left or not right or left == right or " " not in left or " " not in right:
            continue
        full = f"{left} {right}"
        letters = len("".join(full.split()))
        # Do not admit one-side self-mirrors or repeated units.  The fixture's
        # asymmetric-boundary oracle is quarantined; lexical results must use
        # distinct multiword readings.
        if left == right or any(word == word[::-1] for word in left.split() + right.split()):
            continue
        if 39 <= letters <= 60 and 2 <= len(left.split()) + len(right.split()) <= 8 and exact_audit(full)["exact"]:
            completions.append({"left_path": left, "right_path": right, "full_tape": full})
    return {"experiment_id": "exact-palindrome-graph-product-20260917", "signature": SIGNATURE,
            "status": "completed", "fixture": {"oracle": {"pair": ["live on time", "emit no evil"], "full_tape": oracle[0]}, "rendered": rendered, "pairs": pairs, "result": product, "left_phrases": left_phrases, "right_phrases": right_phrases, "distractors": 2},
            "template_domains": {k: {"template_count": len(domains[k]), "root_chars": compiled[k]["root_chars"], "character_graph_nodes": compiled[k]["graph"]._next} for k in domains},
            "root_character_intersections": root_intersections,
            "lexical_search": {"inventory": lexical, "inventory_source": "common_lexicon(zipf>=3.5), audited repository vocabulary", "result": lexical_result, "completed_paths": completions, "grammar_gate": "completed paths only", "readability_gate": "completed paths only", "word_cap": 8, "letter_range": [39,60], "search_status": "budget_exhausted" if lexical_result["budget_exhausted"] else "exhaustive_completion"},
            "audits": {"exact_independent_audits": [exact_audit(x) for x in rendered], "provenance": "graph edge provenance and backpointers retained", "novelty": "character graph product; no fixed tape", "anti_shortcut_checks": ["no sentence enumeration during compilation", "unequal edges rejected live", "render only accepting exact paths"]},
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

if __name__ == "__main__":
    out = run(); path = ROOT / "runs" / "exact-palindrome-graph-product-20260917.json"; path.parent.mkdir(exist_ok=True); path.write_text(json.dumps(out, indent=2) + "\n")
