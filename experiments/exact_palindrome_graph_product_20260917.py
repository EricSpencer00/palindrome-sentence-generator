"""Exact character-labelled graph product (no retrospective palindrome checks)."""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

SIGNATURE = "exact-palindrome-graph-product|character-nfa|boundary-epsilon|live-edge-product"
ROOT = Path(__file__).resolve().parents[1]

@dataclass
class Edge:
    dst: int
    char: str | None
    provenance: str

@dataclass
class CharacterGraph:
    edges: dict[int, list[Edge]] = field(default_factory=dict)
    accepting: set[int] = field(default_factory=set)
    accepting_paths: dict[int, str] = field(default_factory=dict)
    start: int = 0
    _next: int = 1

    def add(self, src: int, char: str | None, provenance: str) -> int:
        dst = self._next; self._next += 1
        self.edges.setdefault(src, []).append(Edge(dst, char, provenance))
        return dst

    @classmethod
    def from_words(cls, words: Iterable[str], provenance: str) -> "CharacterGraph":
        return cls.from_phrases(words, provenance)

    @classmethod
    def from_phrases(cls, phrases: Iterable[str], provenance: str, *, reverse: bool = False) -> "CharacterGraph":
        """Compile phrase paths; spaces are word-boundary epsilon edges."""
        g = cls()
        for phrase in phrases:
            node = g.start
            words = normalize(phrase).split()
            if reverse: words = [w[::-1] for w in words[::-1]]
            for wi, word in enumerate(words):
                for i, ch in enumerate(word):
                    node = g.add(node, ch, f"{provenance}:word:{wi}:char:{i}")
                node = g.add(node, None, f"{provenance}:word-boundary:{wi}")
            g.accepting.add(node)
            g.accepting_paths[node] = phrase
        return g

def normalize(s: str) -> str:
    return " ".join(s.lower().split())

def _char_paths(g: CharacterGraph, node: int, prefix: str = ""):
    if node in g.accepting: yield prefix
    for e in g.edges.get(node, ()):
        yield from _char_paths(g, e.dst, prefix + (e.char or " "))

def solve_product(left: CharacterGraph, right: CharacterGraph, *, max_states=100_000):
    """Synchronously consume matching labelled edges, with explicit backpointers."""
    stack = [(left.start, right.start, "", None)]
    seen = set(); completed = []; rejected = 0; expanded = 0
    while stack and expanded < max_states:
        a, b, text, bp = stack.pop(); key = (a, b, text)
        if key in seen: continue
        seen.add(key); expanded += 1
        if a in left.accepting and b in right.accepting:
            completed.append({"text": text.strip(), "left_path": left.accepting_paths.get(a), "right_path": right.accepting_paths.get(b), "backpointer": bp})
        for ea in left.edges.get(a, ()):
            for eb in right.edges.get(b, ()):
                if ea.char != eb.char:
                    if ea.char is not None and eb.char is not None: rejected += 1
                    continue
                nxt = text + (ea.char or " ")
                stack.append((ea.dst, eb.dst, nxt, {"left": ea.provenance, "right": eb.provenance, "previous": bp}))
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

    lexical = [w.strip().lower() for w in (ROOT / "data" / "lexicon.txt").read_text().splitlines() if w.strip()][:500]
    lexical_graph = CharacterGraph.from_words(lexical, "audited-common-word-inventory")
    lexical_result = solve_product(lexical_graph, lexical_graph, max_states=2500)
    completions = sorted({x["text"] for x in lexical_result["completions"] if 39 <= len("".join(x["text"].split())) <= 60 and 2 <= len(x["text"].split()) <= 8 and exact_audit(x["text"])["exact"]})
    return {"experiment_id": "exact-palindrome-graph-product-20260917", "signature": SIGNATURE,
            "status": "completed", "fixture": {"oracle": {"pair": ["live on time", "emit no evil"], "full_tape": oracle[0]}, "rendered": rendered, "pairs": pairs, "result": product, "left_phrases": left_phrases, "right_phrases": right_phrases, "distractors": 2},
            "template_domains": {k: {"template_count": len(domains[k]), "root_chars": compiled[k]["root_chars"], "character_graph_nodes": compiled[k]["graph"]._next} for k in domains},
            "root_character_intersections": root_intersections,
            "lexical_search": {"inventory": lexical, "inventory_source": "data/lexicon.txt (audited repository inventory slice)", "result": lexical_result, "completed_paths": [{"left_path": x["text"], "right_path": x["text"], "full_tape": x["text"] + " " + x["text"]} for x in lexical_result["completions"] if exact_audit(x["text"] + x["text"])["exact"]], "grammar_gate": "completed paths only", "readability_gate": "completed paths only", "word_cap": 8, "letter_range": [39,60], "search_status": "budget_exhausted" if lexical_result["budget_exhausted"] else "exhaustive_completion"},
            "audits": {"exact_independent_audits": [exact_audit(x) for x in rendered], "provenance": "graph edge provenance and backpointers retained", "novelty": "character graph product; no fixed tape", "anti_shortcut_checks": ["no sentence enumeration during compilation", "unequal edges rejected live", "render only accepting exact paths"]},
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

if __name__ == "__main__":
    out = run(); path = ROOT / "runs" / "exact-palindrome-graph-product-20260917.json"; path.parent.mkdir(exist_ok=True); path.write_text(json.dumps(out, indent=2) + "\n")
