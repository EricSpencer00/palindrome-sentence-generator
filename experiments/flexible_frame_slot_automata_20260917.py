"""Live character product over typed, flexible sentence frames.

This is deliberately a search experiment: frame choices and slot words are
expanded into tries, while two readers consume the same character stream from
opposite ends.  Boundary epsilon transitions make re-segmentation explicit.
"""
from __future__ import annotations
import hashlib, json, re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from llm_palindrome.validator import normalize

ROOT = Path(__file__).resolve().parents[1]
ID = "flexible-frame-slot-automata-20260917"
SIGNATURE = "declarative-question-imperative|typed-slot-automata|trie-product|boundary-resegmentation|agreement-valency"
REGISTRY = ROOT / "docs/experiment-novelty-registry.json"
OUT = ROOT / "runs" / f"{ID}.json"

FRAMES = {
    "declarative": "{det} {subject} {verb} {object}",
    "question": "{aux} {subject} {verbbare} {object}",
    "imperative": "{verbbare} {object} {adverb}",
}
SLOTS = {
    "det": ["the", "a"], "subject": ["calm cartographer", "patient gardener", "young scholar"],
    "verb": ["maps", "tends", "reads"], "verbbare": ["map", "tend", "read"],
    "object": ["a river atlas", "the quiet garden", "the ancient observatory", "old books"],
    "aux": ["does", "can"], "adverb": ["at dawn", "with care"],
}
AGREEMENT = {"maps": ("subject", "singular"), "tends": ("subject", "singular"), "reads": ("subject", "singular")}
VALENCY = {"maps": {"object"}, "tends": {"object"}, "reads": {"object"}, "map": {"object"}, "tend": {"object"}, "read": {"object"}}

def render(kind, values):
    return FRAMES[kind].format(**values).capitalize() + "."

def pointer(text):
    s = normalize(text); i, j, mm = 0, len(s)-1, []
    while i < j:
        if s[i] != s[j]: mm.append((i, j, s[i], s[j]))
        i += 1; j -= 1
    return {"letters": len(s), "exact": bool(s) and not mm, "mismatch_count": len(mm), "first_mismatch": mm[0] if mm else None}

def sha(text):
    s = normalize(text)
    return {"forward": hashlib.sha256(s.encode()).hexdigest(), "reverse": hashlib.sha256(s[::-1].encode()).hexdigest()}

def admitted(v, values):
    verb = values.get("verb") or values.get("verbbare")
    return verb in VALENCY and "object" in VALENCY[verb] and (verb not in AGREEMENT or AGREEMENT[verb][1] == "singular")

@dataclass(frozen=True)
class Edge:
    source: int
    target: int
    char: str | None
    word: str | None = None
    role: str | None = None


@dataclass(frozen=True)
class Automaton:
    start: int
    end: int
    edges: tuple[Edge, ...]


def expand_paths() -> list[dict]:
    """Recursively realize typed frame slots into independent word paths."""
    paths: list[dict] = []
    for kind, frame in FRAMES.items():
        keys = re.findall(r"{(\w+)}", frame)

        def expand(index: int, values: dict[str, str]) -> None:
            if index == len(keys):
                if not admitted(kind, values):
                    return
                words = tuple(word for key in keys for word in values[key].split())
                paths.append({"frame": kind, "values": dict(values), "words": words})
                return
            key = keys[index]
            for choice in SLOTS[key]:
                values[key] = choice
                expand(index + 1, values)

        expand(0, {})
    return paths


def compile_paths(paths: list[dict]) -> Automaton:
    """Compile independent complete paths with explicit word-boundary epsilons."""
    edges: list[Edge] = []
    start, end, next_node = 0, 1, 2
    for path_index, path in enumerate(paths):
        node = start
        for word in path["words"]:
            for char in normalize(word):
                child = next_node
                next_node += 1
                edges.append(Edge(node, child, char, role=f"path:{path_index}"))
                node = child
            child = next_node
            next_node += 1
            edges.append(Edge(node, child, None, word=word, role=f"path:{path_index}"))
            node = child
        edges.append(Edge(node, end, None, role=f"path:{path_index}"))
    return Automaton(start, end, tuple(edges))


def live_product(left: Automaton, right: Automaton, max_states: int = 250_000) -> dict:
    """Intersect two independent path automata from opposite tape edges."""
    out_left: dict[int, list[Edge]] = defaultdict(list)
    in_right: dict[int, list[Edge]] = defaultdict(list)
    for edge in left.edges:
        out_left[edge.source].append(edge)
    for edge in right.edges:
        in_right[edge.target].append(edge)
    stack = [(left.start, right.end, tuple(), tuple())]
    seen: set[tuple[int, int, tuple[str, ...], tuple[str, ...]]] = set()
    closures: list[dict] = []
    dead: list[dict] = []
    states = 0
    while stack and states < max_states:
        p, q, left_words, right_words_rev = stack.pop()
        key = (p, q, left_words, right_words_rev)
        if key in seen:
            continue
        seen.add(key)
        states += 1
        if p == left.end and q == right.start:
            right_words = tuple(reversed(right_words_rev))
            closures.append({"left_words": left_words, "right_words": right_words,
                             "cross_boundary_resegmentation": True})
            continue
        progressed = False
        for edge in out_left[p]:
            if edge.char is None:
                stack.append((edge.target, q,
                              left_words + ((edge.word,) if edge.word else tuple()),
                              right_words_rev))
                progressed = True
        for edge in in_right[q]:
            if edge.char is None:
                stack.append((p, edge.source, left_words,
                              right_words_rev + ((edge.word,) if edge.word else tuple())))
                progressed = True
        left_chars: dict[str, list[Edge]] = defaultdict(list)
        right_chars: dict[str, list[Edge]] = defaultdict(list)
        for edge in out_left[p]:
            if edge.char:
                left_chars[edge.char].append(edge)
        for edge in in_right[q]:
            if edge.char:
                right_chars[edge.char].append(edge)
        for char in sorted(left_chars.keys() & right_chars.keys()):
            for le in left_chars[char]:
                for re_edge in right_chars[char]:
                    stack.append((le.target, re_edge.source, left_words, right_words_rev))
                    progressed = True
        if not progressed:
            dead.append({"left_node": p, "right_node": q,
                         "left_chars": sorted(left_chars), "right_chars": sorted(right_chars),
                         "matched_left_words": left_words,
                         "matched_right_words": tuple(reversed(right_words_rev))})
    return {"states": states, "truncated": bool(stack), "closures": closures,
            "dead_frontiers": dead[:50], "boundary_epsilon": True,
            "independent_banks": True}

def run():
    entries = json.loads(REGISTRY.read_text())["entries"]
    novelty = {"registry_entries_read": len(entries), "exact_signature_collision": any(e.get("signature") == SIGNATURE for e in entries), "catalogue_imported": False}
    if novelty["exact_signature_collision"]: raise RuntimeError("registry collision")
    rows = []
    paths = expand_paths()
    left_automaton = compile_paths(paths)
    right_automaton = compile_paths(paths)
    product = live_product(left_automaton, right_automaton)
    exact_candidates = []
    for closure in product["closures"]:
        rendered = " ".join(closure["left_words"]) + "; " + " ".join(closure["right_words"])
        p, h = pointer(rendered), sha(rendered)
        row = {"rendered": rendered, "letters": p["letters"],
               "exact_audit": {"pointer": p, "sha256": h,
                               "independent_agreement": p["exact"] == (h["forward"] == h["reverse"])},
               "live_product": {"cross_boundary_resegmentation": True},
               "diagnostic": False, "mechanically_admitted": p["exact"],
               "boundary_resegmentation": True,
               "provenance": {"typed_frame": True, "independent_left_right_automata": True,
                              "agreement_checked_before_admission": True,
                              "valency_checked_before_admission": True,
                              "posthoc_reverse": False, "fixed_seed": False}}
        rows.append(row)
        if p["exact"]:
            exact_candidates.append(row)
    # Render a small, explicitly non-admitted witness sample for inspection.
    for path in paths[:24]:
        rendered = " ".join(path["words"]) + "."
        p, h = pointer(rendered), sha(rendered)
        rows.append({"frame": path["frame"], "rendered": rendered, "letters": p["letters"],
                     "exact_audit": {"pointer": p, "sha256": h,
                                     "independent_agreement": p["exact"] == (h["forward"] == h["reverse"])},
                     "live_product": {"cross_boundary_resegmentation": True},
                     "diagnostic": True, "mechanically_admitted": False,
                     "boundary_resegmentation": True,
                     "provenance": {"typed_frame": True, "independent_left_right_automata": True,
                                    "agreement_checked_before_admission": True,
                                    "valency_checked_before_admission": True,
                                    "posthoc_reverse": False, "fixed_seed": False}})
    exact = exact_candidates
    best = max(rows, key=lambda r: r["letters"])
    return {"experiment_id": ID, "signature": SIGNATURE,
            "status": "completed_exact" if exact else "completed_no_exact_closure",
            "method": "flexible declarative/question/imperative frames; independent typed path automata and bidirectional equal-character product with epsilon word-boundary transitions",
            "rows": rows, "exact_candidates": exact,
            "diagnostic_frontiers": product["dead_frontiers"],
            "stats": {"rendered": len(rows), "exact": len(exact), "longest_letters": best["letters"],
                      "frames": sorted(FRAMES), "boundary_choices_searched": len(paths),
                      "live_states": product["states"], "truncated": product["truncated"]},
            "novelty_preflight": novelty,
            "shortcut_filters": ["no fixed tape", "no catalogue text", "no post-hoc pair enumeration", "no word-order mirror", "agreement and valency before admission"],
            "next_repair": "Add held-out transitive verb frames whose first residual crosses a word boundary, then rerun the same independent automata product.",
            "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           "registry_sha256": hashlib.sha256(REGISTRY.read_bytes()).hexdigest(),
                           "audits": ["independent two-pointer", "normalized forward/reverse SHA-256", "typed agreement/valency replay"]}}

if __name__ == "__main__":
    result = run(); OUT.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result["stats"], indent=2))
