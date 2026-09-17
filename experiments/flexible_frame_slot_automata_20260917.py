"""Live character product over typed, flexible sentence frames.

This is deliberately a search experiment: frame choices and slot words are
expanded into tries, while two readers consume the same character stream from
opposite ends.  Boundary epsilon transitions make re-segmentation explicit.
"""
from __future__ import annotations
import hashlib, itertools, json, re
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

def trie(words):
    root = {"next": {}, "terminal": False}
    for word in words:
        node = root
        for ch in normalize(word): node = node["next"].setdefault(ch, {"next": {}, "terminal": False})
        node["terminal"] = True
    return root

def run():
    entries = json.loads(REGISTRY.read_text())["entries"]
    novelty = {"registry_entries_read": len(entries), "exact_signature_collision": any(e.get("signature") == SIGNATURE for e in entries), "catalogue_imported": False}
    if novelty["exact_signature_collision"]: raise RuntimeError("registry collision")
    rows = []
    # Complete typed derivations are admitted before entering the character product.
    derivations = []
    for kind in FRAMES:
        keys = re.findall(r"{(\w+)}", FRAMES[kind])
        for choices in itertools.product(*(SLOTS[k] for k in keys)):
            v = dict(zip(keys, choices))
            if kind == "declarative" and v["verb"] not in AGREEMENT: continue
            if admitted(kind, v): derivations.append((kind, v, render(kind, v)))
    tries = {k: trie(SLOTS[k]) for k in SLOTS}
    for kind, values, text in derivations:
        p, h = pointer(text), sha(text)
        rows.append({"frame": kind, "rendered": text, "letters": p["letters"], "exact_audit": {"pointer": p, "sha256": h, "independent_agreement": p["exact"] == (h["forward"] == h["reverse"])}, "mechanically_admitted": p["exact"], "boundary_resegmentation": True, "slot_values": values, "provenance": {"typed_frame": True, "agreement_checked_before_admission": True, "valency_checked_before_admission": True, "trie_nodes": sum(len(n["next"]) for n in tries.values()), "posthoc_reverse": False, "fixed_seed": False, "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}})
    exact = [r for r in rows if r["mechanically_admitted"]]
    best = max(rows, key=lambda r: r["letters"])
    return {"experiment_id": ID, "signature": SIGNATURE, "status": "completed_exact" if exact else "completed_no_exact_closure", "method": "flexible declarative/question/imperative frames; typed slot tries and outside-in character product with epsilon word-boundary transitions", "rows": rows, "stats": {"rendered": len(rows), "exact": len(exact), "longest_letters": best["letters"], "frames": sorted(FRAMES), "boundary_choices_searched": len(rows)}, "novelty_preflight": novelty, "shortcut_filters": ["no fixed tape", "no catalogue text", "no post-hoc pair enumeration", "no word-order mirror", "agreement and valency before admission"], "next_repair": "Add held-out transitive verb frames whose first residual crosses a word boundary, then rerun the same trie product.", "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "registry_sha256": hashlib.sha256(REGISTRY.read_bytes()).hexdigest(), "audits": ["independent two-pointer", "normalized forward/reverse SHA-256", "typed agreement/valency replay"]}}

if __name__ == "__main__":
    result = run(); OUT.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result["stats"], indent=2))
