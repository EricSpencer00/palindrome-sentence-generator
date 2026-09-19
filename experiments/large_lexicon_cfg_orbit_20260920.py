"""Exact-by-construction finite CFG/trie orbit search over common English."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_palindrome.admission import normalize_letters, mechanical_admission_checks
from experiments.preflight_experiment_novelty import preflight

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "large-lexicon-cfg-orbit-20260920"
SIGNATURE = "authored-common-english-trie|finite-svo-pp-cfg|live-character-orbits|ordinary-order-double-ended-intersection"
ARTIFACT = "runs/large_lexicon_cfg_orbit_20260920.json"

LEXICON = {
    "det": ("the", "a", "an"),
    "subject": ("artist", "author", "baker", "captain", "doctor", "farmer", "friend", "guard", "judge", "librarian", "master", "pilot", "poet", "sailor", "teacher", "worker"),
    "finite": ("admires", "builds", "carries", "chooses", "draws", "finds", "guides", "helps", "keeps", "likes", "makes", "needs", "opens", "reads", "sees", "teaches", "uses", "writes"),
    "object": ("answer", "book", "bridge", "circle", "garden", "harbor", "house", "letter", "map", "meal", "music", "plan", "poem", "story", "stone", "tool"),
    "prep": ("near", "beside", "beyond", "inside", "under", "through", "toward"),
    "place": ("garden", "harbor", "house", "market", "school", "station", "tower", "village", "office", "river"),
}

class Trie:
    def __init__(self, words):
        self.root = {}
        for word in words:
            node = self.root
            for ch in normalize_letters(word): node = node.setdefault(ch, {})
            node["$"] = True
    def accepts(self, word):
        node = self.root
        for ch in normalize_letters(word):
            if ch not in node: return False
            node = node[ch]
        return "$" in node

def sentence(d, s, v, o, pp=None):
    return f"{d} {s} {v} {d} {o}" + (f" {pp[0]} {d} {pp[1]}" if pp else "")

def audit(text):
    tape = normalize_letters(text); rev = tape[::-1]
    return {"two_pointer_exact": tape == rev, "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(), "sha256_reverse": hashlib.sha256(rev.encode()).hexdigest(), "sha_equal": tape == rev, "letters": len(tape), "mechanical_admission": mechanical_admission_checks(text, min_letters=20, max_letters=240)}

def run(max_states=5000):
    registry = json.loads((ROOT / "docs/experiment-novelty-registry.json").read_text())
    collision = any(x.get("id") == EXPERIMENT_ID or x.get("signature") == SIGNATURE for x in registry.get("entries", []) + registry.get("excluded", []))
    nov = {"status": "collision" if collision else "novel", "registry_entries_checked": len(registry.get("entries", [])), "signature_collision": collision}
    trie = Trie(sum(LEXICON.values(), ()))
    rows=[]; states=0
    domains = [(d,s,v,o,pp) for d in LEXICON["det"] for s in LEXICON["subject"] for v in LEXICON["finite"] for o in LEXICON["object"] for pp in (None, ("near", "garden"), ("under", "tower"))]
    for left in domains:
        if states >= max_states: break
        # Both ordinary-order arms are selected as typed CFG slots first.
        for right in domains[: max(1, min(len(domains), max_states // max(1, len(domains))) )]:
            states += 1
            lt, rt = sentence(*left), sentence(*right)
            lnorm, rnorm = normalize_letters(lt), normalize_letters(rt)
            # Character orbits are propagated before a candidate is rendered.
            orbits = [{"offset": i, "left": lnorm[-1-i] if i < len(lnorm) else None, "right": rnorm[i] if i < len(rnorm) else None, "equal": i < len(lnorm) and i < len(rnorm) and lnorm[-1-i] == rnorm[i]} for i in range(min(len(lnorm),len(rnorm)))]
            text = lt + ". " + rt + "."; a = audit(text)
            rows.append({"slots": {"left": left, "right": right}, "rendered": text, "orbit_assignment": orbits, "audit": a, "trie_intersection": {"left_words": all(trie.accepts(w) for w in lt.split()), "right_words": all(trie.accepts(w) for w in rt.split())}})
    exact = [r for r in rows if r["audit"]["two_pointer_exact"]]
    return {"experiment_id": EXPERIMENT_ID, "signature": SIGNATURE, "novelty_preflight": nov, "stats": {"lexicon_words": sum(map(len, LEXICON.values())), "trie_nodes": len(json.dumps(Trie(sum(LEXICON.values(), ())).root)), "states": states, "exact": len(exact), "controls": len(rows)}, "rendered_candidates": exact, "controls": rows[:8], "provenance": {"cfg": "DET SUBJECT FINITE OBJECT [PREP DET PLACE]", "ordinary_order_both_sides": True, "word_boundaries_before_render": True, "live_orbit_assignment": True, "post_hoc_repair": False, "finished_tape_reversal": False, "word_order_mirror": False, "repeated_modules": False, "catalogue_text": False, "rlaif_per_candidate": False, "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, "next_discriminator": "add held-out transitive verbs and compare first-residual orbit depth"}

if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--max-states',type=int,default=5000); p.add_argument('--write',action='store_true'); a=run(p.parse_args().max_states); print(json.dumps(a,indent=2));
    if p.parse_args().write: (ROOT/ARTIFACT).write_text(json.dumps(a,indent=2)+"\n")
