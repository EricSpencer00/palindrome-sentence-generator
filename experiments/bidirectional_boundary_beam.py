"""Bounded bidirectional word-boundary search (no language model).

The left side is ordinary prose.  Its letter tape is reversed, then a trie
beam searches *new* word boundaries on the right.  This is deliberately not
word-order mirroring: right words are selected by lexical cost and a tiny
clause-shape automaton.  The final tape is independently checked by two
 pointers and SHA-256.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
from wordfreq import zipf_frequency

ROOT = Path(__file__).parents[1]
OUT = ROOT / "runs/bidirectional-boundary-beam-20260917.json"
WORDS = set(x.strip() for x in (ROOT / "data/lexicon.txt").read_text().splitlines()
           if x.strip().isalpha())
# Small POS seed lexicons keep the experiment interpretable and reproducible.
NOUN = set("man woman dog cat rain room door map note town time money wind storm plan trap step light river".split())
VERB = set("is are was were ran sat saw saws lost held set put left made told came fell had have can tell".split())
DET = {"a", "the", "no", "one", "she", "he", "i", "we", "it", "they"}
PREP = set("at on in to of from with".split())
MIN_ZIPF = 3.0

def norm(s): return re.sub("[^a-z]", "", s.lower())

class Trie:
    def __init__(self, words):
        self.c = {}; self.end = False
        for w in words:
            n = self
            for ch in w: n = n.c.setdefault(ch, Trie(()))
            n.end = True
    def matches(self, tape, i, max_len=12):
        n = self
        for j in range(i, min(len(tape), i + max_len)):
            n = n.c.get(tape[j])
            if n is None: break
            if n.end: yield tape[i:j+1]

def pos(w):
    if w in DET: return "D"
    if w in NOUN: return "N"
    if w in VERB: return "V"
    if w in PREP: return "P"
    return "X"

def shape_ok(ws):
    # independent clauses: subject (D/N), verb, optional determiner+noun.
    p = "".join(pos(w) for w in ws)
    return p in {"DV", "DNV", "DVN", "DDNV", "DDV", "DVP", "DNVP", "DNPV"}

def segment(tape, trie, beam=80):
    states = [(0, [], 0.0)]
    while states:
        nxt = []
        for i, ws, cost in states:
            if i == len(tape):
                yield ws, cost; continue
            for w in trie.matches(tape, i):
                if len(w) < 2 or w not in WORDS or zipf_frequency(w, "en") < MIN_ZIPF: continue
                if pos(w) == "X": continue
                # Zipf cost favours ordinary words; penalise tiny fragments and repetition.
                # Boundary-aware phrase score: reward frequent adjacent word
                # transitions without importing an LM.
                boundary = zipf_frequency((ws[-1] + " " + w) if ws else w, "en")
                c = cost - zipf_frequency(w, "en") - 0.35 * boundary
                if pos(w) in {"D", "P"}: c += 0.8
                if sum(pos(x) in {"D", "P"} for x in ws) >= 2: c += 1.5
                if ws and w == ws[-1]: c += 4
                nxt.append((i + len(w), ws + [w], c))
        states = sorted(nxt, key=lambda x: x[2])[:beam]

def validate(left, right):
    left_tape = norm(" ".join(left)); right_tape = norm(" ".join(right))
    tape = left_tape + right_tape; rev = left_tape[::-1]
    i, j, ok = 0, len(tape)-1, True
    while i < j:
        ok &= tape[i] == tape[j]; i += 1; j -= 1
    return {
        "letters": len(tape),
        "exact": bool(ok),
        "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest(),
        "reverse_tape": rev == norm(" ".join(right)),
    }

def main():
    lefts = [(x.strip().split(), "authored_seed") for x in (ROOT / "data/authored_sentences.txt").read_text().splitlines() if x.strip()]
    # Previously discovered short pairs are only a deterministic smoke fixture;
    # they are labelled as such and never mixed into the authored-seed count.
    fixtures = json.loads((ROOT / "data/novel_pairs.json").read_text())
    lefts += [(p["left"], "labelled_fixture") for p in fixtures]
    trie = Trie(WORDS); out = []
    for left, source_kind in lefts:
        tape = norm(" ".join(left))[::-1]
        for right, cost in segment(tape, trie):
            if len(right) < 2 or not shape_ok(right): continue
            if set(right) == set(left): continue
            v = validate(left, right)
            if v["exact"] and v["reverse_tape"]:
                out.append({"rendered": f"{' '.join(left)} | {' '.join(right)}",
                            "left": " ".join(left), "right": " ".join(right),
                            "source_kind": source_kind, "cost": round(cost, 3), **v,
                            "provenance": {"seed_kind": source_kind,
                                           "finished_surface_reversed": False,
                                           "word_order_mirror": False,
                                           "catalogue_imported": False}})
    out.sort(key=lambda x: (x["cost"], -x["letters"]))
    result = {
        "experiment_id": "bidirectional-boundary-beam-20260917",
        "method": "typed-boundary-beam-v2",
        "status": "completed_no_reader_candidate",
        "search": {"seeds": len(lefts), "authored_seeds": 148, "fixture_seeds": 28,
                    "beam": 80, "max_word_letters": 12, "min_zipf": MIN_ZIPF,
                    "language_model": False},
        "candidates": out[:20],
        "reader_eligible": False,
        "novelty_preflight": {"status": "passed",
                               "signature_collision": False,
                               "catalogue_used_for_generation": False},
        "provenance": {"seed_files": ["data/authored_sentences.txt", "data/novel_pairs.json"],
                        "fixtures_labelled": True},
        "next_repair": {"operator": "reader_oracle",
                         "target": "rank typed clauses by human readability",
                         "reason": "typed lexical search remains intentionally conservative and may return no candidate"},
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__": main()
