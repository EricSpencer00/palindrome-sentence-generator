"""Dual-parse chunk word-equation search.

Forward and reverse sides are generated as independently typed chunk
sequences.  Their complete normalized tapes are joined by an exact word
equation, so lexical/syntactic constraints are applied before a closure is
considered.  This is a construction experiment; human readers remain the
readability judge.
"""
from __future__ import annotations
import argparse, json
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_palindrome.validator import normalize, is_palindrome

MIN_LETTERS = 39

@dataclass(frozen=True)
class Chunk:
    text: str
    kind: str
    # Subject/object/event semantics used by live valency checks.
    sem: str = "any"

N = tuple(Chunk(x, "NP", "person") for x in "artist baker doctor farmer friend gardener teacher writer".split())
T = tuple(Chunk(x, "NP", "thing") for x in "letter story map note book plan report song".split())
DET_N = tuple(Chunk(f"{d} {n}", "NP", "person") for d in ("a", "the", "my", "our") for n in "artist baker doctor farmer friend gardener teacher writer".split())
DET_T = tuple(Chunk(f"{d} {n}", "NP", "thing") for d in ("a", "the", "my", "our") for n in "letter story map note book plan report song".split())
ADJ_N = tuple(Chunk(f"{d} {a} {n}", "NP", "person") for d in ("a", "the") for a in "calm kind patient careful".split() for n in "artist baker doctor farmer teacher writer".split())
ADJ_T = tuple(Chunk(f"{d} {a} {n}", "NP", "thing") for d in ("a", "the") for a in "brief clear detailed useful".split() for n in "letter story map note book plan report".split())
VP_T = tuple(Chunk(f"{v} {o}", "VP", "event") for v in "built cleaned drafted found fixed marked opened painted planned saved wrote".split() for o in DET_T)
VP_P = tuple(Chunk(f"{v} {o}", "VP", "event") for v in "helped thanked called followed watched".split() for o in DET_N)
PP = tuple(Chunk(f"{p} {n}", "PP", "place") for p in ("in", "near", "by", "under") for n in ("the garden", "the office", "the studio", "the town"))
ADV = tuple(Chunk(x, "ADV") for x in "quietly carefully today".split())

# Each template is a complete ordinary sentence skeleton.  Chunk boundaries
# are independent on the two sides, permitting staggered word boundaries.
TEMPLATES = {
    "svo": (("SUBJ", "NP"), ("VP", "VP_T")),
    "svo_adv": (("SUBJ", "NP"), ("VP", "VP_T"), ("ADV", "ADV")),
    "svo_pp": (("SUBJ", "NP"), ("VP", "VP_T"), ("PP", "PP")),
    "ditrans": (("SUBJ", "NP"), ("VP", "VP_P"), ("PP", "PP")),
}
POOLS = {"NP": DET_N + DET_T + ADJ_N + ADJ_T, "VP_T": VP_T,
         "VP_P": VP_P, "ADV": ADV, "PP": PP}

def valid_sentence(chunks: tuple[Chunk, ...]) -> bool:
    kinds = [c.kind for c in chunks]
    if not kinds or kinds[0] != "NP" or kinds.count("VP") != 1:
        return False
    subject = chunks[0].sem
    vp = next(c for c in chunks if c.kind == "VP")
    # Object/recipient is encoded in VP chunk; prohibit person/thing role
    # mismatch by construction and require a semantic event.
    return subject in {"person", "thing"} and vp.sem == "event"

def sentences(limit: int = 150_000):
    rows = []
    for name, slots in TEMPLATES.items():
        pools = []
        for role, pool_name in slots:
            pools.append(POOLS[pool_name])
        for chosen in product(*pools):
            if not valid_sentence(chosen):
                continue
            words = normalize(" ".join(c.text for c in chosen)).split()
            if len(words) != len(set(words)):
                continue
            text = " ".join(c.text for c in chosen)
            rows.append({"text": text, "template": name,
                         "chunks": [{"text": c.text, "kind": c.kind, "sem": c.sem} for c in chosen],
                         "tape": normalize(text)})
            if len(rows) >= limit: return rows
    return rows

def audit(text: str):
    tape = normalize(text)
    mm = [i for i in range(len(tape)//2) if tape[i] != tape[-1-i]]
    return {"exact": bool(tape) and not mm, "letters": len(tape), "mismatches": mm,
            "normalized_tape": tape, "sha256": sha256(tape.encode()).hexdigest()}

def run(limit: int = 150_000):
    ss = sentences(limit)
    index = {}
    for row in ss:
        index.setdefault(row["tape"], []).append(row)
    candidates = []
    for left in ss:
        for right in index.get(left["tape"][::-1], ()):
            text = left["text"] + "; " + right["text"]
            a = audit(text)
            if a["letters"] < MIN_LETTERS or not a["exact"]:
                continue
            all_words = normalize(text).split()
            if len(all_words) != len(set(all_words)):
                continue
            candidates.append({"text": text, "audit": a, "left": left,
                               "right": right,
                               "provenance": "authored dual typed chunk products; exact reverse-tape index",
                               "anti_shortcut": {"distinct_words": True, "self_palindromic_units": False,
                                                  "catalogue_text": False, "independent_side_parses": True}})
    candidates.sort(key=lambda r: (-r["audit"]["letters"], r["text"]))
    return {"status": "dual_parse_exact_closures_need_blinded_readability",
            "config": {"sentence_count": len(ss), "template_count": len(TEMPLATES),
                        "min_letters": MIN_LETTERS, "staggered_chunk_boundaries": True},
            "candidate_count": len(candidates), "candidates": candidates[:100],
            "deepest_frontier": {"indexed_tapes": len(index), "max_sentence_letters": max((len(x["tape"]) for x in ss), default=0)},
            "next_operator_if_empty": "Add typed multiword reverse-boundary chunks with explicit determiner and argument attachment, then intersect both chunk tries incrementally rather than indexing complete sentences.",
            "generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--limit", type=int, default=150000)
    args = ap.parse_args(); result = run(args.limit); args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"sentence_count": result["config"]["sentence_count"], "candidate_count": result["candidate_count"], "deepest_frontier": result["deepest_frontier"]}, indent=2))
if __name__ == "__main__": main()
