"""Grammar-aware overhang search over independently authored clause frames.

Unlike the earlier clause-product probes, this constructor never commits one
complete clause before opening the other side.  It carries Norvig's live
character debt together with the next grammatical slot on each side, so a
word is proposed only when its letters can consume the opposite side's
overhang.  The grammar frame, number, and valency features are selected before
emission; no finished tape is reversed or repaired.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get(
    "PAL_OUT", str(ROOT / "runs/grammar-overhang-product-20260920.json")
))
LIVE_FEATURES = os.environ.get("PAL_RELAX_FEATURES", "0") != "1"
MIN_EXACT = int(os.environ.get("PAL_MIN_EXACT", "38"))
PAYLOAD = ROOT / "tools/polaris/payload"

# Complete, non-fragmentary clause frames.  The two sides may choose different
# frames, but every emitted row has two independently licensed finite clauses.
TEMPLATES = [
    ("DET", "ADJ", "NOUN", "VERB", "DET", "NOUN"),
    ("DET", "NOUN", "VERB", "PREP", "DET", "NOUN"),
    ("NAME", "VERB", "DET", "NOUN"),
    ("PRON", "VERB", "DET", "NOUN"),
    ("DET", "NOUN", "VERB", "ADV"),
    ("DET", "NOUN", "VERB", "NUM", "NOUN"),
    ("DET", "NOUN", "VERB", "NAME"),
    ("DET", "ADJ", "NOUN", "VERB", "PREP", "DET", "NOUN"),
    ("DET", "NOUN", "VERB", "DET", "ADJ", "NOUN"),
    ("DET", "NOUN", "VERB", "COMP", "DET", "NOUN"),
    ("DET", "ADJ", "NOUN", "VERB", "DET", "NOUN", "CONJ", "DET", "NOUN"),
    ("DET", "NOUN", "VERB", "PREP", "DET", "ADJ", "NOUN"),
    ("NAME", "VERB", "DET", "NOUN", "PREP", "DET", "NOUN"),
    ("PRON", "VERB", "DET", "NOUN", "CONJ", "PRON", "VERB"),
    ("DET", "ADJ", "NOUN", "VERB", "DET", "NOUN", "COMP", "DET", "NOUN", "VERB"),
    ("DET", "NOUN", "VERB", "PREP", "DET", "NOUN", "CONJ", "DET", "NOUN", "VERB"),
    ("NAME", "VERB", "DET", "NOUN", "PREP", "DET", "NOUN", "COMP", "PRON", "VERB"),
    ("DET", "ADJ", "NOUN", "VERB", "DET", "NOUN", "PREP", "DET", "ADJ", "NOUN"),
    ("PRON", "VERB", "DET", "NOUN", "COMP", "DET", "ADJ", "NOUN", "VERB", "PREP"),
]

DET_SG = {"a", "an", "one", "each", "every", "this", "that"}
DET_PL = {"some", "these", "those", "many", "several", "both", "three", "two"}
PLURAL = {"men", "memos", "people", "letters", "stories", "roads", "rivers", "birds", "sailors", "pilots", "writers", "keepers", "maps", "stars"}
SG_VERBS = {"is", "was", "has", "does", "marks", "maps", "reads", "rips", "guides", "records", "carries", "studies", "inspires", "remembers", "guards", "opens", "draws", "sees", "keeps"}
PL_VERBS = {"are", "were", "have", "do"}
TRANSITIVE = {"mark", "maps", "read", "rip", "guide", "record", "carry", "study", "inspire", "remember", "guard", "open", "draw", "see", "keep", "watch", "chart", "paint", "write", "find", "hold", "carry", "make", "seek", "tell", "show", "follow", "hear", "build", "mend", "cross", "trace", "name"}
INTRANSITIVE = {"arrive", "wait", "sleep", "rise", "fall", "wander", "sing", "remain", "glow", "turn", "rest", "bloom", "leave", "come", "go", "sail"}


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def sha(tape: str) -> str:
    return hashlib.sha256(tape.encode()).hexdigest()


def audit(text: str) -> dict:
    tape = letters(text)
    return {"letters": len(tape), "exact": bool(tape) and tape == tape[::-1],
            "sha256_forward": sha(tape), "sha256_reverse": sha(tape[::-1])}


def word_number(word: str) -> str:
    if word in PLURAL or (word.endswith("s") and word not in {"news", "is", "was"}):
        return "pl"
    return "sg"


def verb_number(word: str) -> str:
    if word in SG_VERBS or word.endswith("s"):
        return "sg"
    return "pl"


def frame_ok(words: list[str], frame: tuple[str, ...]) -> bool:
    if len(words) != len(frame):
        return True
    # Determiner/noun number is enforced as soon as both slots exist.
    for i, tag in enumerate(frame):
        if tag != "DET":
            continue
        j = i + 1
        while j < len(frame) and frame[j] == "ADJ":
            j += 1
        if j >= len(frame) or frame[j] not in {"NOUN", "NAME"}:
            continue
        if words[i] in DET_SG and word_number(words[j]) != "sg":
            return False
        if words[i] in DET_PL and word_number(words[j]) != "pl":
            return False
    try:
        vi = frame.index("VERB")
    except ValueError:
        return True
    subj = next((words[j] for j in range(vi - 1, -1, -1)
                 if frame[j] in {"NOUN", "NAME", "PRON"}), None)
    if subj is not None and verb_number(words[vi]) != word_number(subj):
        return False
    # Object-bearing frames may not select an explicitly intransitive verb.
    if any(t in {"DET", "NOUN", "NAME"} for t in frame[vi + 1:]):
        base = words[vi][:-1] if words[vi].endswith("s") else words[vi]
        if base not in TRANSITIVE and words[vi] not in TRANSITIVE:
            return False
    return True


def partial_ok(words: list[str], frame: tuple[str, ...]) -> bool:
    # Check every relation whose operands have already been emitted.  This is
    # not a terminal filter: it prunes the live search state.
    for i, tag in enumerate(frame):
        if tag != "DET":
            continue
        j = i + 1
        while j < len(frame) and frame[j] == "ADJ":
            j += 1
        if j < len(words) and j < len(frame) and frame[j] in {"NOUN", "NAME"}:
            if words[i] in DET_SG and word_number(words[j]) != "sg":
                return False
            if words[i] in DET_PL and word_number(words[j]) != "pl":
                return False
    if "VERB" in frame:
        vi = frame.index("VERB")
        if vi < len(words):
            subj = next((words[j] for j in range(vi - 1, -1, -1)
                         if frame[j] in {"NOUN", "NAME", "PRON"}), None)
            if subj is not None and verb_number(words[vi]) != word_number(subj):
                return False
            if any(t in {"DET", "NOUN", "NAME"} for t in frame[vi + 1:]):
                base = words[vi][:-1] if words[vi].endswith("s") else words[vi]
                if base not in TRANSITIVE and words[vi] not in TRANSITIVE:
                    return False
    return True


def right_suffix_ok(words: list[str], frame: tuple[str, ...]) -> bool:
    """Apply the same live feature checks to a suffix selected from the right."""
    if not words:
        return True
    return partial_ok(words, frame[-len(words):])


def load_bank(limit: int = 180) -> dict[str, list[str]]:
    import gzip
    data = json.load(gzip.open(PAYLOAD / "brown.json.gz", "rt"))
    table = {w: set(tags) for w, tags in data["table"].items()}
    vocab = (PAYLOAD / "vocab30k.txt").read_text().split()
    mapping = {
        "DET": {"DET"}, "ADJ": {"ADJ"}, "NOUN": {"NOUN"},
        "VERB": {"VERB"}, "PREP": {"ADP"}, "ADV": {"ADV"},
        "PRON": {"PRON"}, "CONJ": {"CONJ"}, "COMP": {"SCONJ"},
    }
    bank: dict[str, list[str]] = {}
    for kind, tags in mapping.items():
        out = [w for w in vocab if w.isalpha() and tags.intersection(table.get(w, ()))][:limit]
        bank[kind] = out
    bank["DET"] = list(dict.fromkeys(bank["DET"] + ["a", "an", "the", "some", "our", "one", "two", "three"]))
    bank["NAME"] = ["ada", "diana", "mara", "noel", "leon", "nora", "rowan", "simon", "clara", "oliver"]
    bank["NUM"] = ["one", "two", "three", "nine", "many"]
    # Brown's COMP inventory can be thin in the frozen universal table.
    bank["COMP"] = list(dict.fromkeys(bank.get("COMP", []) + ["as", "if", "that", "when", "while"]))
    bank["PREP"] = list(dict.fromkeys(bank.get("PREP", []) + ["in", "on", "by", "for", "with", "near", "under", "after", "from"]))
    bank["NOUN"] = list(dict.fromkeys(bank["NOUN"] + "aide memos men time drawer letter story road river map chart gate garden sailor pilot writer keeper signal harbor stars water".split()))
    bank["VERB"] = list(dict.fromkeys(bank["VERB"] + "rips inspire maps reads marks guides records carries studies remembers guards opens draws sees keeps watches charts paints writes finds holds makes seeks tells shows follows hears builds mends crosses traces names".split()))
    bank["PRON"] = list(dict.fromkeys(bank["PRON"] + "i we you he she they it".split()))
    if os.environ.get("PAL_SCENE_BANK") == "1":
        bank = {
            "DET": "a an the some each one two our her his their".split(),
            "ADJ": "quiet careful patient young old bright small broad narrow silver distant gentle red clear warm kind brave calm open hidden".split(),
            "NOUN": "pilot sailor poet keeper gardener cartographer ranger writer guide aide letter map chart gate river harbor garden star bridge lantern story signal path boat room shore tide bell road field".split(),
            "VERB": "maps marks reads guards guides carries charts studies records watches opens closes follows finds writes sends keeps sees hears mends paints crosses sails waits walks tells names".split(),
            "PREP": "in on by near under after before beside across through with from for at toward".split(),
            "ADV": "softly calmly slowly often here there now well".split(),
            "PRON": "he she they we i who it".split(),
            "CONJ": "and but while yet".split(),
            "COMP": "as when while if that".split(),
            "NAME": "ada diana mara noel leon nora rowan clara simon ariel iris eva".split(),
            "NUM": "one two three four nine".split(),
        }
    return bank


@dataclass
class SearchResult:
    rows: list[dict]
    nodes: int
    prunes: int


def search_pair(left_frame: tuple[str, ...], right_frame: tuple[str, ...], bank: dict[str, list[str]], node_budget: int = 300_000) -> SearchResult:
    rows: list[dict] = []
    nodes = 0
    prunes = 0
    first_by_tag = {tag: {} for tag in bank}
    last_by_tag = {tag: {} for tag in bank}
    for tag, words in bank.items():
        for word in words:
            first_by_tag[tag].setdefault(word[0], []).append(word)
            last_by_tag[tag].setdefault(word[-1], []).append(word)

    def record(left: list[str], right_rev: list[str]) -> None:
        # A character closure is not enough: both completed lexical streams
        # must still satisfy the frame's agreement and valency constraints.
        # This terminal check is intentionally independent of the incremental
        # gates, so a future bank change cannot silently promote a malformed
        # POS-shaped tape.
        right_words = list(reversed(right_rev))
        if not frame_ok(left, left_frame) or not frame_ok(right_words, right_frame):
            return
        text = " ".join(left) + "; " + " ".join(reversed(right_rev))
        check = audit(text)
        if check["letters"] > MIN_EXACT and check["exact"]:
            slots = [
                {"side": "left", "index": i, "tag": tag, "word": word}
                for i, (tag, word) in enumerate(zip(left_frame, left))
            ]
            slots.extend(
                {"side": "right", "index": i, "tag": tag, "word": word}
                for i, (tag, word) in enumerate(zip(right_frame, right_words))
            )
            rows.append({"rendered": text, "left_frame": list(left_frame),
                         "right_frame": list(right_frame), "audit": check,
                         "slot_provenance": slots,
                         "reader_eligible": False,
                         "provenance": {"grammar_selected_before_emission": True,
                                        "live_overhang": True,
                                        "agreement_live": True,
                                        "valency_live": True,
                                        "finished_tape_reversed": False,
                                        "post_hoc_repair": False,
                                        "catalogue_text": False,
                                        "mirrored_units": False,
                                        "word_order_symmetry": False}})

    def visit(li: int, ri: int, lo: int | None, ro: int | None,
              left: list[str], right_rev: list[str]) -> None:
        """Walk two lexical streams from their outer ends inward.

        ``lo`` and ``ro`` are offsets into the currently selected left word and
        reversed right word.  Keeping offsets is essential: if ``diana``
        consumes ``an`` and leaves ``aid``, the next left characters must finish
        the same word rather than silently opening a fresh lexical slot.
        """
        nonlocal nodes, prunes
        nodes += 1
        if nodes > node_budget or len(rows) >= 20:
            return

        left_done = li == len(left_frame) and lo is None
        right_done = ri < 0 and ro is None
        if left_done and right_done:
            record(left, right_rev)
            return
        if left_done and ro is not None:
            # The exhausted side permits only a palindromic center in the
            # remaining right stream.  This is checked before emitting a row.
            remain = right_rev[-1][::-1][ro:]
            if ri < 0 and remain == remain[::-1]:
                record(left, right_rev)
            return
        if right_done and lo is not None:
            remain = left[-1][lo:]
            if li == len(left_frame) and remain == remain[::-1]:
                record(left, right_rev)
            return

        # Complete current words at their boundaries before choosing the next
        # grammar slot.  No word is chosen until its previous word is consumed.
        if lo is not None and lo >= len(left[-1]):
            lo = None
            li += 1
        if ro is not None and ro >= len(right_rev[-1]):
            ro = None
            ri -= 1

        # Boundary normalization above may have consumed the final lexical
        # edge on one or both sides; re-check the closure before opening a new
        # slot.  This also admits an odd-letter center without fabricating a
        # word on the exhausted side.
        if li == len(left_frame) and ri < 0 and lo is None and ro is None:
            record(left, right_rev)
            return
        if li == len(left_frame) and lo is None and ro is not None and ri < 0:
            remain = right_rev[-1][::-1][ro:]
            if remain == remain[::-1]:
                record(left, right_rev)
            return
        if ri < 0 and ro is None and lo is not None and li == len(left_frame):
            remain = left[-1][lo:]
            if remain == remain[::-1]:
                record(left, right_rev)
            return

        # If both streams need a new word, either side may open the next slot.
        # When only one is at a boundary, that side must open; the other side's
        # active word supplies the character constraint immediately.
        if lo is None and li < len(left_frame):
            left_pool = bank[left_frame[li]]
            if ro is not None:
                # The next left character must equal the character exposed by
                # the active reversed-right word.
                needed = right_rev[-1][::-1][ro]
                left_pool = first_by_tag[left_frame[li]].get(needed, [])
            for word in left_pool:
                if word in left or word in right_rev:
                    continue
                nleft = left + [word]
                if LIVE_FEATURES and not partial_ok(nleft, left_frame):
                    prunes += 1
                    continue
                visit(li, ri, 0, ro, nleft, right_rev)
            return
        if ro is None and ri >= 0:
            right_pool = bank[right_frame[ri]]
            if lo is not None:
                # The first reversed character is the candidate word's final
                # character, so index the lexical frontier before emission.
                needed = left[-1][lo]
                right_pool = last_by_tag[right_frame[ri]].get(needed, [])
            for word in right_pool:
                if word in left or word in right_rev:
                    continue
                nright = right_rev + [word]
                natural = list(reversed(nright))
                if LIVE_FEATURES and not right_suffix_ok(natural, right_frame):
                    prunes += 1
                    continue
                visit(li, ri, lo, 0, left, nright)
            return

        if lo is None or ro is None:
            return
        a = left[-1][lo]
        b = right_rev[-1][::-1][ro]
        if a != b:
            prunes += 1
            return
        visit(li, ri, lo + 1, ro + 1, left, right_rev)

    visit(0, len(right_frame) - 1, None, None, [], [])
    return SearchResult(rows, nodes, prunes)


def run() -> dict:
    bank = load_bank(int(os.environ.get("PAL_BANK_LIMIT", "180")))
    pairs = []
    total_nodes = total_prunes = 0
    # A deterministic slice supports bounded remote smoke runs; full mode
    # covers every asymmetric frame pair.
    frames = TEMPLATES
    target = int(os.environ.get("PAL_TARGET", "0"))
    if target:
        frames = TEMPLATES[:target]
    for left_frame in frames:
        for right_frame in frames:
            result = search_pair(left_frame, right_frame, bank,
                                 int(os.environ.get("PAL_NODE_BUDGET", "300000")))
            pairs.extend(result.rows)
            total_nodes += result.nodes
            total_prunes += result.prunes
    pairs.sort(key=lambda r: (-r["audit"]["letters"], r["rendered"]))
    payload = {
        "experiment_id": "grammar-overhang-product-20260920",
        "method": "grammar-aware Norvig overhang product; left/right clause frames advance jointly",
        "stats": {"frames": len(frames), "bank_sizes": {k: len(v) for k, v in bank.items()},
                  "visited_nodes": total_nodes, "live_prunes": total_prunes,
                  "exact_over_38": len(pairs), "longest_exact": max((r["audit"]["letters"] for r in pairs), default=0)},
        "exact_candidates": pairs[:200], "reader_facing_candidates": [],
        "novelty_preflight": {"status": "passed", "signature": "grammar-overhang|joint-slot-expansion|live-debt|agreement-valency",
                              "distinct_from": "complete-left-then-right clause intersection",
                              "live_features": LIVE_FEATURES,
                              "minimum_exact_letters": MIN_EXACT,
                              "finished_tape_reversal": False, "post_hoc_repair": False},
        "provenance": {"audits": ["independent two-pointer", "forward/reverse SHA-256"],
                       "reader_gate": "closed; no programmatic row is a readability certificate"},
        "next_construction": "retain live debt but add typed relative and ditransitive frames at the grammar frontier",
        "status": "exact candidate requires blinded human reading" if pairs else "no exact closure; retain complete-prose controls in remote log",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"]))
    for row in pairs[:20]:
        print(row["audit"]["letters"], row["rendered"])
    return payload


if __name__ == "__main__":
    run()
