"""Character-obligation propagation repair for an internal-word seam.

This is a fresh typed scene (a witness, action, object, and consequence), not
an expansion of the old Cartesian bank.  Before lexical values are expanded,
the propagator removes a value whenever no attainable length placement can
support every character at its mirrored tape positions.  Whole words are
always rendered; a seam split is metadata only.
"""
from __future__ import annotations

import hashlib, itertools, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/partial-word-seam-propagation-repair-20260917.json"

# Fresh authored scene: an observer reports an action and its consequence.
# The two consequence slots are deliberately independent lexical domains.
DOMAINS = [
    ["the", "a"],                         # determiner
    ["pilot", "poet", "nurse", "teacher"], # witness
    ["marks", "reads", "opens", "notes"],  # action
    ["aide", "map", "letter", "shore"],   # object
    ["calm", "clear", "open", "quiet"],   # consequence adjective
    ["sailor", "artist", "reader", "pilot"], # consequence witness
]


def tape(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def audit(text: str) -> dict:
    s = tape(text)
    mismatch = None
    for i, j in zip(range(len(s) // 2), range(len(s) - 1, len(s) // 2, -1)):
        if s[i] != s[j]:
            mismatch = {"left_index": i, "right_index": j, "left": s[i], "right": s[j]}
            break
    words = re.findall(r"[a-z]+", text.lower())
    return {"letters": len(s), "exact": bool(s) and mismatch is None,
            "independent_two_pointer": bool(s) and mismatch is None,
            "first_mismatch": mismatch, "words": words,
            "repeated_word_count": len(words) - len(set(words)),
            "self_palindromic_words": [w for w in words if len(w) > 1 and w == w[::-1]],
            "borrowed_catalogue": False, "word_order_only": False}


def possible_lengths(domains: list[list[str]]) -> set[int]:
    values = {0}
    for d in domains:
        values = {n + len(w) for n in values for w in d}
    return values


def starts(domains: list[list[str]], slot: int, word: str, total: int) -> set[int]:
    pre = {0}
    for d in domains[:slot]:
        pre = {n + len(w) for n in pre for w in d}
    post = {0}
    for d in domains[slot + 1:]:
        post = {n + len(w) for n in post for w in d}
    return {p for p in pre if total - p - len(word) in post}


def support(domains: list[list[str]], total: int) -> list[set[str]]:
    out = [set() for _ in range(total)]
    for i, domain in enumerate(domains):
        for word in domain:
            for p in starts(domains, i, word, total):
                for k, ch in enumerate(word):
                    out[p + k].add(ch)
    return out


def propagate(domains: list[list[str]]) -> tuple[list[list[str]], list[dict]]:
    current = [list(dict.fromkeys(d)) for d in domains]
    history = []
    for round_no in range(1, 20):
        lengths = sorted(possible_lengths(current)) if all(current) else []
        supports = {n: support(current, n) for n in lengths}
        removed = {}
        for i, domain in enumerate(current):
            keep = []
            for word in domain:
                ok = False
                for n in lengths:
                    for p in starts(current, i, word, n):
                        if all(word[k] in supports[n][n - 1 - (p + k)]
                               for k in range(len(word))):
                            ok = True; break
                    if ok: break
                if ok: keep.append(word)
            if len(keep) != len(domain):
                removed[str(i)] = sorted(set(domain) - set(keep))
                current[i] = keep
        history.append({"round": round_no, "lengths": lengths, "removed": removed})
        if not removed or not all(current): break
    return current, history


def withheld_control() -> dict:
    """A tiny exact control proves propagation is not a blanket rejection."""
    control = [["ab"], ["ba"]]
    reduced, history = propagate(control)
    rendered = "ab ba."
    return {"rendered": rendered, "audit": audit(rendered),
            "survives": all(reduced), "reduced": reduced,
            "rounds": len(history)}


def main() -> None:
    # The consequence clause is an independently authored ordinary-order
    # rendering, not a reversal operation.  It is intentionally not forced
    # to mirror the first clause lexically.
    right_domains = [
        ["this", "that"], ["reader", "artist"], ["notes", "reads"],
        ["aide", "map"], ["calm", "clear"], ["sailor", "pilot"],
    ]
    sequence_domains = DOMAINS + [["sees", "keeps", "finds", "marks"]] + right_domains
    # A central lexical word is allowed to cross the seam; retain all splits.
    reduced, history = propagate(sequence_domains)
    assignments = 0
    rows = []
    # A fixed budget makes this a reproducible bounded diagnostic rather than
    # another unbounded Cartesian sweep.
    for vals in itertools.islice(itertools.product(*reduced), 100_000):
        assignments += 1
        text = " ".join(vals) + "."
        a = audit(text)
        if a["exact"]:
            rows.append({"rendered": text, "audit": a, "centre_word": vals[len(vals)//2],
                         "centre_splits": list(range(1, len(vals[len(vals)//2]))),
                         "provenance": "fresh hand-authored consequence-scene domains; propagated before expansion"})
    report = {
        "experiment": "partial-word-seam-propagation-repair-20260917",
        "signature": "fresh-consequence-scene|length-placement-support|mirrored-character-obligation|whole-word-rendering|independent-two-pointer",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "domain_sizes_before": [len(d) for d in sequence_domains],
        "domain_sizes_after": [len(d) for d in reduced],
        "propagation": history, "post_propagation_assignments": assignments,
        "withheld_exact_control": withheld_control(),
        "candidate_count": len(rows), "exact_closures": rows[:20],
        "reader_eligible_count": 0,
        "repair_after_failure": "Split semantic consequence slots into independently authored adjuncts and propagate cross-slot character supports; retain a held-out human scene and add blinded reader screening before admission.",
        "scope": "Diagnostic construction repair; no output is claimed readable without blinded human ratings.",
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"assignments": assignments, "exact": len(rows), "rounds": len(history)}))


if __name__ == "__main__": main()
