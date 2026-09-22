"""Packed function-word/attachment CFG probe.

This lane keeps function words and attachment scope in the live search state:
determiners, auxiliaries, prepositions, and relative markers are selected
before character obligations are emitted.  It deliberately does not import
the incumbent or any reversed-word catalogue.  A small ordinary-English CFG
is intersected with its character reversal; complete sentences are never
enumerated first.
"""
from collections import defaultdict, deque
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def norm(text):
    return re.sub(r"[^a-z]", "", text.lower())


def audit(text):
    tape = norm(text)
    mismatch = next(((i, tape[i], tape[-1-i]) for i in range(len(tape)//2)
                     if tape[i] != tape[-1-i]), None)
    return {"letters": len(tape), "exact": bool(tape) and mismatch is None,
            "first_mismatch": mismatch,
            "sha256": hashlib.sha256(tape.encode()).hexdigest(),
            "reverse_sha256": hashlib.sha256(tape[::-1].encode()).hexdigest(),
            "normalized": tape}


class CFG:
    def __init__(self):
        self.edges = defaultdict(list)
        self.next_state = 1
        self.start = 0
        self.finish = 0

    def slot(self, alternatives, role):
        before, after = self.finish, self.next_state
        self.next_state += 1
        for phrase, tags in alternatives:
            chars = norm(phrase)
            cur = before
            for pos, char in enumerate(chars):
                nxt = after if pos == len(chars)-1 else self.next_state
                if nxt != after:
                    self.next_state += 1
                self.edges[cur].append((nxt, char, role, phrase, tags))
                cur = nxt
        self.finish = after


def grammar():
    g = CFG()
    # Attachment is a typed state, not a post-hoc string rewrite.  Each
    # content event chooses its function-word frame jointly with scope.
    g.slot((("the", ("det", "def")), ("a", ("det", "indef")),
            ("one", ("det", "indef"))), "det")
    g.slot((("scribe", ("n", "human")), ("guard", ("n", "human")),
            ("pilot", ("n", "human")), ("poet", ("n", "human"))), "subject")
    g.slot((("can", ("aux", "modal")), ("will", ("aux", "modal")),
            ("must", ("aux", "modal")), ("does", ("aux", "do"))), "aux")
    g.slot((("read", ("v", "trans")), ("write", ("v", "trans")),
            ("carry", ("v", "trans")), ("mark", ("v", "trans"))), "event")
    g.slot((("the", ("det", "def")), ("a", ("det", "indef")),
            ("one", ("det", "indef"))), "object_det")
    g.slot((("map", ("n", "artifact")), ("note", ("n", "artifact")),
            ("letter", ("n", "artifact")), ("book", ("n", "artifact"))), "object")
    # The attachment choice is live: PP, relative clause, or no attachment.
    g.slot((("", ("attach", "none")), ("in", ("prep", "loc")),
            ("by", ("prep", "agent")), ("with", ("prep", "instrument"))), "attachment")
    g.slot((("", ("rel", "none")), ("that", ("rel", "restrictive")),
            ("which", ("rel", "restrictive"))), "relative_marker")
    g.slot((("", ("tail", "none")), ("it", ("pron", "object")),
            ("us", ("pron", "object")), ("them", ("pron", "object"))), "attachment_tail")
    return g


def intersect(g, cap=150000, max_letters=180):
    # Pair two positions in the same packed grammar.  The forward edge on the
    # left must equal the backward edge on the right; tags remain attached to
    # every transition and are emitted only after the pair is accepted.
    incoming = defaultdict(list)
    for src, edges in g.edges.items():
        for eid, edge in enumerate(edges):
            incoming[edge[0]].append((src, eid, edge))
    q = deque([(g.start, g.finish, (), (), ())])
    seen = set(); dead = []; candidates = {}
    transitions = 0
    while q and len(seen) < cap:
        left, right, lp, rp, tags = q.popleft()
        key = (left, right, len(lp))
        if key in seen: continue
        seen.add(key)
        if left == g.finish and right == g.start and lp:
            pieces = [g.edges[src][eid][3] for src, eid in lp]
            text = " ".join(pieces).capitalize() + "."
            check = audit(text)
            if check["exact"]:
                candidates[check["normalized"]] = {"rendered": text,
                    "audit": check, "roles": list(tags), "provenance": "packed_cfg"}
        if 2*len(lp) >= max_letters: continue
        outs = g.edges[left]
        ins = incoming[right]
        by_char = defaultdict(list)
        for eid, edge in enumerate(outs): by_char[edge[1]].append((eid, edge))
        in_char = defaultdict(list)
        for src, eid, edge in ins: in_char[edge[1]].append((src, eid, edge))
        common = sorted(set(by_char) & set(in_char))
        if not common:
            dead.append({"depth": len(lp), "left": left, "right": right,
                         "left_chars": sorted(by_char), "right_chars": sorted(in_char)})
        for char in common:
            for le, ledge in by_char[char]:
                for rsrc, reid, redge in in_char[char]:
                    q.append((ledge[0], rsrc, lp+((left, le),),
                              rp+((rsrc, reid),), tags+(ledge[2],)))
                    transitions += 1
    return {"states": len(seen), "transitions": transitions,
            "candidates": sorted(candidates.values(), key=lambda x: -x["audit"]["letters"]),
            "dead_frontiers": sorted(dead, key=lambda x: -x["depth"])[:20],
            "grammar_states": g.next_state,
            "grammar_edges": sum(map(len, g.edges.values()))}


def run():
    result = intersect(grammar())
    result.update({"experiment_id": "function-attachment-cfg-20260930",
        "method": "typed function-word and attachment CFG with live character reversal intersection",
        "ordinary_controls": ["The scribe can read the map in the hall.",
                               "A pilot will mark a letter that guides us."],
        "provenance": {"complete_sentence_enumeration": False,
                        "reversed_catalogue": False, "seed_reused": False,
                        "posthoc_repair": False, "per_candidate_rlaif": False,
                        "function_words_live": True, "attachment_scope_live": True,
                        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "novelty_preflight": {"status": "new orthogonal lane",
            "distinction": "function-word and attachment choices are transition tags in the packed product, not a fixed content-word scaffold"},
        "readability_evidence": {"status": "not human-rated",
            "programmatic_metrics_are_diagnostic": True,
            "reader_protocol": "Any exact candidate is shown with provenance and randomized blinded intact/shuffled controls before readability claims."},
        "next_operator": "Add a typed finite relative-clause attachment with independently chosen subject and agreement, retaining live function-word scope; do not widen the same lexical bank."})
    return result


if __name__ == "__main__":
    path = ROOT / "runs" / "function-attachment-cfg-20260930.json"
    path.write_text(json.dumps(run(), indent=2) + "\n")
    print(json.dumps(run(), indent=2))
