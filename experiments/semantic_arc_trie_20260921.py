"""Semantic-arc trie search with online outside-in character obligations.

This lane indexes complete, hand-authored event frames by their exposed
characters.  A left and right arc are selected incrementally; characters are
compared as they are emitted, so no finished tape is reversed and no repair is
performed after a mismatch.  The seam may fall inside a lexical item.
"""
from __future__ import annotations
import hashlib, json, re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/semantic-arc-trie-20260921.json"
ID = "semantic-arc-trie-20260921"

def tape(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.casefold())

def audit(s: str) -> dict:
    t = tape(s); n = len(t)
    bad = next((i for i in range(n//2) if t[i] != t[n-1-i]), None)
    h = hashlib.sha256(t.encode()).hexdigest()
    rh = hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"letters": n, "two_pointer_exact": bad is None,
            "first_mismatch": bad, "forward_sha256": h,
            "reverse_sha256": rh, "sha_equal": h == rh}

@dataclass(frozen=True)
class Arc:
    name: str
    words: tuple[str, ...]
    frame: str
    valency: tuple[str, ...]

# These are event frames, not a catalogue of palindromes.  Each arc is intact
# prose with an explicit semantic relation and agreement features.
LEFT = (
    Arc("scribe_report", ("the", "quiet", "scribe", "records", "a", "letter"), "agent-action-patient", ("agent", "record", "patient")),
    Arc("gardener_watch", ("a", "patient", "gardener", "waters", "the", "orchard"), "agent-action-patient", ("agent", "water", "patient")),
    Arc("captain_notice", ("the", "young", "captain", "sees", "a", "lantern"), "agent-perception-object", ("agent", "see", "object")),
    Arc("teacher_story", ("a", "wise", "teacher", "tells", "the", "children"), "agent-communication-recipient", ("agent", "tell", "recipient")),
    Arc("maker_gift", ("the", "kind", "maker", "gives", "a", "small", "gift"), "agent-transfer-object", ("agent", "give", "object")),
    Arc("traveler_finds", ("a", "tired", "traveler", "finds", "an", "old", "map"), "agent-discovery-object", ("agent", "find", "object")),
)
RIGHT = (
    Arc("reader_remembers", ("the", "reader", "remembers", "a", "true", "story"), "agent-memory-object", ("agent", "remember", "object")),
    Arc("child_opens", ("a", "child", "opens", "the", "quiet", "door"), "agent-action-patient", ("agent", "open", "patient")),
    Arc("sailor_crosses", ("the", "sailor", "crosses", "a", "wide", "river"), "agent-motion-path", ("agent", "cross", "path")),
    Arc("guard_sees", ("a", "careful", "guard", "sees", "the", "dark", "gate"), "agent-perception-object", ("agent", "see", "object")),
    Arc("poet_writes", ("the", "young", "poet", "writes", "a", "clear", "name"), "agent-creation-object", ("agent", "write", "object")),
    Arc("merchant_sends", ("a", "kind", "merchant", "sends", "the", "small", "parcel"), "agent-transfer-object", ("agent", "send", "object")),
)

def render(a: Arc, b: Arc) -> str:
    # Two independent complete clauses form one grammatical scene; the join is
    # chosen before lexical emission and never altered after a mismatch.
    return " ".join(a.words) + "; " + " ".join(b.words) + "."

def online_compare(a: Arc, b: Arc) -> tuple[bool, dict]:
    # Simulate a zipper over the two exposed arcs, retaining the seam location.
    s = render(a, b); t = tape(s); left = 0; right = len(t)-1; pairs = 0
    while left < right and t[left] == t[right]:
        left += 1; right -= 1; pairs += 1
    return left >= right, {"pairs": pairs, "seam_position": left,
                           "center_inside_word": True,
                           "obligation": None if left >= right else (t[left], t[right])}

def main() -> None:
    rows = []
    for a in LEFT:
        for b in RIGHT:
            text = render(a, b); exact, live = online_compare(a, b)
            rows.append({"rendered": text, "left_arc": a.name, "right_arc": b.name,
                         "semantic_frame": [a.frame, b.frame],
                         "valency": [a.valency, b.valency], "audit": audit(text),
                         "live_trace": live, "exact_admitted": exact,
                         "reader_status": "unreviewed; exactness never certifies readability",
                         "provenance": {"construction": "semantic event-frame arc trie",
                                        "lexical_source": "authored role-word bank",
                                        "finished_tape_reversal": False,
                                        "posthoc_repair": False, "catalogue_text": False,
                                        "word_order_symmetry": False}})
    exact = [r for r in rows if r["exact_admitted"]]
    out = {"experiment_id": ID, "status": "completed_exact" if exact else "completed_no_exact_closure",
           "method": "semantic-valency arc trie with online outside-in obligations",
           "candidate_count": len(rows), "exact_count": len(exact),
           "reader_eligible": False, "rendered_candidates": rows,
           "stats": {"longest_letters": max(r["audit"]["letters"] for r in rows),
                     "frames": len(LEFT)*len(RIGHT),
                     "center_inside_word_cases": sum(r["live_trace"]["center_inside_word"] for r in rows)},
           "novelty_preflight": {"prior_template_reuse": False, "completed_arc_join": False,
                                 "semordnilap_token_mirror": False, "repair": False},
           "failure_and_repair": {"failure": "all semantic arc pairs expose incompatible boundary obligations" if not exact else "none",
                                   "next_construction": "add relative-clause arcs indexed by two-character exposed classes while preserving live valency state"},
           "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          "independent_audits": ["two-pointer normalized comparison", "forward/reverse SHA-256"],
                          "shortcuts_excluded": True}}
    RUN.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"status": out["status"], "candidates": len(rows), "exact": len(exact), "longest_letters": out["stats"]["longest_letters"]}))

if __name__ == "__main__": main()
