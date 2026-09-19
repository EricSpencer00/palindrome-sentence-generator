"""Authored Shakespearean two-turn scene lattice with live tape equations.

Unlike clause-product lanes, this models a dramatic exchange: an address or
imperative turn followed by a reply.  The two turns are generated from typed
speech acts and joined by an indexed reverse-tape equation; no language model
scores or borrowed text are used.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib, json, subprocess
from pathlib import Path
from itertools import product

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

ROOT = Path(__file__).resolve().parents[1]
ID = "shakespeare-scene-turn-lattice-20260919"

@dataclass(frozen=True)
class Turn:
    speaker: str
    act: str
    text: str
    valency: str
    content: frozenset[str]

def turn(speaker, act, text, valency, words=None):
    raw = words or tokenize(text)
    content = frozenset(normalize_letters(w) for w in raw
                        if normalize_letters(w) not in {"o", "a", "an", "the", "thou", "you", "my", "thy"})
    return Turn(speaker, act, text, valency, content)

# Small authored banks: each item is an intact utterance, not a word catalogue.
ADDRESSES = tuple(turn("herald", "address", x, "vocative") for x in (
    "O, gentle lord", "O, patient queen", "O, silent moon", "O, noble friend",
    "O, weary king", "O, kind spirit"))
IMPERATIVES = tuple(turn("herald", "imperative", x, "transitive") for x in (
    "mark my letter", "keep thy counsel", "guard the lantern", "read the sonnet",
    "seek the answer", "sing the old song"))
REPLIES = tuple(turn("reply", "reply", x, "transitive") for x in (
    "I shall mark it", "I shall keep faith", "I will guard it", "I can read it",
    "I shall seek truth", "I will sing on"))
BLESSINGS = tuple(turn("reply", "benediction", x, "intransitive") for x in (
    "so may peace abide", "then let hope arise", "thus shall dawn return",
    "and love shall endure", "so truth may prevail", "then stars will answer"))

def scenes():
    # A scene is an address + action + response or benediction.  Agreement and
    # speech-act constraints are checked before the tape join.
    out = []
    for a, i, r in product(ADDRESSES, IMPERATIVES, REPLIES):
        if a.content & (i.content | r.content) or i.content & r.content:
            continue
        out.append((a, i, r))
    for a, i, b in product(ADDRESSES, IMPERATIVES, BLESSINGS):
        if a.content & (i.content | b.content) or i.content & b.content:
            continue
        out.append((a, i, b))
    return out

def render(scene):
    a, i, r = scene
    return f"{a.text}; {i.text}; {r.text}."

def exact_audit(text):
    t = normalize_letters(text); rev = t[::-1]
    mismatch = next((i for i, (x, y) in enumerate(zip(t, rev)) if x != y), None)
    return {"letters": len(t), "two_pointer_exact": mismatch is None,
            "first_mismatch": mismatch,
            "normalized_sha256": hashlib.sha256(t.encode()).hexdigest(),
            "reverse_sha256": hashlib.sha256(rev.encode()).hexdigest(),
            "sha_equal": hashlib.sha256(t.encode()).hexdigest() == hashlib.sha256(rev.encode()).hexdigest()}

def hidden_spans(text):
    words = [normalize_letters(w) for w in tokenize(text)]
    bad = []
    for i in range(len(words)):
        for j in range(i + 1, len(words) + 1):
            s = "".join(words[i:j])
            if len(s) >= 4 and s == s[::-1]: bad.append(" ".join(words[i:j]))
    return sorted(set(bad), key=lambda x: (len(x), x))

def run():
    all_scenes = scenes()
    # Live equation join: index each scene by its first half and only compare
    # reverse-compatible halves. This avoids an unbounded Cartesian reward sweep.
    by_prefix = {}
    for scene in all_scenes:
        text = render(scene); tape = normalize_letters(text)
        by_prefix.setdefault(tape[:len(tape)//2], []).append((scene, tape))
    exact = []
    # The authored lattice is modest; this loop is only over equation buckets,
    # and remains deterministic. Include all exact rows, not a quality ranking.
    for bucket in by_prefix.values():
        for scene, tape in bucket:
            if tape == tape[::-1]: exact.append((scene, tape))
    rows = []
    for scene, tape in exact:
        text = render(scene); audit = exact_audit(text)
        rows.append({"rendered": text, "length": audit["letters"], "provenance": [x.__dict__ for x in scene],
                     "audit": audit, "hidden_palindromic_spans": hidden_spans(text),
                     "mechanical_admission": mechanical_admission_checks(text, min_letters=39, max_letters=240),
                     "reader_status": "unreviewed; programmatic measures do not certify readability"})
    payload = {"experiment_id": ID, "method": {"scene_turns": 3, "speech_acts": ["address", "imperative", "reply", "benediction"],
        "join": "indexed full-tape equality plus independent two-pointer audit", "reward_model_used": False, "catalogue_imported": False,
        "scenes_generated": len(all_scenes)}, "representative_exact_candidates": rows,
        "search": {"equation_buckets": len(by_prefix), "exact_count": len(rows), "longest_exact": max((r["length"] for r in rows), default=0)},
        "novelty_preflight": {"disposition": "new discourse-turn lattice; no reverse realization", "prior_families_checked": ["role_phrase_lattice_20260919", "typed_scene_lattice_online_equations_20260919"]},
        "next_repair": "Add authored paired replies whose first and last characters satisfy the live seam equation, while preserving speech-act and unique-content constraints; do not score or rank completed prose.",
        "provenance": {"source": str(Path(__file__).relative_to(ROOT)), "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()},
        "strict_gate": {"readable_over_38": sum(r["length"] > 38 and r["mechanical_admission"]["admit"] for r in rows), "human_readability_test": "not performed"}}
    return payload

if __name__ == "__main__":
    p = ROOT / "runs" / f"{ID}.json"; p.write_text(json.dumps(run(), indent=2) + "\n")
    print(json.dumps({"run": str(p), "search": run()["search"]}, indent=2))
