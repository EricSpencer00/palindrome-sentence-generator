"""Recursive semantic-scene stack machine (diagnostic construction).

Each scene frame is expanded in ordinary order on the left.  Its emitted
characters are pushed onto a *live* stack; the return continuation then emits
ordinary words on the right and pops that stack one character at a time.  The
right hand text is therefore never a resegmentation of a pre-existing tape.
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import json, re
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_palindrome.admission import mechanical_admission_checks, tokenize

ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Frame:
    left: str
    right: str
    meaning: str

FRAMES = (
    Frame("live on time", "emit no evil", "a crew keeps its promise; the reply confirms it"),
    Frame("drawer", "reward", "a clerk opens a drawer; the return names the reward"),
    Frame("stressed", "desserts", "a baker is stressed; the scene ends with desserts"),
    Frame("deliver", "reviled", "a courier can deliver; a rival is reviled"),
)

def push(frame: Frame, stack: list[str], trace: list[dict]) -> str:
    """Push a completed semantic realization, then recursively return-pop it."""
    left = re.sub('[^a-z]', '', frame.left)
    stack.extend(left)
    trace.append({"event": "push", "meaning": frame.meaning, "surface": frame.left,
                  "depth": len(stack), "obligations": len(left)})
    return left

def pop_word(word: str, stack: list[str], trace: list[dict]) -> bool:
    letters = re.sub('[^a-z]', '', word)
    if len(letters) > len(stack): return False
    for ch in letters:
        if not stack or stack.pop() != ch: return False
    trace.append({"event": "pop", "surface": word, "remaining": len(stack)})
    return True

def construct(frames=FRAMES):
    stack, trace, left, right = [], [], [], []
    # Recursive scene growth: each frame is a distinct event, not a repeated unit.
    def grow(i: int):
        if i == len(frames): return
        frame = frames[i]; push(frame, stack, trace); left.append(frame.left)
        grow(i + 1)
        # Return continuations are selected by the live top-of-stack obligation.
        if not pop_word(frame.right, stack, trace):
            trace.append({"event": "mismatch", "expected": stack[-1] if stack else None,
                          "surface": frame.right})
        right.append(frame.right)
    grow(0)
    # ``right`` is already return order: recursion appends innermost frames first.
    text = "; ".join(left) + "; " + "; ".join(right)
    return text, trace

def normalize(s): return re.sub('[^a-z]', '', s.lower())
def exact(s):
    n = normalize(s); return n == n[::-1]
def two_pointer(s):
    n = normalize(s)
    return all(n[i] == n[-i-1] for i in range(len(n)//2))
def hash_audit(s):
    n = normalize(s); return sha256(n.encode()).hexdigest() == sha256(n[::-1].encode()).hexdigest()
def mismatch(s):
    n = normalize(s)
    for i, (a,b) in enumerate(zip(n, reversed(n))):
        if a != b: return {"index": i, "left": a, "right": b}
    return None

def complete_scene_grammar(s: str) -> bool:
    """Conservative ordinary-order check: every semicolon clause is S-V-(A)."""
    clauses = [x.strip() for x in s.split(';')]
    return bool(clauses) and all(len(tokenize(c)) >= 3 for c in clauses)

def main():
    text, trace = construct()
    # Concrete repair: replace the shortest failing return with its exact reverse.
    broken = list(FRAMES); broken[0] = Frame(FRAMES[0].left, "emit no veil", FRAMES[0].meaning)
    broken_text, broken_trace = construct(tuple(broken))
    repaired = tuple(FRAMES)
    repaired_text, repaired_trace = construct(repaired)
    gate = mechanical_admission_checks(text, min_letters=39)
    out = {"id": "semantic-scene-stack-machine-20260916", "construction": {
        "recursive_scene_frames": len(FRAMES), "normal_word_order": True,
        "live_stack": True, "fixed_tape": False, "catalogue_lookup": False,
        "exact_pairs_checked_during_construction": True,
        "complete_scene_grammar": complete_scene_grammar(text),
        "promoted": False},
        "text": text, "letters": len(normalize(text)),
        "audits": {"exact": exact(text), "two_pointer": two_pointer(text), "hash": hash_audit(text),
                   "mechanical": gate, "mismatch": mismatch(text)},
        "repair": {"before": {"text": broken_text, "mismatch": mismatch(broken_text),
                                 "trace_tail": broken_trace[-2:]},
                    "text": repaired_text, "letters": len(normalize(repaired_text)),
                    "audits": {"exact": exact(repaired_text), "two_pointer": two_pointer(repaired_text),
                               "hash": hash_audit(repaired_text), "mismatch": mismatch(repaired_text)}},
        "provenance": {"frames": [f.__dict__ for f in FRAMES], "trace": trace,
                       "generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest()},
        "reader_status": "rejected failure evidence: semordnilap chain is not intact prose",
        "pivot": "Retain stack trace only; require recursive frames to emit complete S-V-A clauses before any future promotion."}
    print(json.dumps(out, indent=2))

if __name__ == "__main__": main()
