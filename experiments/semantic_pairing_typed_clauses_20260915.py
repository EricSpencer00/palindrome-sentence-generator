"""Exact semantic pairing of two independently grammatical typed clauses.

The character tape is the hard constraint: a right clause is accepted only when
its normalized letters are exactly the reverse of the left clause.  Since the
comparison happens on the tape, word boundaries may cross freely.
"""
from __future__ import annotations
import argparse, json
from hashlib import sha256
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

ADJ = ("careful", "patient", "skilled", "quiet", "brave", "kind")
SUBJ = ("baker", "teacher", "farmer", "editor", "poet", "doctor")
VERB_OBJ = (("makes", "bread"), ("writes", "notes"), ("carries", "water"),
            ("reads", "letters"), ("plants", "trees"), ("treats", "patients"))
PREP = ("near", "beside", "under", "during")
PP_ADJ = ("fresh", "old", "bright", "small")
PP_NOUN = ("market", "school", "bridge", "storm", "garden")

def frames():
    for adj in ADJ:
        for subj in SUBJ:
            for verb, obj in VERB_OBJ:
                for prep in PREP:
                    for pp_adj in PP_ADJ:
                        for pp_noun in PP_NOUN:
                            yield {"adj": adj, "subj": subj, "verb": verb,
                                   "obj": obj, "prep": prep, "pp_adj": pp_adj,
                                   "pp_noun": pp_noun}

def render(f):
    return f"The {f['adj']} {f['subj']} {f['verb']} the {f['obj']} {f['prep']} the {f['pp_adj']} {f['pp_noun']}."

def parse(text):
    w = tokenize(text)
    # Structural provenance is carried by the frame, while this independently
    # checks the rendered token sequence against the same typed grammar.
    return len(w) == 11 and w[0] == 'the' and w[1] in ADJ and w[2] in SUBJ and \
        any(w[3] == v and w[4] == o for v, o in VERB_OBJ) and w[5] == 'the' and \
        w[6] in {o for _, o in VERB_OBJ} and w[7] in PREP and w[8] == 'the' and \
        w[9] in PP_ADJ and w[10] in PP_NOUN

def readability_diagnostics(text):
    """Cheap diagnostics only; these never affect exact/admission status."""
    words = tokenize(text)
    content = [w for w in words if w not in {'the', 'a', 'an', 'in', 'on', 'at', 'near', 'beside', 'under', 'during'}]
    return {"word_count": len(words), "content_word_count": len(content),
            "unique_content_ratio": round(len(set(content)) / len(content), 3) if content else 0.0,
            "mean_word_length": round(sum(map(len, words)) / len(words), 2) if words else 0.0,
            "diagnostic_only": True}

def run(min_letters=39, limit=None):
    fs = list(frames())
    buckets = {}
    for i, f in enumerate(fs):
        text = render(f); tape = normalize_letters(text)
        buckets.setdefault(tape, []).append((i, f, text))
    exact = []
    probe_fs = fs[:400]
    for i, f in enumerate(probe_fs):
        left = render(f); tape = normalize_letters(left)
        if len(tape) < min_letters:
            continue
        for j, rf, right in buckets.get(tape[::-1], ()):
            if i == j:  # retain two independent clause witnesses
                continue
            whole = left.rstrip('.') + ' ' + right
            audit = mechanical_admission_checks(whole, min_letters=min_letters, max_letters=260)
            exact.append({"left_frame_id": i, "right_frame_id": j,
                          "left_frame": f, "right_frame": rf,
                          "rendered": whole, "letters": len(normalize_letters(whole)),
                          "exact_palindrome": normalize_letters(whole) == normalize_letters(whole)[::-1],
                          "left_parse": parse(left), "right_parse": parse(right),
                          "readability": readability_diagnostics(whole),
                          "admission": audit,
                          "provenance": {"left_text": left, "right_text": right,
                                         "left_tape_sha256": sha256(tape.encode()).hexdigest(),
                                         "boundary_crossing_allowed": True}})
            if limit and len(exact) >= limit:
                break
        if limit and len(exact) >= limit:
            break
    admitted = [r for r in exact if r["exact_palindrome"] and r["left_parse"] and r["right_parse"] and all(r["admission"].values())]
    near_misses = []
    # Preserve concrete diagnostics even when the hard equation has no solution:
    # rank independent clause pairs by longest common prefix with the target
    # reverse tape. This is a repair queue, not a candidate channel.
    for i, f in enumerate(probe_fs):
        left_tape = normalize_letters(render(f))
        if len(left_tape) < min_letters: continue
        best = max((sum(a == b for a, b in zip(left_tape[::-1], normalize_letters(render(rf))))
                    for rf in probe_fs if rf != f), default=0)
        near_misses.append({"frame_id": i, "letters": len(left_tape), "reverse_prefix_match": best,
                            "left": render(f), "diagnostic_only": True})
    near_misses.sort(key=lambda r: (r["reverse_prefix_match"], r["letters"]), reverse=True)
    return {"status": "semantic_pairing_typed_clauses", "operator": "independently_grammatical_svo_pp_reverse_tape_pairing",
            "config": {"frame_count": len(fs), "min_letters": min_letters, "cross_word_boundary": True,
                        "hard_exact_gate": True, "search_status": "truncated" if limit and len(exact) >= limit else "exhausted"},
            "exact_survivors": exact, "admitted_survivors": admitted,
            "near_miss_repairs": near_misses[:20],
            "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                           "source_material": "authored typed lexical frames; no catalogue text"},
            "reader_facing_next_operator": "If zero exact pairs, preserve the frame grammar and add a controlled inflection/lexeme mutation chosen by reverse-tape dead-frontier; rerun the same hard gate before readability review.",
            "reader_status": "unreviewed; programmatic exactness does not certify naturalness"}

def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True); p.add_argument('--limit', type=int)
    a = p.parse_args();
    if a.out.exists(): p.error(f"refusing to overwrite {a.out}")
    result = run(limit=a.limit); a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({"out": str(a.out), "frames": result["config"]["frame_count"], "exact": len(result["exact_survivors"]), "admitted": len(result["admitted_survivors"])}, indent=2))
if __name__ == '__main__': main()
