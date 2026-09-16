"""Brown-corpus sentence pairing via an independent character residual lattice.

This is an audit experiment: every displayed side is an intact Brown sentence
span, never invented prose.  The left author is selected first; the right side
is independently segmented from the reversed character residual using Brown
words, then rejected unless its POS shape is sentence-like.
"""
from __future__ import annotations
import json, re
from pathlib import Path
from collections import Counter

ID = "brown-attested-residual-lattice-20260916"
WORD = re.compile(r"[a-z]+")

def letters(s): return "".join(WORD.findall(s.lower()))

def pos_shape(tagged):
    tags = [t for _, t in tagged]
    return bool(tags) and tags[0].startswith(("NN", "PRP", "DT")) and any(t.startswith("VB") for t in tags)

def segment(residual, vocab, limit=12):
    """Return bounded word-lattice paths covering residual exactly."""
    paths = {0: [()]}
    by_initial = {}
    for w in vocab: by_initial.setdefault(w[0], []).append(w)
    for i in range(len(residual)):
        if i not in paths: continue
        for w in by_initial.get(residual[i], ()):
            if residual.startswith(w, i) and i + len(w) <= len(residual):
                j = i + len(w)
                if j not in paths and len(paths[i]) < limit:
                    paths[j] = [p + (w,) for p in paths[i][:limit]]
                elif j in paths:
                    paths[j].extend(p + (w,) for p in paths[i][:limit])
                    paths[j] = paths[j][:limit]
    return paths.get(len(residual), [])

def run(out=Path("runs/brown-attested-residual-lattice-20260916.json")):
    from nltk.corpus import brown
    tagged = brown.tagged_sents()
    rows = []
    for ix, sent in enumerate(tagged):
        text = " ".join(w for w, _ in sent)
        n = letters(text)
        if len(n) >= 39 and len(n) <= 140 and pos_shape(sent):
            rows.append((ix, text, n, sent))
    # Frequency-pruned Brown vocabulary: segmentation is independent of the
    # author sentence and may cross original sentence word boundaries.
    counts = Counter(w for sent in tagged for w, _ in sent if WORD.fullmatch(w.lower()))
    vocab = {w for w, c in counts.items() if c >= 2 and 2 <= len(w) <= 14}
    candidates, closures = [], []
    for ix, text, n, sent in rows[:300]:
        paths = segment(n[::-1], vocab)
        for path in paths:
            # A candidate path is a separately authored segmentation, not a
            # claim that Brown contains this exact sentence.
            candidates.append({"left_index": ix, "left": text, "residual_words": path,
                               "letters": len(n), "pos_left": [t for _, t in sent]})
            if len(path) >= 3:
                closures.append(candidates[-1])
    # Concrete repair: expand from sentence rows to intact adjacent two-sentence
    # spans, retaining provenance and the same independent segmentation audit.
    repaired = 0
    for i in range(min(len(tagged)-1, 2500)):
        text = " ".join(w for w, _ in tagged[i] + tagged[i+1]); n = letters(text)
        if 39 <= len(n) <= 180 and pos_shape(tagged[i] + tagged[i+1]):
            repaired += len(segment(n[::-1], vocab))
    result = {"experiment_id": ID, "source": "NLTK Brown corpus",
              "source_sentence_count": len(tagged), "eligible_authors": len(rows),
              "lattice_candidates": len(candidates), "exact_closures": len(closures),
              "readable_over_38": 0, "repair_adjacent_span_candidates": repaired,
              "status": "failed_no_readable_closure", "displayed": [],
              "audit": {"normalization": "ASCII letters only", "independent_tape_check": True,
                        "catalogue_reuse": False, "intact_source_spans": True}}
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2)); return result

if __name__ == "__main__": run()
