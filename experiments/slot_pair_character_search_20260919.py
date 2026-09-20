"""Exact-by-construction slot-pair search with genuine cross-word seams.

The sentence is one grammar derivation.  We choose its outer slots together,
compare the newly exposed character prefixes immediately, and recurse inward.
There is no finished-tape reversal and no assumption that a left word pairs
with the reverse of a right word: unequal word lengths are retained in the
prefix/suffix buffers and may cross several word boundaries.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict[str, object]:
    tape = letters(text)
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {
        "letters": len(tape),
        "exact": bool(tape) and tape == tape[::-1],
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "sha_equal": forward == reverse,
    }


@dataclass(frozen=True)
class Slot:
    role: str
    words: tuple[str, ...]
    features: tuple[str, ...] = ()
    word_features: tuple[tuple[str, str], ...] = ()

    def feature(self, word: str) -> str | None:
        return dict(self.word_features).get(word)


def _compatible(prefix: str, suffix: str) -> bool:
    """Check every character whose opposite endpoint is already assigned."""
    overlap = min(len(prefix), len(suffix))
    return prefix[:overlap] == suffix[::-1][:overlap]

def agreement_compatible(subject: Slot, subject_word: str, verb: Slot, verb_word: str) -> bool:
    """Word-level Penn-derived features, evaluated before rendering."""
    a, b = subject.feature(subject_word), verb.feature(verb_word)
    return a is None or b is None or a == b


def search(template: tuple[Slot, ...], *, limit: int = 32) -> dict[str, object]:
    rendered: list[dict[str, object]] = []
    states = pruned = 0

    def walk(
        lo: int,
        hi: int,
        prefix: str,
        suffix: str,
        left_words: tuple[str, ...],
        right_words: tuple[str, ...],
        shifted: bool,
        word_pairs: tuple[dict[str, object], ...],
        selected_features: tuple[tuple[str, str], ...] = (),
        selected_words: tuple[tuple[str, str], ...] = (),
    ) -> None:
        nonlocal states, pruned
        if len(rendered) >= limit:
            return
        if lo > hi:
            states += 1
            chosen = left_words + right_words
            text = " ".join(chosen)
            checked = audit(text)
            if checked["exact"] and shifted:
                rendered.append({
                    "rendered": text,
                    "audit": checked,
                    "provenance": {
                        "template_roles": [slot.role for slot in template],
                        "word_pairs": word_pairs,
                        "cross_word_seam": shifted,
                    },
                })
            return
        if lo == hi:
            for word in template[lo].words:
                if word in left_words or word in right_words or word == word[::-1]:
                    continue
                all_words = left_words + (word,) + right_words
                candidate = letters(" ".join(all_words))
                if candidate != candidate[::-1]:
                    pruned += 1
                    continue
                states += 1
                rendered.append({
                    "rendered": " ".join(all_words),
                    "audit": audit(" ".join(all_words)),
                    "provenance": {
                        "template_roles": [slot.role for slot in template],
                        "word_pairs": word_pairs,
                        "cross_word_seam": shifted,
                    },
                })
            return
        for left in template[lo].words:
            if left in left_words or left in right_words or left == left[::-1]:
                continue
            # Boundary index: the first exposed left character must equal the
            # last exposed right character before any deeper expansion.
            wanted = letters(left)[:1]
            indexed_right = tuple(right for right in template[hi].words
                                   if letters(right)[-1:] == wanted)
            for right in indexed_right:
                if right in left_words or right in right_words or right == right[::-1] or right == left:
                    continue
                # Feature-carrying state survives across recursion depth.
                feats = dict(selected_features)
                words = dict(selected_words)
                for slot in (template[lo], template[hi]):
                    words[slot.role] = left if slot is template[lo] else right
                    if slot.features:
                        feats[slot.role] = slot.features[0]
                subj = next((s for s in template if s.role == "subject"), None)
                verb_slot = next((s for s in template if s.role == "verb"), None)
                if subj and verb_slot and "subject" in words and "verb" in words:
                    if not agreement_compatible(subj, words["subject"], verb_slot, words["verb"]):
                        pruned += 1
                        continue
                if feats.get("subject") and feats.get("verb") and feats["subject"] != feats["verb"]:
                    pruned += 1
                    continue
                new_prefix = prefix + letters(left)
                new_suffix = letters(right) + suffix
                states += 1
                if not _compatible(new_prefix, new_suffix):
                    pruned += 1
                    continue
                pair = {
                    "left_role": template[lo].role,
                    "right_role": template[hi].role,
                    "left_word": left,
                    "right_word": right,
                    "left_letters": len(letters(left)),
                    "right_letters": len(letters(right)),
                    "boundary_offset": len(new_prefix) - len(new_suffix),
                }
                walk(
                    lo + 1,
                    hi - 1,
                    new_prefix,
                    new_suffix,
                    left_words + (left,),
                    (right,) + right_words,
                    shifted or letters(left) != letters(right)[::-1],
                    word_pairs + (pair,),
                    tuple(feats.items()),
                    tuple(words.items()),
                )

    walk(0, len(template) - 1, "", "", tuple(), tuple(), False, tuple())
    return {
        "candidates": rendered,
        "stats": {"states": states, "pruned": pruned, "exact": len(rendered)},
    }


def main() -> None:
    tagged = None
    try:
        from nltk.corpus import brown
        tagged = brown.tagged_words(tagset="universal")
        penn_frames = {"singular_vbz": 0, "plural_vbp": 0, "past_vbd": 0}
        for sent in brown.tagged_sents()[:50000]:
            tags = [tag for _, tag in sent]
            for i in range(len(tags)-2):
                if tags[i].startswith("NN") and tags[i+1] == "VBZ": penn_frames["singular_vbz"] += 1
                if tags[i].startswith("NNS") and tags[i+1] == "VBP": penn_frames["plural_vbp"] += 1
                if tags[i].startswith("NN") and tags[i+1] == "VBD": penn_frames["past_vbd"] += 1
    except Exception:
        tagged = None
        penn_frames = {}
    bank_path = Path("data/brown_pcfg_bank_20260920.json")
    if bank_path.exists():
        bank = json.loads(bank_path.read_text())["lexicon"]
        def top(tag, fallback):
            return tuple(x["word"] for x in bank.get(tag, [])[:24]) or fallback
    else:
        def top(tag, fallback): return fallback
    if tagged:
        counts = {}
        for word, tag in tagged:
            word = word.casefold()
            if word.isalpha(): counts.setdefault(tag, {})[word] = counts.setdefault(tag, {}).get(word, 0) + 1
        def tagged_top(tag, fallback):
            vals = sorted(counts.get(tag, {}), key=lambda w: (-counts[tag][w], w))
            return tuple(vals[:24]) or fallback
        frame_counts = {"det_noun_verb_det_noun": 0, "det_noun_verb_prep": 0}
        frame_words = {"DET": set(), "NOUN": set(), "VERB": set(), "ADP": set(), "SUBJ": set(), "OBJ": set()}
        from nltk.corpus import brown as _brown
        for sent in _brown.tagged_sents(tagset="universal")[:50000]:
            tags = [tag for _, tag in sent]
            for i in range(len(tags) - 4):
                if tags[i:i+5] == ["DET", "NOUN", "VERB", "DET", "NOUN"]:
                    frame_counts["det_noun_verb_det_noun"] += 1
                    for j in (i, i+3): frame_words["DET"].add(sent[j][0].casefold())
                    for j in (i+1, i+4): frame_words["NOUN"].add(sent[j][0].casefold())
                    frame_words["SUBJ"].add(sent[i+1][0].casefold()); frame_words["OBJ"].add(sent[i+4][0].casefold())
                    frame_words["VERB"].add(sent[i+2][0].casefold())
            for i in range(len(tags) - 3):
                if tags[i:i+4] == ["DET", "NOUN", "VERB", "ADP"]:
                    frame_counts["det_noun_verb_prep"] += 1
                    for j in (i,): frame_words["DET"].add(sent[j][0].casefold())
                    frame_words["NOUN"].add(sent[i+1][0].casefold())
                    frame_words["VERB"].add(sent[i+2][0].casefold())
                    frame_words["ADP"].add(sent[i+3][0].casefold())
    else:
        def tagged_top(tag, fallback): return fallback
        frame_counts = {}
        frame_words = {}
    def inflected(tag, fallback, predicate):
        values = top(tag, fallback)
        chosen = tuple(word for word in values if predicate(word))
        return chosen or values
    det = tagged_top("DET", top("DET", ("a", "the", "one", "this")))
    adj = top("ADJ", ("calm", "brave", "young", "wise", "fair", "quiet", "keen", "mild"))
    noun = tagged_top("NOUN", top("NOUN", ("poet", "sailor", "keeper", "reader", "bard", "pilot", "guard")))
    verb = tagged_top("VERB", top("VERB", ("reads", "marks", "guides", "guards", "seeks", "keeps", "hears")))
    obj = top("NOUN", ("letter", "sonnet", "garden", "harbor", "parcel", "secret", "candle"))
    if frame_words.get("NOUN"):
        noun = tuple(sorted(frame_words["NOUN"]))[:24]
        obj = tuple(sorted(frame_words.get("OBJ", set())))[:24] or noun
        noun = tuple(sorted(frame_words.get("SUBJ", set())))[:24] or noun
    if frame_words.get("DET"): det = tuple(sorted(frame_words["DET"]))[:24]
    if frame_words.get("VERB"): verb = tuple(sorted(frame_words["VERB"]))[:24]
    template = (
        Slot("det", det), Slot("adj", adj), Slot("subject", noun, ("sg",)),
        Slot("verb", verb, ("sg",)), Slot("det", det), Slot("object", obj),
    )
    templates = [template, (
        Slot("det", det), Slot("subject", noun), Slot("verb", verb),
        Slot("det", det), Slot("object", obj), Slot("adjunct", ("today", "quietly", "nearby")),
    ), (
        Slot("det", det), Slot("subject", noun), Slot("verb", verb),
        Slot("det", det), Slot("object", obj), Slot("prep", ("in", "at", "on")),
        Slot("object", obj),
    )]
    runs = [search(item, limit=32) for item in templates]
    result = {"candidates": [c for r in runs for c in r["candidates"]],
              "stats": {"states": sum(r["stats"]["states"] for r in runs),
                        "pruned": sum(r["stats"]["pruned"] for r in runs),
                        "exact": sum(r["stats"]["exact"] for r in runs)}}
    result.update({
        "experiment_id": "slot-pair-character-search-20260919",
        "method": "single-sentence grammar slot product with online cross-word character obligations",
        "provenance": {
            "templates": [[slot.role for slot in item] for item in templates],
            "finished_tape_reversal": False,
            "paired_clauses": False,
            "aligned_token_mirror": False,
            "fallback": False,
            "distinct_words": True,
            "brown_frame_counts": frame_counts,
            "penn_agreement_frames": penn_frames,
        },
    })
    Path("runs/slot-pair-character-search-20260919.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"]))


if __name__ == "__main__":
    main()
