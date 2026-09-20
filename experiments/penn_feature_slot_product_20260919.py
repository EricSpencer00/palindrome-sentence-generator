"""Agreement-conditioned slot product using original Brown Penn tags.

Unlike the compact Brown-POS run, each sentence template is instantiated from
one tagged subject/verb feature bank (singular+VBZ, plural+VBP, or past+VBD).
The existing slot-pair search still enforces mirrored characters online across
word boundaries; this file only constructs the feature-conditioned inventories.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.slot_pair_character_search_20260919 import Slot, search


def _base(tag: str) -> str:
    return tag.split("-", 1)[0]


def extract_frames(limit: int = 50_000):
    from nltk.corpus import brown

    subjects: dict[str, Counter[str]] = defaultdict(Counter)
    verbs: dict[str, Counter[str]] = defaultdict(Counter)
    objects: dict[str, Counter[str]] = defaultdict(Counter)
    frames = Counter()
    adjuncts = Counter()
    feature_adjuncts = defaultdict(Counter)
    frame_ids = defaultdict(Counter)
    complements = defaultdict(Counter)
    for sentence in brown.tagged_sents()[:limit]:
        for i in range(len(sentence) - 4):
            (det, det_tag), (subj, subj_tag), (verb, verb_tag), (obj_det, obj_det_tag), (obj, obj_tag) = sentence[i:i + 5]
            subj_base, verb_base = _base(subj_tag), _base(verb_tag)
            if not det_tag.startswith(("AT", "DT")) or not obj_det_tag.startswith(("AT", "DT")):
                continue
            if subj_base in {"NN", "NP"} and verb_base == "VBZ":
                feature = "singular_vbz"
            elif subj_base in {"NNS", "NPS"} and verb_base in {"VBP", "VB"}:
                feature = "plural_vbp"
            elif verb_base == "VBD" and subj_base in {"NN", "NP", "NNS", "NPS"}:
                feature = "past_vbd"
            else:
                continue
            if not all(word.isalpha() for word in (det, subj, verb, obj_det, obj)):
                continue
            frames[feature] += 1
            frame_id=f'{det.casefold()} {subj.casefold()} {verb.casefold()} {obj_det.casefold()} {obj.casefold()}'
            frame_ids[feature][frame_id] += 1
            subjects[feature][subj.casefold()] += 1
            verbs[feature][verb.casefold()] += 1
            objects[feature][obj.casefold()] += 1
            if i + 6 < len(sentence):
                prep, ptag = sentence[i+5]; noun, ntag = sentence[i+6]
                if ptag == 'IN' and ntag.startswith(('NN','NP')) and prep.isalpha() and noun.isalpha():
                    phrase=f'{prep.casefold()} {noun.casefold()}'; adjuncts[phrase] += 1; feature_adjuncts[feature][phrase] += 1
                    frame_ids[feature][frame_id+' | '+phrase] += 1
                    if i + 8 < len(sentence):
                        p2,t2=sentence[i+7]; n2,u2=sentence[i+8]
                        if t2 == 'IN' and u2.startswith(('NN','NP')) and p2.isalpha() and n2.isalpha(): complements[feature][f'{p2.casefold()} {n2.casefold()}'] += 1
    return subjects, verbs, objects, dict(frames), adjuncts, feature_adjuncts, frame_ids, complements


def top(counter: Counter[str], limit: int = 64) -> tuple[str, ...]:
    return tuple(word for word, _ in counter.most_common(limit))


def run() -> dict[str, object]:
    subjects, verbs, objects, frame_counts, adjunct_counts, feature_adjuncts, frame_ids, complements = extract_frames()
    det = ("a", "the", "this", "that", "one")
    results = []
    feature_stats = {}
    for feature in ("singular_vbz", "plural_vbp", "past_vbd"):
        subject_words = top(subjects[feature])
        verb_words = top(verbs[feature])
        object_words = top(objects[feature])
        adjunct = tuple(word for word, _ in feature_adjuncts[feature].most_common(32)) or ("near town",)
        complement = tuple(word for word, _ in complements[feature].most_common(16)) or ("by town",)
        if not subject_words or not verb_words:
            feature_stats[feature] = {"subject_words": 0, "verb_words": 0, "states": 0, "pruned": 0, "exact": 0}
            continue
        subject_slot = Slot("subject", subject_words, word_features=tuple((w, feature) for w in subject_words))
        verb_slot = Slot("verb", verb_words, word_features=tuple((w, feature) for w in verb_words))
        object_slot = Slot("object", object_words, word_features=tuple((w, feature) for w in object_words))
        templates = [
            (Slot("det", det), subject_slot, verb_slot, Slot("det", det), object_slot),
            (Slot("det", det), subject_slot, verb_slot, Slot("det", det), object_slot, Slot("prep_object", adjunct)),
            (Slot("det", det), subject_slot, verb_slot, Slot("det", det), object_slot, Slot("prep_object", adjunct), Slot("complement", complement)),
        ]
        runs = [search(template, limit=16) for template in templates]
        stats = {
            "subject_words": len(subject_words), "verb_words": len(verb_words), "object_words": len(object_words),
            "states": sum(run["stats"]["states"] for run in runs),
            "pruned": sum(run["stats"]["pruned"] for run in runs),
            "agreement_pruned": sum(run["stats"]["agreement_pruned"] for run in runs),
            "exact": sum(run["stats"]["exact"] for run in runs),
        }
        feature_stats[feature] = stats
        results.extend(candidate for run_result in runs for candidate in run_result["candidates"])
    return {
        "experiment_id": "penn-feature-slot-product-20260919",
        "method": "Penn-feature-conditioned single-sentence slot product with live cross-word character obligations",
        "candidates": results,
        "frame_counts": frame_counts,
        "adjunct_frame_counts": dict(adjunct_counts.most_common(32)),
        "adjunct_frame_unique": len(adjunct_counts),
        "feature_adjunct_frame_counts": {k: dict(v.most_common(32)) for k, v in feature_adjuncts.items()},
        "frame_identity_counts": {k: dict(v.most_common(32)) for k, v in frame_ids.items()},
        "feature_complement_counts": {k: dict(v.most_common(16)) for k, v in complements.items()},
        "feature_stats": feature_stats,
        "provenance": {
            "penn_tags": True,
            "word_feature_maps": True,
            "paired_clauses": False,
            "aligned_token_mirror": False,
            "finished_tape_reversal": False,
            "fallback": False,
            "adjunct_partition_audit": "frame-attested DET-NOUN-VERB-ADP-NOUN",
            "human_readability_evidence": False,
        },
    }


if __name__ == "__main__":
    result = run()
    Path("runs/penn-feature-slot-product-20260919.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["feature_stats"]))
