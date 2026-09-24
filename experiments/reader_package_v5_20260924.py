"""Create blinded readability comparisons; readers, not code, judge prose."""
from __future__ import annotations
import hashlib, json, random, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.incumbent_498_deep_clause_transducer_20261002 import audit
from experiments.reader_package_v4_20260919 import INTACT_CONTROLS
from server.v4 import BEST_KNOWN_TEXT, independent_audit

EXPERIMENT_ID = "reader-package-v5-20260924"
SEED = 20260924
QUESTION = ("Which reads more like ordinary connected English? Consider grammatical "
            "completeness, coherent meaning, and ease of understanding. Ignore length, "
            "punctuation, and how either passage was made. Choose A or B; use TIE or UNSURE if needed.")
REFS = (
    ("benchmark-38", "server/v4.py", "BEST_KNOWN_TEXT", 38, "ce71723a3eab38613adeb89c3ce18bab20286d91e6bcee20b25d3f4a724184c6"),
    ("outer-scene-650", "runs/incumbent-608-repeated-shell-repair-20261002.json", "double-event-shell-repair-650", 650, "2bd92686cbd01945be3869fbeee7ae9415cfcdae5cd616b26d89ec4f0acc54a9"),
    ("outer-scene-654", "runs/incumbent-650-outer-scene-growth-20260924.json", "outer-stop-spot-event-654", 654, "a190df41487ba2d1e1b55054690e4002c63f3a94123202b51aa4e502772b8acf"),
    ("central-scene-666", "runs/incumbent-666-central-mini-scene-comparison-20260922.json", "central-mini-scene-comparison-leon-noel-666", 666, "3951b9449ed3ab28f55d9798e344dfaf3123035f5f07c9dffdf0047bed0e1d79"),
)

def _letters(text):
    return re.sub(r"[^a-z]", "", text.casefold())

def _shuffle(text, seed):
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    random.Random(seed).shuffle(words)
    return " ".join(words) + "."

def build(seed=SEED):
    candidates, items, texts = [], {}, {}
    def add(i, text, condition, source, audit_record=None):
        texts[i] = text
        items[i] = {"item_id": i, "condition": condition, "source": source,
                    "letters": len(_letters(text)), "rendered": text, "audit": audit_record}
    for n, (cid, artifact, row_id, length, expected) in enumerate(REFS):
        if artifact == "server/v4.py":
            text = BEST_KNOWN_TEXT
        else:
            data = json.loads((ROOT / artifact).read_text())
            text = str(next(r for r in data["rows"] if r["id"] == row_id)["rendered"])
        tape = _letters(text)
        one, two = audit(text), independent_audit(text)
        sha = hashlib.sha256(tape.encode("ascii")).hexdigest()
        assert len(tape) == length and sha == expected
        assert one["two_pointer_exact"] and one["project_validator_exact"] and two["exact"]
        meta = {"id": cid, "text": text, "letters": length, "sha256": sha,
                "artifact": artifact, "source_row": row_id,
                "status": "short benchmark" if length == 38 else "unranked comparison candidate",
                "audit": {"two_pointer": one, "second_project_audit": two}}
        candidates.append(meta)
        add(cid, text, "exact_candidate", artifact, meta["audit"])
        add(f"shuffle-{cid}", _shuffle(text, seed+n), "word_shuffle", "seeded control")
    controls = []
    for n, text in enumerate(INTACT_CONTROLS):
        left, right = f"intact-{n}", f"shuffle-intact-{n}"
        add(left, str(text), "intact_prose_control", "pilot control")
        add(right, _shuffle(str(text), seed+100+n), "word_shuffle_control", "seeded control")
        controls.append((left, right))
    pairs = [(f"matched-{c['id']}", c["id"], f"shuffle-{c['id']}", "candidate_vs_shuffle") for c in candidates]
    pairs += [(f"control-{i}", a, b, "intact_vs_shuffle") for i, (a,b) in enumerate(controls)]
    for i, a in enumerate(candidates):
        for b in candidates[i+1:]:
            pairs.append((f"compare-{a['id']}-{b['id']}", a["id"], b["id"], "candidate_vs_candidate"))
    rng = random.Random(seed); rng.shuffle(pairs)
    form, key = [], []
    for index, (_label, left, right, kind) in enumerate(pairs):
        task = f"pair-{index:03d}"
        sides = [(left, texts[left]), (right, texts[right])]; rng.shuffle(sides)
        form.append({"task_id": task, "a": {"text": sides[0][1]}, "b": {"text": sides[1][1]}, "question": QUESTION})
        key.append({"task_id": task, "type": kind, "a_item_id": sides[0][0], "b_item_id": sides[1][0]})
    return {
        "experiment_id": EXPERIMENT_ID, "seed": seed, "status": "blinded_pilot_ready_human_ratings_pending",
        "rater_form": {"instructions": "Judge connected English only. Ignore length, palindrome status, punctuation, and source.", "items": form},
        "answer_key": {"items": [items[k] for k in sorted(items)], "tasks": key, "candidates": candidates},
        "reader_protocol": {"primary_outcome": "blinded human preference for grammatical, connected English",
            "length_is_not_a_quality_score": True, "intact_prose_and_shuffled_controls": True,
            "randomized_blinded_order": True, "target_independent_raters": 10,
            "human_ratings_collected": False, "programmatic_metrics_certify_readability": False,
            "dpo_status": "pilot only; wait for human labels and leakage-safe held-out data",
            "selection_rule": "Length is descriptive; rank by reader evidence, exactness audited separately."},
        "reproducibility": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "pair_count": len(form), "passage_count": len(items), "randomization": "fixed Python seed shuffles task order and A/B placement"},
    }

if __name__ == "__main__":
    result = build()
    out = ROOT / "runs" / f"{EXPERIMENT_ID}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({"pairs": len(result["rater_form"]["items"]), "output": str(out)}))
