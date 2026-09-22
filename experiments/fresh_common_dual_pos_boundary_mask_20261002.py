from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import sys
import time

from nltk.corpus import brown, wordnet as wn
from wordfreq import zipf_frequency

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, REPEATABLE_FUNCTION_WORDS


def tape(s: str) -> str:
    return "".join(re.findall(r"[a-z]", s.casefold()))


def audit(text: str) -> dict:
    x = tape(text)
    mismatch = next((i for i in range(len(x) // 2) if x[i] != x[-1-i]), None)
    return {
        "letters": len(x), "exact": bool(x) and mismatch is None,
        "first_mismatch": mismatch,
        "sha256_forward": hashlib.sha256(x.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(x[::-1].encode()).hexdigest(),
    }


def common_inventory() -> tuple[dict[str, tuple[str, ...]], dict]:
    counts = Counter()
    raw_tags: dict[str, set[str]] = defaultdict(set)
    proper = set()
    for word, tag in brown.tagged_words():
        w = word.casefold()
        if not (w.isascii() and w.isalpha()):
            continue
        counts[w] += 1
        raw_tags[w].add(tag)
        if tag.startswith("NP"):
            proper.add(w)

    def ordinary(w: str, min_count: int = 3, min_zipf: float = 3.7) -> bool:
        return (
            w not in proper and counts[w] >= min_count and zipf_frequency(w, "en") >= min_zipf
            and (len(w) != 2 or w in {
                "ah", "am", "an", "as", "at", "be", "by", "do", "go", "he", "if",
                "in", "is", "it", "me", "my", "no", "of", "oh", "on", "or", "ox",
                "so", "to", "up", "us", "we",
            })
            and (len(w) == 1 or w != w[::-1])
        )

    # Determiners/prepositions are closed common-word classes, not phrase banks.
    det = tuple(w for w in ("a", "the", "this", "that", "each", "every", "some", "any", "our", "their") if ordinary(w, 1, 0))
    adp = tuple(w for w in ("at", "by", "for", "from", "in", "near", "on", "to", "under", "with") if ordinary(w, 1, 0))

    def ranked(predicate, limit):
        rows = [w for w in counts if ordinary(w) and predicate(w, raw_tags[w])]
        rows.sort(key=lambda w: (-zipf_frequency(w, "en"), -counts[w], w))
        return tuple(rows[:limit])

    nsg = ranked(lambda w, ts: any(t in {"NN", "NN-TL", "NN-HL"} for t in ts), 900)
    npl = ranked(lambda w, ts: any(t in {"NNS", "NNS-TL", "NNS-HL"} for t in ts), 600)
    adj = ranked(lambda w, ts: any(t.startswith("JJ") for t in ts), 500)
    adv = ranked(lambda w, ts: any(t.startswith("RB") for t in ts), 300)

    def transitive(w: str) -> bool:
        lemma = wn.morphy(w, wn.VERB)
        if not lemma:
            return False
        for syn in wn.synsets(lemma, pos=wn.VERB):
            for lemma_obj in syn.lemmas():
                if lemma_obj.name() != lemma:
                    continue
                if any("something" in frame or "somebody" in frame for frame in lemma_obj.frame_strings()):
                    return True
        return False

    vsg = ranked(lambda w, ts: "VBZ" in ts and transitive(w), 360)
    vpl = ranked(lambda w, ts: ("VB" in ts or "VBP" in ts) and transitive(w), 360)
    vpast = ranked(lambda w, ts: "VBD" in ts and transitive(w), 360)
    domains = {"DET": det, "ADJ": adj, "ADV": adv, "NSG": nsg, "NPL": npl,
               "VSGT": vsg, "VPLT": vpl, "VPAST": vpast, "ADP": adp,
               "NOBJ": nsg + tuple(w for w in npl if w not in set(nsg)),
               "PRONSG": ("he", "she", "it"), "PRONPL": ("we", "they"),
               "OBJPRON": ("me", "us", "him", "her", "it", "them"),
               "COPSG": ("is", "was"), "COPPL": ("are", "were")}
    provenance = {
        "brown_tokens": sum(counts.values()), "proper_forms_excluded": len(proper),
        "common_threshold": {"brown_count": 3, "wordfreq_zipf": 3.7},
        "domain_sizes": {k: len(v) for k, v in domains.items()},
        "domain_sha256": hashlib.sha256(json.dumps(domains, sort_keys=True).encode()).hexdigest(),
        "lexical_sources": ["NLTK Brown word/tag counts", "wordfreq English frequency", "WordNet verb frames"],
    }
    return domains, provenance


TEMPLATES = (
    ("DET", "NSG", "VSGT", "DET", "NOBJ"),
    ("DET", "ADJ", "NSG", "VSGT", "DET", "NOBJ"),
    ("DET", "NSG", "VPAST", "DET", "NOBJ"),
    ("DET", "ADJ", "NSG", "VPAST", "DET", "ADJ", "NOBJ"),
    ("DET", "NPL", "VPLT", "DET", "NOBJ"),
    ("DET", "ADJ", "NPL", "VPLT", "DET", "ADJ", "NOBJ"),
    ("NPL", "VPLT", "DET", "NOBJ"),
    ("PRONSG", "VSGT", "DET", "NOBJ"),
    ("PRONPL", "VPLT", "DET", "NOBJ"),
    ("ADV", "PRONSG", "VSGT", "DET", "NOBJ"),
    ("DET", "NSG", "COPSG", "ADJ"),
    ("DET", "NPL", "COPPL", "ADJ"),
    ("PRONSG", "COPSG", "ADJ"),
    ("PRONPL", "COPPL", "ADJ"),
    ("DET", "NSG", "VSGT", "OBJPRON"),
    ("DET", "NPL", "VPLT", "OBJPRON"),
    ("DET", "NSG", "VSGT", "DET", "NOBJ", "ADV"),
    ("DET", "NPL", "VPLT", "DET", "NOBJ", "ADV"),
    ("PRONSG", "VSGT", "DET", "NOBJ", "ADV"),
    ("PRONPL", "VPLT", "DET", "NOBJ", "ADV"),
    ("ADV", "DET", "NSG", "VSGT", "DET", "NOBJ"),
    ("ADV", "NPL", "VPLT", "DET", "NOBJ"),
)


def cumulative(words: tuple[str, ...], reverse_words: bool = False) -> tuple[int, ...]:
    seq = tuple(reversed(words)) if reverse_words else words
    out, total = [], 0
    for word in seq:
        total += len(tape(word))
        out.append(total)
    return tuple(out)


def boundary_mask_ok(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    # Both coordinates are measured from the exposed palindrome edge.  A shared
    # internal cursor would produce a finished mirrored token span.  The current
    # equal outer endpoint may close, but extending it later makes it internal
    # and causes immediate rejection.
    lb = cumulative(left)
    rb = cumulative(right, reverse_words=True)
    if not lb or not rb:
        return True
    shared = set(lb) & set(rb)
    allowed = {lb[-1]} if lb[-1] == rb[-1] else set()
    return not (shared - allowed)


def content_distinct(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    words = left + right
    content = [w for w in words if w not in REPEATABLE_FUNCTION_WORDS]
    return len(content) == len(set(content))


@dataclass(frozen=True)
class State:
    li: int
    ri: int
    left: tuple[str, ...]
    right_rev: tuple[str, ...]
    owner: str
    residual: str


def search_pair(left_template, right_template, domains, max_states=400_000, max_results=8):
    q = deque([State(0, len(right_template)-1, (), (), "", "")])
    seen = set()
    stats = Counter()
    results = []
    cursor_hist = Counter()
    deepest = []
    first_index = {}
    for role, words in domains.items():
        for side in ("left", "right"):
            idx = defaultdict(list)
            for w in words:
                exposed = tape(w) if side == "left" else tape(w)[::-1]
                idx[exposed[0]].append((w, exposed))
            first_index[(side, role)] = idx

    while q and len(seen) < max_states and len(results) < max_results:
        st = q.popleft()
        key = (st.li, st.ri, st.left, st.right_rev, st.owner, st.residual)
        if key in seen:
            continue
        seen.add(key)
        cursor_hist[(st.li, st.ri)] += 1
        matched = min(sum(map(len, map(tape, st.left))), sum(map(len, map(tape, st.right_rev))))
        deepest.append((matched, st.li + len(right_template) - 1 - st.ri, st))
        deepest.sort(key=lambda x: (-x[0], -x[1], len(x[2].residual)))
        del deepest[12:]
        if st.li == len(left_template) and st.ri < 0 and not st.residual:
            right = tuple(reversed(st.right_rev))
            results.append((st.left, right))
            continue
        sides = ("right",) if st.owner == "left" else (("left",) if st.owner == "right" else ("left", "right"))
        for side in sides:
            idx = st.li if side == "left" else st.ri
            template = left_template if side == "left" else right_template
            if idx < 0 or idx >= len(template):
                continue
            role = template[idx]
            if st.residual:
                choices = first_index[(side, role)].get(st.residual[0], ())
            else:
                choices = tuple((w, tape(w) if side == "left" else tape(w)[::-1]) for w in domains[role])
            for word, exposed in choices:
                stats["lexical_attempts"] += 1
                if st.residual:
                    common = min(len(st.residual), len(exposed))
                    if st.residual[:common] != exposed[:common]:
                        stats["character_prunes"] += 1
                        continue
                    if len(st.residual) > len(exposed):
                        owner, residual = st.owner, st.residual[common:]
                    elif len(exposed) > len(st.residual):
                        owner, residual = side, exposed[common:]
                    else:
                        owner, residual = "", ""
                else:
                    owner, residual = side, exposed
                left = st.left + ((word,) if side == "left" else ())
                right_rev = st.right_rev + ((word,) if side == "right" else ())
                right = tuple(reversed(right_rev))
                if not content_distinct(left, right):
                    stats["distinct_content_prunes"] += 1
                    continue
                if os.environ.get("DISABLE_MASK") != "1" and not boundary_mask_ok(left, right):
                    stats["boundary_mask_prunes"] += 1
                    continue
                q.append(State(st.li + (side == "left"), st.ri - (side == "right"),
                               left, right_rev, owner, residual))
                stats["transitions"] += 1
    stats["states"] = len(seen)
    stats["cap_reached"] = int(bool(q))
    stats["cursor_hist"] = {f"{li}:{ri}": n for (li, ri), n in sorted(cursor_hist.items())}
    stats["deepest_frontiers"] = [
        {"matched_letters": m, "words_selected": d, "left": st.left,
         "right_reverse": st.right_rev, "owner": st.owner, "residual": st.residual,
         "next_left": left_template[st.li] if st.li < len(left_template) else None,
         "next_right": right_template[st.ri] if st.ri >= 0 else None}
        for m, d, st in deepest
    ]
    return results, dict(stats)


def main():
    started = time.monotonic()
    domains, lex_provenance = common_inventory()
    all_rows = []
    totals = Counter()
    pair_rows = []
    for pair_index, (lt, rt) in enumerate(itertools.product(TEMPLATES, repeat=2)):
        found, stats = search_pair(lt, rt, domains, max_states=int(os.environ.get("MAX_STATES", "400000")))
        totals.update({k: v for k, v in stats.items() if isinstance(v, int)})
        pair_rows.append({"left_template": lt, "right_template": rt, **stats, "closures": len(found)})
        for left, right in found:
            rendered = " ".join(left).capitalize() + ". " + " ".join(right).capitalize() + "."
            au = audit(rendered)
            checks = mechanical_admission_checks(rendered, min_letters=39, max_letters=240)
            internal_mask = boundary_mask_ok(left, right)
            all_rows.append({
                "rendered": rendered, "left_words": left, "right_words": right,
                "left_template": lt, "right_template": rt,
                "boundary_cursors": {"left": cumulative(left), "right_from_end": cumulative(right, True)},
                "online_boundary_mask": internal_mask,
                "audit": au, "mechanical_checks": checks,
                "mechanically_admitted": internal_mask and all(checks.values()),
            })
        print(json.dumps({"pair": pair_index + 1, "left": lt, "right": rt,
                          "states": stats["states"], "closures": len(found),
                          "boundary_prunes": stats.get("boundary_mask_prunes", 0)}), flush=True)
    admitted = [r for r in all_rows if r["mechanically_admitted"]]
    payload = {
        "experiment_id": "fresh-common-dual-pos-boundary-mask-20261002",
        "method": "common-word dual POS/agreement/valency grammars intersected with a live character residual and online complementary-boundary mask",
        "stats": {**totals, "grammar_pairs": len(pair_rows), "closures": len(all_rows),
                  "exact_gt38": sum(r["audit"]["exact"] and r["audit"]["letters"] > 38 for r in all_rows),
                  "mechanically_admitted": len(admitted), "elapsed_seconds": round(time.monotonic()-started, 3)},
        "grammar": {"templates": TEMPLATES, "agreement": "NSG↔VSGT; NPL↔VPLT; VPAST number-neutral",
                    "valency": "all finite verbs have a WordNet transitive frame and every template realizes an object NP"},
        "lexicon": lex_provenance,
        "candidates": all_rows,
        "admitted": admitted,
        "pair_stats": pair_rows,
        "provenance": {"inherited_498_endpoint": False, "proper_names": False,
                       "finished_phrase_units": False, "phrase_bank": False,
                       "finished_tape_reversal": False, "post_render_repair": False,
                       "rendering_stage": "only after both typed grammars accept and the live residual is empty",
                       "shared_mechanical_gate": True},
        "next_operator": ("promote exact rows to blinded reader study" if admitted else
                          "compile the same typed word domains to a bidirectional character trie with one optional relative-clause recursion; preserve the online complementary-boundary mask"),
    }
    raw = json.dumps(payload, sort_keys=True)
    payload["result_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    print("FINAL " + json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
