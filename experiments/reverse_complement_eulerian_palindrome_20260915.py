"""Bounded reverse-complement Eulerian overlap-graph construction.

The graph is character-context based, not a word or phrase graph.  A directed
edge is a corpus-observed k-character context shift ``u -> v``.  Only edges
whose reverse-complement shift is also observed are eligible.  A self-avoiding
Euler trail is then mirrored through those edge partners; word boundaries are
recovered *after* the character trail and never drive the search.

This is a construction experiment.  Exactness is independently audited, while
segmentation and a conservative POS/dependency proxy are rejection filters;
neither is a human readability certificate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"
ID = "reverse-complement-eulerian"
SIG = "character-context-overlap-graph|reverse-complement-edge-balance|edge-disjoint-eulerian-trail|dependency-parse-after-trail|independent-tape-audit|structural-trail-repair"
SEED = "An aide rips nine memos; some men inspire Diana."
FUNCTION = set("a an the of to in on at for and or as is was are be by with from it i he she we they this that not but".split())


def norm(text: str) -> str:
    return "".join(c.lower() for c in text if "a" <= c.lower() <= "z")


def audit(text: str) -> dict:
    tape = norm(text)
    bad = [{"left": i, "right": len(tape)-1-i, "left_char": tape[i],
            "right_char": tape[-1-i]}
           for i in range(len(tape)//2) if tape[i] != tape[-1-i]]
    return {"exact": bool(tape) and not bad, "letters": len(tape),
            "comparisons": len(tape)//2, "mismatch_count": len(bad),
            "first_mismatches": bad[:8]}


def prior_snapshot() -> dict:
    rows = json.loads(REGISTRY.read_text())["entries"]
    if any(r["id"] == ID or r["signature"] == SIG for r in rows):
        raise RuntimeError("self-registration must occur after preflight")
    return {"count": len(rows), "ids": sorted(r["id"] for r in rows),
            "signatures": sorted(r["signature"] for r in rows)}


def corpus() -> tuple[list[str], set[str]]:
    data = json.loads((ROOT / "data/ngrams_wikitext2.json").read_text())
    source = list((ROOT / "data/authored_sentences.txt").read_text().splitlines())
    for key in ("sent6", "sent8", "sent10"):
        source.extend(data.get(key, []))
    clean = [norm(s) for s in source if len(norm(s)) >= 8]
    return clean, {" ".join(re.findall(r"[a-z]+", s.lower())) for s in source}


def edges(texts: list[str], k: int, heldout: bool = False):
    """Mine unique character-context shifts and their provenance counts."""
    counts: Counter[tuple[str, str]] = Counter()
    origins: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
    for si, text in enumerate(texts):
        for i in range(len(text) - k):
            key = (text[i:i+k], text[i+1:i+k+1])
            counts[key] += 1
            if len(origins[key]) < 4:
                origins[key].append(si)
    # The repair deliberately changes only which observed context order is
    # admitted; it never invents a lexical edge.
    eligible = {e for e in counts if (e[1][::-1], e[0][::-1]) in counts}
    if heldout:
        eligible = {e for e in eligible if counts[e] == 1}
    return counts, origins, eligible


def trails(counts, origins, eligible, k: int, max_states: int = 100_000):
    adj: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    for u, v in eligible:
        adj[u].append((v, (u, v)))
    for u in adj:
        adj[u].sort(key=lambda x: (x[0], x[1]))
    out = []
    states = 0
    # A trail is emitted as its initial context followed by each edge's new
    # character.  The mirrored edge trail is checked separately below.
    def walk(start, node, used, chars, path):
        nonlocal states
        if states >= max_states:
            return
        states += 1
        if 20 <= len(chars) <= 90:
            mirror = [(v[::-1], u[::-1]) for u, v in reversed(path)]
            balanced = (len(set(path + mirror)) == 2 * len(path)
                        and all(e in eligible for e in mirror))
            # Keep unbalanced trails as explicit near-miss probes.  They are
            # never promoted, but retaining their rendered palindromic tapes
            # makes the failed graph constraint inspectable rather than an
            # empty result.  Balanced rows carry the only reader-eligibility
            # path through the later hard gate.
            out.append({"start": start, "chars": chars,
                        "left_edges": path, "right_edges": mirror,
                        "edge_count": len(path) + len(mirror),
                        "context_k": k, "balanced_reverse_complement": balanced,
                        "source_edge_counts": [counts[e] for e in path],
                        "source_edge_origins": [origins[e] for e in path]})
            if len(out) >= 160:
                return
        if len(chars) >= 90:
            return
        for nxt, edge in adj.get(node, []):
            if edge in used:
                continue
            walk(start, nxt, used | {edge}, chars + nxt[-1], path + [edge])
            if len(out) >= 160 or states >= max_states:
                return
    for start in sorted(adj):
        walk(start, start, set(), start, [])
        if len(out) >= 160 or states >= max_states:
            break
    return out, states, {"nodes": len(adj), "edges": len(eligible)}


def segment(tape: str) -> list[str]:
    from llm_palindrome.respace import respace_k
    from llm_palindrome.lexicon import load_lexicon
    vocab = set(load_lexicon(str(ROOT / "data/lexicon.txt")))
    readings = respace_k(tape, vocab, k=4)
    return readings[0] if readings else []


def parse_proxy(ws: list[str]) -> dict:
    """Conservative structural diagnostic; never certifies readability."""
    try:
        import nltk
        tags = [t for _, t in nltk.pos_tag(ws)]
    except Exception:
        tags = []
    finite = {"VB", "VBD", "VBP", "VBZ", "MD", "BE", "BED", "BEDZ",
              "BEN", "BER", "BEZ", "DO", "DOD", "DOZ", "HV", "HVD", "HVZ"}
    vi = next((i for i, t in enumerate(tags) if t.split("-")[0] in finite), None)
    subject = bool(vi is not None and any(t.startswith(("NN", "PRP")) for t in tags[:vi]))
    complement = bool(vi is not None and len(tags) > vi + 1)
    return {"status": "diagnostic_only", "tag_count": len(tags),
            "has_finite_verb": vi is not None, "has_subject_before_verb": subject,
            "has_complement_after_verb": complement,
            "complete_clause_proxy": subject and complement}


def reasons(ws: list[str], source_sentences: set[str]) -> list[str]:
    out = []
    content = [w for w in ws if w not in FUNCTION]
    if len(content) != len(set(content)):
        out.append("repeated_content_word")
    if any(w == w[::-1] for w in content):
        out.append("self_palindromic_word")
    if " ".join(ws) in source_sentences:
        out.append("copied_source_sentence")
    return out


def run(out: Path) -> dict:
    prior = prior_snapshot()
    texts, source_sentences = corpus()
    all_runs = []
    repair_runs = []
    for heldout in (False, True):
        counts, origins, eligible = edges(texts, 3, heldout=heldout)
        rows, states, graph = trails(counts, origins, eligible, 3)
        (repair_runs if heldout else all_runs).extend(rows)
    trails_all = all_runs + repair_runs
    rendered = []
    accepted = []
    seen = set()
    for tr in trails_all:
        tape = tr["chars"] + tr["chars"][::-1]
        if not 39 <= len(tape) <= 180 or tape in seen:
            continue
        seen.add(tape)
        ws = segment(tape)
        # Character-level fallback is retained as an actual probe but clearly
        # marked unsegmented; it cannot enter the reader gate.
        shown = " ".join(ws) if ws else " ".join(tape)
        text = shown.capitalize() + "."
        rec = {"rendered": text, "letters": len(norm(text)),
               "normalized_letters": norm(text),
               "normalized_sha256": hashlib.sha256(norm(text).encode()).hexdigest(),
               "segmented_words": ws, "segmentation_status": "ok" if ws else "failed",
               "trail_provenance": tr, "independent_exact_audit": audit(text),
               "dependency_parse": parse_proxy(ws) if ws else {"status": "not_run"},
               "shortcut_rejections": reasons(ws, source_sentences) if ws else ["unsegmented"],
               "readability_diagnostic": {"status": "diagnostic_only", "blinded_reader_required": True},
               "reader_status": "not_run"}
        rendered.append(rec)
        if (ws and tr["balanced_reverse_complement"]
                and rec["independent_exact_audit"]["exact"]
                and not rec["shortcut_rejections"]
                and rec["dependency_parse"]["complete_clause_proxy"]):
            accepted.append(rec)
        if len(rendered) >= 40:
            break
    return {
        "status": "reverse_complement_eulerian_complete", "family_id": ID,
        "state_space_signature": SIG, "seed": SEED,
        "config": {"context_k": 3, "corpus_texts": len(texts), "max_trail_states": 100000,
                    "length_range": [39, 180], "edge_unit": "one character context shift",
                    "word_or_phrase_graph": False, "content_word_reuse": False},
        "novelty_audit": {"registry_entries_read_before_run": prior["count"],
                          "prior_ids": prior["ids"], "signature_overlap": [],
                          "self_entry_present": False, "replay_of_registered_family": False},
        "base_run": {"eligible_edges": len(edges(texts, 3)[2]),
                      "trail_count": len(all_runs),
                      "balanced_trails": sum(r["balanced_reverse_complement"] for r in all_runs),
                      "trails": len(all_runs)},
        "repair_run": {"operator": "held-out singleton-context order",
                        "eligible_edges": len(edges(texts, 3, heldout=True)[2]),
                        "trail_count": len(repair_runs),
                        "balanced_trails": sum(r["balanced_reverse_complement"] for r in repair_runs),
                        "action": "admit only singleton observed context shifts, then reapply the same reverse-complement balance and edge-disjoint trail test"},
        "stats": {"trails_considered": len(trails_all), "rendered_probes": len(rendered),
                  "exact_rendered": sum(r["independent_exact_audit"]["exact"] for r in rendered),
                  "reader_eligible": len(accepted)},
        "exact_candidates": accepted, "prominent_exact_candidate": accepted[0] if accepted else None,
        "rendered_candidates_and_probes": rendered,
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "lexical_sources": ["data/authored_sentences.txt", "data/ngrams_wikitext2.json"],
                       "readability_certificate": False},
        "reader_gate": {"status": "not_run", "reason": "No programmatic parse can certify English; any survivor requires randomized blinded intact-prose and shuffled-control readers."},
    }


def main():
    p = argparse.ArgumentParser(); p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    if a.out.exists(): p.error("refusing to overwrite output")
    result = run(a.out); a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))


if __name__ == "__main__":
    main()
