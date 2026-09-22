"""Search a typed scene lattice while consuming palindrome characters live.

This is deliberately not a phrase-pair or sentence-bank sweep.  Each event is
an ordinary authored clause with typed participants; event order and anaphoric
accessibility are graph state.  The character product is applied before any
complete derivation is rendered.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from experiments.packed_seam_grammar_20260927 import SEED, audit, intersect, norm
from llm_palindrome.validator import is_palindrome

ROOT = Path(__file__).resolve().parents[1]
ID = "scene-event-lattice-20260930"


class Lattice:
    def __init__(self):
        self.count = 1
        self.edges = []
        self.epsilon = defaultdict(list)
        self.start = 0
        self.finish = 0

    def new(self):
        n = self.count
        self.count += 1
        return n

    def branch_slot(self, source, alternatives, role):
        target = self.new()
        for phrase in alternatives:
            tape = norm(phrase)
            if not tape:
                self.epsilon[source].append(target)
                continue
            current = source
            for i, char in enumerate(tape):
                nxt = target if i == len(tape) - 1 else self.new()
                self.edges.append((current, nxt, char,
                                   phrase if i == 0 else "", role))
                current = nxt
        return target


EVENTS = {
    "sort_memos": {
        "requires": set(), "adds": {"aide", "memos"},
        "slots": (("subject", ("An aide", "A clerk", "A nurse")),
                  ("verb", (" sorts", " files", " reads")),
                  ("object", (" the memos.", " the notes."))),
    },
    "men_help_diana": {
        "requires": set(), "adds": {"men", "Diana"},
        "slots": (("subject", ("Some men", "Two men", "The writers")),
                  ("verb", (" help", " guide", " praise")),
                  ("object", (" Diana.", " Anna."))),
    },
    "diana_read_memos": {
        "requires": {"Diana", "memos"}, "adds": set(),
        "slots": (("subject", ("Diana", "She")),
                  ("verb", (" reads", " files", " saves")),
                  ("object", (" the memos.", " them."))),
    },
}


def build_lattice(max_events=3):
    g = Lattice()
    by_mask = {0: g.start}
    traces = []
    # States are semantic event masks.  Every order enters the same mask
    # state, so the lattice shares discourse history instead of enumerating
    # complete sentences.
    def state(mask):
        if mask not in by_mask:
            by_mask[mask] = g.new()
        return by_mask[mask]

    def expand(mask):
        if mask.bit_count() >= max_events:
            return
        have = set()
        if mask & 1: have |= EVENTS["sort_memos"]["adds"]
        if mask & 2: have |= EVENTS["men_help_diana"]["adds"]
        for bit, name in ((1, "sort_memos"), (2, "men_help_diana"),
                          (4, "diana_read_memos")):
            if mask & bit:
                continue
            event = EVENTS[name]
            if not event["requires"] <= have:
                continue
            source, target = state(mask), state(mask | bit)
            current = source
            for role, alternatives in event["slots"]:
                current = g.branch_slot(current, alternatives,
                                         f"{name}:{role}")
            g.epsilon[current].append(target)
            traces.append(dict(source_mask=mask, target_mask=mask | bit,
                               event=name, requires=sorted(event["requires"]),
                               adds=sorted(event["adds"])))
            expand(mask | bit)
    expand(0)
    g.finish = state(7)
    # Keep a zero-length path only for structural completeness; accepted
    # paths must contain all three events.
    return g, traces


def controls():
    return [
        dict(rendered="An aide sorts the memos. Some men help Diana. Diana reads the memos.",
             kind="intact_authored_prose", audit=audit("An aide sorts the memos. Some men help Diana. Diana reads the memos.")),
        dict(rendered="Diana reads the memos. Some men help Diana. An aide sorts the memos.",
             kind="shuffled_order_control", audit=audit("Diana reads the memos. Some men help Diana. An aide sorts the memos.")),
    ]


def run():
    g, trace = build_lattice()
    result = intersect(g, max_letters=180, cap=80000)
    for row in result["candidates"]:
        row["audit"]["independent_validator_exact"] = is_palindrome(row["rendered"])
        row["audit"]["sha_pair_equal"] = row["audit"]["sha256"] == row["audit"]["reverse_sha256"]
        row["new_content"] = norm(row["rendered"]) != norm(SEED)
        row["human_readability_evidence"] = "not collected; candidate requires blinded reader gate"
    result.update(experiment_id=ID, method="typed semantic event-mask lattice with live character intersection",
                  event_transitions=trace, controls=controls(), incumbent=SEED,
                  provenance=dict(complete_sentence_enumeration=False,
                      generated_sentence_count_before_intersection=0,
                      source="fresh authored SVO event clauses with typed participants and anaphora",
                      catalogue_replay=False, semordnilap_bank=False,
                      per_candidate_rlaif=False, generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
                  novelty_preflight=dict(novel_algorithm_claim=False,
                      representation_change="semantic event masks merge event orders and gate anaphora before character matching"),
                  reader_test=dict(status="not collected",
                      next="randomized blinded intact-prose and shuffled controls for any new exact closure"),
                  next_repair=dict(operator="replace one shared object slot with a transitive attachment while retaining event identity",
                                   reason="current lattice has no productive exact closure beyond inherited material; change attachment topology, not vocabulary breadth"))
    return result


if __name__ == "__main__":
    out = run()
    (ROOT / "runs" / f"{ID}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(dict(states=out["states"], transitions=out["transitions"],
                          exact=len(out["candidates"]), new=sum(x["new_content"] for x in out["candidates"]))))
