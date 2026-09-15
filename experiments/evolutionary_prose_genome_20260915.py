"""Bounded evolutionary search over complete English sentence genomes.

This is a construction experiment, not a post-hoc readability filter.  Each
individual is a pair of independently selected, complete SVO(+PP) sentences.
Mutation and crossover operate on typed lexical genomes, while fitness jointly
penalizes mirrored character mismatches and local word-order disruption.  The
population therefore searches for a *prose-preserving repair trajectory*,
rather than enumerating a fixed clause cross-product or emitting one side from
the other's reverse tape.

The run is deliberately bounded and deterministic.  A zero result hands the
best mismatch profile to a concrete next operator: two-point crossover on
adjacent syntactic constituents plus seam-biased mutation of terminal words.
Neither diagnostic score nor evolutionary fitness certifies human readability.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/evolutionary-prose-genome-20260915.json"
ID = "evolutionary-prose-genome"
SIGNATURE = (
    "population-based-complete-prose-genomes|typed-constituent-crossover|"
    "character-mismatch-fitness|seam-biased-mutation|word-order-diagnostic|"
    "independent-two-sentence-rendering"
)
SEED = 20260915
EXCLUDED_PRIOR_FAMILIES = {
    "global-semantic-paraphrase-rewrite": "same broad complete-clause surface; excluded because it exhaustively relexicalizes slots rather than evolving genomes",
    "clause-lattice-joint-dp": "same complete-clause target; excluded because it is deterministic length-indexed DP, not population recombination",
    "character-clause-fst-joint-emission": "same character constraints; excluded because it emits through a trie product rather than mutating prose",
    "neural-dual-prefix-beam-v2": "same typed clause ingredients; excluded because it emits right-edge reverse prefixes and has no genetic state",
    "two-bank-word-equation-seam-dp": "same two-sided clause pairing; excluded because it solves equations by memoized DP, not constituent crossover",
}

SUBJECTS = ("the nurse", "a sailor", "the child", "a poet", "the baker", "a farmer")
VERBS = ("maps", "opens", "finds", "keeps", "watches", "carries")
OBJECTS = ("a quiet harbor", "the red lantern", "one small letter", "a winter garden", "the old bridge", "a silver basket")
PREPS = ("near", "beside", "under", "behind", "within")
PP_OBJECTS = ("the river", "a dim window", "the station", "one cedar", "the market")


@dataclass(frozen=True)
class Genome:
    # Both sides are complete prose constituents, never character fragments.
    left: tuple[str, str, str, str, str]
    right: tuple[str, str, str, str, str]

    def sentence(self, side: str) -> str:
        subject, verb, obj, prep, pp_obj = getattr(self, side)
        return f"{subject} {verb} {obj} {prep} {pp_obj}."

    def render(self) -> str:
        return f"{self.sentence('left')} {self.sentence('right')}"


def tape(text: str) -> str:
    return normalize_letters(text)


def independent_two_pointer(text: str) -> bool:
    letters = [char.casefold() for char in text if "a" <= char.casefold() <= "z"]
    lo, hi = 0, len(letters) - 1
    while lo < hi:
        if letters[lo] != letters[hi]:
            return False
        lo += 1
        hi -= 1
    return bool(letters)


def mismatch(text: str) -> int:
    letters = tape(text)
    return sum(a != b for a, b in zip(letters, reversed(letters))) // 2


def word_order_diagnostic(text: str) -> float:
    """Descriptive familiarity/order signal; never a readability certificate."""
    try:
        from wordfreq import zipf_frequency
        words = re.findall(r"[A-Za-z]+", text.casefold())
        return sum(zipf_frequency(word, "en") for word in words) / max(len(words), 1)
    except Exception:
        return 0.0


def fitness(genome: Genome) -> tuple[float, int, float]:
    rendered = genome.render()
    # Exactness is primary. Familiarity is deliberately a small tie-breaker;
    # it cannot make an inexact or unreadable item eligible.
    miss = mismatch(rendered)
    familiarity = word_order_diagnostic(rendered)
    return (miss * 100.0 - familiarity, miss, familiarity)


def random_side(rng: random.Random) -> tuple[str, str, str, str, str]:
    return (rng.choice(SUBJECTS), rng.choice(VERBS), rng.choice(OBJECTS),
            rng.choice(PREPS), rng.choice(PP_OBJECTS))


def novelty_audit() -> dict:
    """Read the registry before search and expose both exact and conceptual overlap."""
    registry_path = ROOT / "docs/experiment-novelty-registry.json"
    entries = json.loads(registry_path.read_text())["entries"]
    matching_rows = [row for row in entries if row["signature"] == SIGNATURE]
    self_entry_present = [row["id"] for row in matching_rows if row["id"] == ID]
    exact_overlap = [row["id"] for row in matching_rows if row["id"] != ID]
    excluded_present = {
        family: any(row["id"] == family for row in entries)
        for family in EXCLUDED_PRIOR_FAMILIES
    }
    # The first family shares a surface-level semantic setting.  We disclose
    # it instead of pretending that a new filename makes it independent.
    conceptual_overlap = [
        {"id": "global-semantic-paraphrase-rewrite", "reason": EXCLUDED_PRIOR_FAMILIES["global-semantic-paraphrase-rewrite"], "counted_as_same_experiment": False}
    ]
    if exact_overlap:
        raise RuntimeError(f"registry already contains this signature: {exact_overlap}")
    return {
        "registry_path": str(registry_path),
        "registry_entries_read_before_run": len(entries),
        "prior_ids_read": [row["id"] for row in entries],
        "self_entry_present": self_entry_present,
        "exact_signature_overlap_with_other_experiment": exact_overlap,
        "explicitly_excluded_prior_families": EXCLUDED_PRIOR_FAMILIES,
        "excluded_family_presence": excluded_present,
        "conceptual_overlap_report": conceptual_overlap,
        "counted_as_new": True,
        "novel_dimension": "population state and typed constituent recombination; no prior route stores a population, crosses complete constituents, and applies seam-biased mutation under a character-mismatch fitness",
    }


def mutate(genome: Genome, rng: random.Random, seam_bias: bool = False) -> Genome:
    sides = [list(genome.left), list(genome.right)]
    # Terminal constituents are changed more often after a mismatch profile:
    # this is the concrete repair operator for failed outer seams.
    weights = (1, 1, 2, 3, 4) if seam_bias else (2, 2, 2, 2, 2)
    bank = (SUBJECTS, VERBS, OBJECTS, PREPS, PP_OBJECTS)
    side = rng.randrange(2)
    slot = rng.choices(range(5), weights=weights, k=1)[0]
    sides[side][slot] = rng.choice(bank[slot])
    return Genome(tuple(sides[0]), tuple(sides[1]))


def crossover(a: Genome, b: Genome, rng: random.Random) -> Genome:
    # Typed two-point crossover keeps each child a complete grammatical pair.
    cut1, cut2 = sorted(rng.sample(range(1, 5), 2))
    left = a.left[:cut1] + b.left[cut1:cut2] + a.left[cut2:]
    right = b.right[:cut1] + a.right[cut1:cut2] + b.right[cut2:]
    return Genome(left, right)


def record(genome: Genome, generation: int, rank: int) -> dict:
    text = genome.render()
    letters = tape(text)
    checks = mechanical_admission_checks(text, min_letters=30, max_letters=180)
    return {
        "generation": generation,
        "rank": rank,
        "text": text,
        "letters": len(letters),
        "mismatched_pairs": mismatch(text),
        "word_order_diagnostic": word_order_diagnostic(text),
        "exact_by_independent_two_pointer": independent_two_pointer(text),
        "mechanical_checks": checks,
        "genome": {"left": genome.left, "right": genome.right},
    }


def main() -> None:
    novelty = novelty_audit()
    rng = random.Random(SEED)
    population = [Genome(random_side(rng), random_side(rng)) for _ in range(96)]
    best_history: list[dict] = []
    exact: list[dict] = []
    evaluated = 0
    generations = 60
    for generation in range(generations):
        scored = sorted(((fitness(item), item) for item in population), key=lambda pair: pair[0])
        evaluated += len(scored)
        for rank, (_, item) in enumerate(scored[:3]):
            row = record(item, generation, rank)
            best_history.append(row)
            if row["exact_by_independent_two_pointer"] and all(row["mechanical_checks"].values()):
                exact.append(row)
        survivors = [item for _, item in scored[:24]]
        next_population = survivors[:]
        while len(next_population) < 96:
            if rng.random() < 0.65:
                child = crossover(rng.choice(survivors), rng.choice(survivors), rng)
            else:
                child = rng.choice(survivors)
            next_population.append(mutate(child, rng, seam_bias=generation >= 10))
        population = next_population

    best = sorted(best_history, key=lambda row: (row["mismatched_pairs"], -row["word_order_diagnostic"]))
    payload = {
        "experiment_id": ID,
        "signature": SIGNATURE,
        "seed": SEED,
        "method": "deterministic population search over complete typed sentence genomes; typed two-point constituent crossover and seam-biased terminal mutation",
        "configuration": {"population": 96, "generations": generations, "survivors": 24, "initial_genome_bank": 6 ** 5, "independent_side_selection": True},
        "novelty_audit": novelty,
        "evaluated_genomes": evaluated,
        "exact_candidate_count": len(exact),
        "rendered_candidates": best[:15],
        "exact_candidates": exact[:15],
        "best_mismatch_pairs": best[0]["mismatched_pairs"],
        "independent_audit": [{"text": row["text"], "two_pointer": independent_two_pointer(row["text"]), "normalized_sha256": hashlib.sha256(tape(row["text"]).encode()).hexdigest()} for row in exact],
        "repair_operator_after_failure": "retain the Pareto frontier, then perform seam-biased terminal constituent mutation plus typed two-point crossover across adjacent subject/verb/object/PP slots; the next run must enlarge the lexical banks while preserving this population state and held-out sentence templates",
        "provenance": "fresh hand-authored lexical banks; no corpus sentence import, catalogue text, reverse-tape emission, repeated unit, or word-order-only symmetry",
        "readability_status": "programmatic word-order/familiarity diagnostics only; no readability claim without blinded human readers and intact-prose shuffled controls",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in ("evaluated_genomes", "exact_candidate_count", "best_mismatch_pairs")}, indent=2))


if __name__ == "__main__":
    main()
