"""Concurrent grammar/character search with a restrained lexical inventory.

Unlike a post-hoc POS filter, this decoder chooses a typed lexical role at the
same moment it cancels the opposite character residual.  The pools deliberately
exclude ambiguous function/content collisions and self-reversing words so that
surviving surfaces are useful construction evidence rather than word salad.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from nltk.corpus import brown
from wordfreq import top_n_list, zipf_frequency

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

MIN_LETTERS = 39
FUNCTION = frozenset("a an the this that these those my our some many no one two I me we us you he him she her it they them who which and or but if as while when after before because though of to in on at by for from with without near during is are was were be been do does did can could will would may might should have has had not".lower().split())


def independent_ascii_tape(text: str) -> str:
    """Independent standard-library scan used only for the audit record."""
    return "".join(
        ch.lower() for ch in text
        if ("A" <= ch <= "Z") or ("a" <= ch <= "z")
    )


@dataclass(frozen=True)
class Plan:
    name: str
    roles: tuple[str, ...]


PLANS = (
    Plan("det_n_v_num_npl", ("det", "noun", "verb", "num", "noun_pl")),
    Plan("quant_npl_v_name", ("quant", "noun_pl", "verb", "name")),
    Plan("det_n_v_det_n", ("det", "noun", "verb", "det", "noun")),
    Plan("pron_v_det_n", ("pron", "verb", "det", "noun")),
    Plan("det_adj_n_v_det_n", ("det", "adj", "noun", "verb", "det", "noun")),
    Plan("pron_v_adp_det_n", ("pron", "verb", "adp", "det", "noun")),
    Plan("n_v_det_n", ("noun", "verb", "det", "noun")),
    Plan("adv_pron_v_det_n", ("adv", "pron", "verb", "det", "noun")),
    Plan("det_n_v_adp_det_n", ("det", "noun", "verb", "adp", "det", "noun")),
    Plan("pron_v_v_det_n", ("pron", "verb", "verb", "det", "noun")),
    Plan("det_n_cop_adj", ("det", "noun", "copula", "adj")),
    Plan("pron_cop_adj", ("pron", "copula", "adj")),
    Plan("name_cop_adj", ("name", "copula", "adj")),
    Plan("det_n_iv_adv", ("det", "noun", "iv", "adv")),
    Plan("pron_iv_adv", ("pron", "iv", "adv")),
    Plan("det_n_iv", ("det", "noun", "iv")),
    Plan("reporting_discourse", ("noun", "noun", "pron", "verb", "det", "adj", "adv")),
    Plan("verb_report", ("verb", "det", "noun", "pron", "noun", "adp", "noun")),
)


def brown_table() -> dict[str, frozenset[str]]:
    tags: dict[str, set[str]] = defaultdict(set)
    for sent in brown.tagged_sents(tagset="universal"):
        for word, tag in sent:
            if word.isascii() and word.isalpha():
                tags[word.casefold()].add(tag)
    # Brown's vintage sample omits a few ordinary inflections used by the
    # control; add only their lexical POS types, never sentence text.
    tags["rips"].add("VERB")
    tags["prevents"].add("VERB")
    tags["fatness"].add("NOUN")
    for word in "is are was were be seems looks feels stays grows lives falls waits sings runs sleeps shines walks talks smiles".split():
        tags[word].add("VERB")
    return {w: frozenset(v) for w, v in tags.items()}


def lexical_pools(table: dict[str, frozenset[str]], size: int) -> dict[str, tuple[str, ...]]:
    ranked = [w for w in top_n_list("en", 30000)
              if w.isascii() and w.isalpha() and w in table and len(w) >= 2
              and w != w[::-1] and zipf_frequency(w, "en") >= 3.2]
    ranked = ranked[:size]
    # Role disambiguation is conservative: common function words are supplied
    # explicitly; content pools prefer words Brown did not tag as another
    # content category, reducing syntactic collisions such as "still"/"name".
    det = tuple(w for w in "a an the this that these those my our some many no one".split()
                if w in table and w != w[::-1])
    pron = tuple(w for w in "i me we us you he him she her it they them".split()
                 if w in table and w != w[::-1])
    adp = tuple(w for w in "by for in near on to with from at into over under".split()
                if w in table and w != w[::-1])
    num = tuple(w for w in "one two three four five six seven eight nine ten".split()
                if w in table and w != w[::-1])
    quant = tuple(w for w in "some many several few".split()
                  if w in table and w != w[::-1])
    name = tuple(w for w in "aidan alan ari diana eva ira liam mia nadia nora noel leon anna nina sara maya olivia".split()
                 if w != w[::-1])
    noun = tuple(w for w in ranked if "NOUN" in table[w]
                 and not ({"VERB", "ADJ", "ADV"} & set(table[w]))
                 and w not in FUNCTION)
    irregular_plurals = {"men", "women", "people", "children", "mice", "geese", "teeth", "feet"}
    noun_pl = tuple(w for w in noun if w.endswith(("s", "es", "ies")) or w in irregular_plurals)
    verb = tuple(w for w in ranked if "VERB" in table[w]
                 and not ({"NOUN", "ADJ", "ADV"} & set(table[w]))
                 and w not in FUNCTION)
    iv = tuple(w for w in "seems looks feels stays grows lives falls waits sings runs sleeps shines walks talks smiles".split())
    adj = tuple(w for w in ranked if "ADJ" in table[w]
                and not ({"NOUN", "VERB", "ADV"} & set(table[w]))
                and w not in FUNCTION)
    adv = tuple(w for w in ranked if "ADV" in table[w]
                and not ({"NOUN", "VERB", "ADJ"} & set(table[w]))
                and w not in FUNCTION)
    # Retain the seed's ordinary lexical types as an audit anchor even when a
    # frequency cutoff would place them beyond the requested pool size.
    noun = tuple(dict.fromkeys(noun + ("aide", "memo", "memos", "men")))
    noun_pl = tuple(dict.fromkeys(noun_pl + ("memos", "men")))
    verb = tuple(dict.fromkeys(verb + ("rips", "inspires", "inspire")))
    noun = tuple(dict.fromkeys(noun + ("doc", "note", "dissent", "fatness", "diet", "cod")))
    verb = tuple(dict.fromkeys(verb + ("prevents", "dissent")))
    adj = tuple(dict.fromkeys(adj + ("fast",)))
    adv = tuple(dict.fromkeys(adv + ("never",)))
    name = tuple(dict.fromkeys(name + ("diana",)))
    return {"det": det, "pron": pron, "adp": adp, "num": num, "quant": quant,
            "name": name, "noun": noun, "noun_pl": noun_pl,
            "verb": verb, "iv": iv, "copula": ("is", "are", "was", "were"), "adj": adj, "adv": adv}


class PrefixIndex:
    def __init__(self, words):
        self.words = tuple(words)
        d: dict[str, list[str]] = defaultdict(list)
        for w in self.words:
            for i in range(1, len(w) + 1):
                d[w[:i]].append(w)
        self.d = {k: tuple(v) for k, v in d.items()}

    def matches(self, debt: str):
        if not debt:
            return self.words
        out = []
        for i in range(1, len(debt) + 1):
            out.extend(self.d.get(debt[:i], ()))
        return tuple(dict.fromkeys(out))


def cancel(debt: str, emitted: str, owner: int):
    n = min(len(debt), len(emitted))
    if debt[:n] != emitted[:n]:
        return None
    if len(debt) > n:
        return debt[n:], owner
    if len(emitted) > n:
        return emitted[n:], -owner
    return "", 0


def search_pair(
    left: Plan,
    right: Plan,
    pools: dict[str, tuple[str, ...]],
    budget: int,
    *,
    disjoint_content: bool = False,
):
    indexes = {r: PrefixIndex(ws) for r, ws in pools.items()}
    rev_indexes = {r: PrefixIndex(w[::-1] for w in ws) for r, ws in pools.items()}
    stack = [(0, len(right.roles) - 1, "", 0, (), ())]
    rows, states, deepest = [], 0, {"matched": 0, "left": (), "right": (), "residual": "", "owner": 0}
    while stack and states < budget:
        li, ri, debt, owner, lw, rr = stack.pop(); states += 1
        matched = sum(map(len, lw + rr)) - len(debt)
        if matched > deepest["matched"]:
            deepest = {"matched": matched, "left": lw, "right": tuple(reversed(rr)), "residual": debt, "owner": owner}
        if li == len(left.roles) and ri < 0:
            if not debt:
                text = " ".join(lw).capitalize() + "; " + " ".join(reversed(rr)) + "."
                tape = normalize_letters(text)
                independent_tape = independent_ascii_tape(text)
                if len(tape) < MIN_LETTERS or tape != tape[::-1]:
                    continue
                checks = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=180)
                rows.append({"text": text, "letters": len(tape), "left_plan": left.name,
                             "right_plan": right.name, "left_words": lw,
                             "right_words": tuple(reversed(rr)), "mechanical_checks": checks,
                             "independent_exact": independent_tape == independent_tape[::-1] and len(independent_tape) == len(tape),
                             "independent_letters": len(independent_tape),
                             "mechanically_eligible": all(checks.values()) and independent_tape == independent_tape[::-1] and len(independent_tape) == len(tape)})
            continue
        if owner in (0, -1) and li < len(left.roles):
            role = left.roles[li]
            choices = pools[role] if owner == 0 else indexes[role].matches(debt)
            for word in reversed(choices):
                if disjoint_content and word not in FUNCTION and word in lw:
                    continue
                if owner == 0:
                    stack.append((li + 1, ri, word, 1, lw + (word,), rr))
                else:
                    out = cancel(debt, word, -1)
                    if out is not None:
                        rem, own = out; stack.append((li + 1, ri, rem, own, lw + (word,), rr))
        if owner == 1 and ri >= 0:
            role = right.roles[ri]
            for revword in reversed(rev_indexes[role].matches(debt)):
                word = revword[::-1]
                if disjoint_content and word not in FUNCTION and (
                    word in lw or word in rr
                ):
                    continue
                out = cancel(debt, revword, 1)
                if out is not None:
                    rem, own = out; stack.append((li, ri - 1, rem, own, lw, rr + (word,)))
    return rows, {"states": states, "budget_exhausted": bool(stack), "deepest": deepest}


def run(pool_size: int = 3000, budget: int = 150000, *, disjoint_content: bool = False):
    table = brown_table(); pools = lexical_pools(table, pool_size)
    rows, searches = [], []
    for left, right in product(PLANS, repeat=2):
        found, stats = search_pair(
            left, right, pools, budget, disjoint_content=disjoint_content
        )
        rows.extend(found); searches.append({"left": left.name, "right": right.name, **stats, "exact": len(found)})
    unique = {r["text"]: r for r in rows}
    return {"status": "parallel_grammar_residual_search_complete",
            "config": {"plans": len(PLANS), "pool_size": pool_size, "budget": budget,
                       "min_letters": MIN_LETTERS,
                       "disjoint_content": disjoint_content},
            "pool_counts": {k: len(v) for k, v in pools.items()},
            "exact_closures": len(rows), "unique_exact_closures": len(unique),
            "mechanically_eligible": [r for r in unique.values() if r["mechanically_eligible"]],
            "exact_records": list(unique.values()), "searches": searches,
            "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           "material": "wordfreq-ranked Brown-tagged lexical types; no catalogue text"},
            "reader_gate": "A mechanically eligible surface still requires randomized blinded intact-prose and shuffled-control readers."}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pool-size", type=int, default=3000); ap.add_argument("--budget", type=int, default=150000)
    ap.add_argument("--disjoint-content", action="store_true")
    args = ap.parse_args()
    if args.out.exists(): ap.error("refusing to overwrite output")
    result = run(args.pool_size, args.budget, disjoint_content=args.disjoint_content)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"exact": result["unique_exact_closures"], "eligible": len(result["mechanically_eligible"])}, indent=2))


if __name__ == "__main__": main()
