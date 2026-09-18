"""Discover typed edge channels, then intersect complete character grammars.

Generation uses authored individual lexical inventories, not phrase pairs.
The separately declared surface validator imports none of these inventories.
No prose is promoted before central admission AND external provenance review.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json
from math import prod
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.whole_text_palindrome_product_20260913 import compile_slots, replay_path
from experiments.boundary_disjoint_semantic_chart_20260913 import lexical_boundaries, boundary_island
from experiments.semantic_edge_validator_20260913 import parse_surface
from llm_palindrome.admission import REPEATABLE_FUNCTION_WORDS, mechanical_admission_checks

SUBJECTS = tuple("""
actors actresses agents analysts artists artisans assistants athletes authors
bakers bankers barbers builders butchers carpenters cashiers chefs clerks
coaches cooks couriers critics dancers designers detectives directors doctors
drivers editors engineers farmers firefighters fishers gardeners guides hunters
inspectors instructors janitors jewelers judges lawyers lecturers librarians
managers mechanics merchants messengers miners musicians nurses officers
painters pilots plumbers poets porters printers professors programmers
publishers researchers sailors scientists sculptors secretaries sergeants
singers soldiers tailors teachers technicians translators waiters wardens
watchmakers welders workers writers buyers clients customers guests neighbors
owners parents partners passengers patients people residents shoppers
spectators students survivors tenants tourists travelers visitors voters
witnesses volunteers friends lovers men women youths adults peers elders pupils readers
""".split())

NOUN_INVENTORY = {
    ("artifact", "singular"): "car cart truck wagon engine motor pump machine tool hammer mallet saw drill file knife needle pin clip net grid map chart book box bag basket bottle cup plate bowl fork spoon chair desk table bed sofa shelf cabinet lamp candle clock watch ring jewel key lock door window gate fence rope rag mat sheet panel prop paddle saddle handle cradle bathtub bucket shovel barrel drum helmet glove shoe boot coat scarf hat bonnet computer phone camera laser eraser browser freezer generator battery magnet",
    ("artifact", "plural"): "cars carts trucks wagons engines motors pumps machines tools hammers mallets saws drills files knives needles pins clips nets grids maps charts books boxes bags baskets bottles cups plates bowls forks spoons chairs desks tables beds sofas shelves cabinets lamps candles clocks watches rings jewels keys locks doors windows gates fences ropes rags mats sheets panels props paddles saddles handles cradles bathtubs buckets shovels barrels drums helmets gloves shoes boots coats scarves hats bonnets computers phones cameras lasers erasers browsers freezers generators batteries magnets steps",
    ("material", "mass"): "wood metal steel iron copper silver gold paper cloth cotton wool silk denim leather plastic rubber glass clay sand soil mud gravel gypsum plaster concrete pulp ink paint glue wax soap oil coal dust",
    ("animal", "singular"): "dog cat bird horse cow pig goat sheep deer duck goose hen rabbit mouse rat bear wolf fox lion tiger elephant monkey ape camel donkey zebra giraffe otter beaver badger mole squirrel hedgehog bat whale dolphin seal walrus shark turtle frog snake lizard iguana gerbil sparrow blackbird bluebird parrot owl eagle falcon swan bee wasp beetle scarab ant butterfly moth spider crab lobster shrimp snail clam mussel oyster octopus squid barracuda",
    ("food", "mass"): "bread rice pasta cheese butter cream milk meat beef pork chicken turkey fish cod trout herring salmon tuna flour sugar salt pepper honey jam soup stew sauce curry fruit corn cabbage spinach broccoli garlic ginger cereal oatmeal toast grits chocolate yogurt custard pie cake pudding popcorn",
    ("food", "singular"): "apple pear peach plum banana orange lemon lime melon grape cherry strawberry raspberry potato tomato carrot onion cucumber pepper turnip radish bean pea egg sandwich tart biscuit cracker pancake waffle noodle",
    ("food", "plural"): "apples pears peaches plums bananas oranges lemons limes melons grapes cherries strawberries raspberries potatoes tomatoes carrots onions cucumbers peppers turnips radishes beans peas eggs sandwiches tarts biscuits crackers pancakes waffles noodles nuts spices",
    ("drink", "mass"): "water tea coffee juice cider beer wine champagne lemonade soda cocoa",
    ("text", "singular"): "report letter essay note article story novel poem sonnet verse memoir diary journal manual recipe message notice document contract invoice receipt ticket record theorem formula proof argument definition description summary review proposal plan question answer",
    ("text", "plural"): "reports letters essays notes articles stories novels poems sonnets verses memoirs diaries journals manuals recipes messages notices documents contracts invoices receipts tickets records theorems formulas proofs arguments definitions descriptions summaries reviews proposals plans questions answers",
    ("person", "singular"): "doctor nurse teacher writer author editor artist driver worker farmer gardener sailor scientist researcher lawyer judge clerk cook chef baker mechanic pilot guide reader student visitor neighbor friend partner parent tenant patient guest buyer client customer owner stranger leader assistant manager inspector translator musician singer dancer actor carpenter plumber electrician soldier officer peer elder",
    ("ensemble", "singular"): "orchestra band choir ensemble team crew committee group club company class audience family",
    ("music", "mass"): "music jazz rap rock blues folk opera reggae gospel pop",
    ("sport", "mass"): "tennis golf soccer football baseball basketball cricket rugby hockey volleyball badminton squash chess checkers poker bridge",
    ("activity", "mass"): "sleep work exercise rest travel dance play recreation meditation practice training study research gardening reading writing drawing painting sewing knitting fishing hunting hiking sailing swimming running cycling walking cooking shopping cleaning",
    ("activity", "singular"): "run walk ride trip visit race game match dance swim climb flight drive tour break nap draw",
    ("discipline", "mass"): "physics chemistry biology mathematics algebra geometry calculus statistics mechanics dynamics economics politics ethics logic philosophy history geography geology astronomy linguistics literature medicine law psychology sociology ecology art architecture engineering education music",
    ("place", "singular"): "park office library studio museum school hospital market shop store station harbor airport theater hotel cafe restaurant factory garage garden forest river beach island desert mountain valley village town city country bedroom workshop bank bridge street road avenue alley courtyard kitchen temple church mosque chapel cathedral monastery stadium pavilion arena palace castle cottage cabin barn shed cabana crag laboratory depot port nursery greenhouse",
    ("place", "plural"): "parks offices libraries studios museums schools hospitals markets shops stores stations harbors airports theaters hotels cafes restaurants factories garages gardens forests rivers beaches islands deserts mountains valleys villages towns cities countries bedrooms workshops banks bridges streets roads avenues alleys courtyards kitchens temples churches mosques chapels cathedrals monasteries stadiums pavilions arenas palaces castles cottages cabins barns sheds cabanas crags laboratories depots ports nurseries greenhouses",
    ("plant", "singular"): "plant tree flower herb seed seedling sapling bush shrub rose tulip lily orchid fern moss vine cactus palm oak elm pine maple willow reed grass crop",
    ("plant", "plural"): "plants trees flowers herbs seeds seedlings saplings bushes shrubs roses tulips lilies orchids ferns vines palms oaks elms pines maples willows reeds crops",
}
MODIFIERS = {
    "artifact": "small large old new red black", "material": "clean dry raw", "animal": "young small large calm",
    "food": "fresh hot cold", "drink": "fresh cold", "text": "brief detailed old new",
    "person": "quiet patient skilled young old", "ensemble": "small large skilled", "music": "fine soft loud fast slow",
    "sport": "competitive", "activity": "brief long", "discipline": "modern advanced basic",
    "place": "quiet large old", "plant": "young small large",
}
PHYSICAL = frozenset("artifact material animal food drink text person ensemble place plant".split())
VERBS = {
    "see": PHYSICAL, "observe": PHYSICAL, "inspect": PHYSICAL, "find": PHYSICAL,
    "need": PHYSICAL | {"activity"}, "want": PHYSICAL | {"activity"}, "choose": PHYSICAL, "read": {"text"},
    "study": set("text discipline animal plant material activity place artifact".split()),
    "buy": set("artifact material animal food drink text plant".split()), "sell": set("artifact material animal food drink text plant".split()),
    "use": {"artifact", "material", "text"}, "hire": {"person", "ensemble"}, "help": {"person"},
    "admire": set("animal artifact person ensemble place plant".split()), "enjoy": set("food drink sport activity discipline music text".split()),
    "play": {"sport", "music"}, "eat": {"food"}, "drink": {"drink"}, "collect": {"artifact", "material", "text", "plant"},
    "clean": {"artifact", "material", "food"}, "carry": {"artifact", "material", "food", "text", "plant", "animal"},
    "avoid": {"person", "animal", "food", "drink", "place", "activity"}, "visit": {"place", "person"}, "enter": {"place"},
    "join": {"ensemble"}, "hear": {"ensemble", "music", "animal", "person"}, "cook": {"food"}, "prepare": {"food", "drink"},
    "fix": {"artifact"}, "repair": {"artifact"}, "move": {"artifact", "material"}, "handle": {"artifact", "material", "animal"},
    "write": {"text"}, "recite": {"text"}, "discuss": {kind for kind, _ in NOUN_INVENTORY},
}


@dataclass(frozen=True)
class Noun:
    word: str
    kind: str
    number: str


NOUNS = tuple(Noun(word, kind, number) for (kind, number), words in NOUN_INVENTORY.items() for word in words.split())


def overlap(left, right):
    depth = 0
    for a, b in zip(left, right[::-1]):
        if a != b:
            break
        depth += 1
    return depth


def discover_channels(subjects=SUBJECTS, nouns=NOUNS, verbs=VERBS, min_pairs=3):
    if min_pairs < 3:
        raise ValueError("edge eligibility requires at least three real matched letters")
    index = defaultdict(list)
    for noun in nouns:
        if len(noun.word) >= min_pairs:
            index[noun.word[::-1][:min_pairs]].append(noun)
    channels = []
    for subject in subjects:
        if len(subject) < min_pairs:
            continue
        for noun in index.get(subject[:min_pairs], ()):
            for verb, object_types in sorted(verbs.items()):
                if noun.kind in object_types:
                    channels.append((subject, verb, noun, overlap(subject, noun.word)))
    return channels


def noun_phrase_forms(noun):
    """All finite determiner/modifier forms; no phrase is selected as a seed."""
    groups = [("plain", ())]
    adjectives = tuple(MODIFIERS[noun.kind].split())
    groups.extend((label, (words,)) for label, words in (
        ("vowel_modifier", tuple(w for w in adjectives if w[0] in "aeiou")),
        ("consonant_modifier", tuple(w for w in adjectives if w[0] not in "aeiou"))) if words)
    for label, modifiers in groups:
        initial = modifiers[0][0] if modifiers else noun.word
        if noun.number == "singular":
            det = "an" if initial[0] in "aeiou" else "a"
            yield label, ((det, "the"),) + modifiers + ((noun.word,),)
        else:
            yield label + ":bare", modifiers + ((noun.word,),)
            yield label + ":determined", (("the", "some"),) + modifiers + ((noun.word,),)


def complete_layouts(subject, verb, final_np):
    start, predicate = ((subject,),), ((verb,),)
    manner = (("carefully", "quietly", "slowly"),)
    relative = (("who",), ("read", "reviewed", "edited", "printed"), ("brief", "detailed", "old", "new"),
                ("reports", "letters", "essays", "notes"))
    source = (("from",), ("tired", "careful", "skilled"), ("authors", "editors", "writers", "researchers"))
    location = (("in", "inside", "near"), ("quiet", "large", "old"), ("offices", "studios", "libraries", "laboratories", "workshops", "schools"))
    return {
        "simple": start + predicate + final_np,
        "manner": start + manner + predicate + final_np,
        "relative": start + relative + predicate + final_np,
        "relative_source": start + relative + source + manner + predicate + final_np,
        "relative_source_location": start + relative + source + location + manner + predicate + final_np,
    }


def chart(slots):
    grammar = compile_slots(slots)
    boundaries = lexical_boundaries(grammar)
    outgoing, incoming = defaultdict(list), defaultdict(list)
    for edge in grammar.edges:
        outgoing[edge.source].append(edge)
        incoming[edge.target].append(edge)

    @lru_cache(maxsize=None)
    def reachable(node):
        return frozenset({node}.union(*(reachable(edge.target) for edge in outgoing[node])))

    reachable(grammar.start)
    stack = [(grammar.start, grammar.end, (), (), 0, (), ())]
    stats, depths, boundary_depths = Counter(), Counter(), Counter()
    records, witnesses = [], []
    deepest = {"depth": -1}
    while stack:
        left, right, prefix, suffix, depth, left_depths, right_depths = stack.pop()
        stats["states"] += 1
        depths[depth] += 1
        if depth > 0:
            if left in boundaries:
                boundary_depths[f"left:{depth}"] += 1
            if right in boundaries:
                boundary_depths[f"right:{depth}"] += 1
            if left in boundaries and right in boundaries:
                boundary_depths[f"joint:{depth}"] += 1
        if depth > deepest["depth"]:
            deepest = {"depth": depth, "left_boundary_depths": left_depths, "right_boundary_depths": right_depths,
                       "left_token_offset": boundaries.get(left), "right_token_offset": boundaries.get(right),
                       "left_next_characters": sorted({e.char for e in outgoing[left]}),
                       "right_next_characters": sorted({e.char for e in incoming[right]})}
        island = boundary_island(left, right, boundaries, depth)
        if island:
            stats["proper_island_prunes"] += 1
            if len(witnesses) < 3:
                witnesses.append(island)
            continue
        if depth and left in boundaries and right in boundaries and boundaries[right] - boundaries[left] == 1:
            if not any(w == w[::-1] and w in REPEATABLE_FUNCTION_WORDS for w in slots[boundaries[left]]):
                stats["single_content_center_prunes"] += 1
                continue
        middles = [()] if left == right else []
        middles.extend((edge,) for edge in outgoing[left] if edge.target == right)
        for middle in middles:
            row = replay_path(grammar, prefix + middle + suffix)
            if not row["ok"] or not row["exact"]:
                raise AssertionError("invalid full character-path closure")
            records.append({**row, "matched_depth": depth, "center_characters": len(middle)})
        advances = 0
        for first in outgoing[left]:
            for last in incoming[right]:
                if first.char == last.char and last.source in reachable(first.target):
                    advances += 1
                    nxt = depth + 1
                    stack.append((first.target, last.source, prefix + (first,), (last,) + suffix, nxt,
                                  left_depths + ((nxt,) if first.target in boundaries else ()),
                                  right_depths + ((nxt,) if last.source in boundaries else ())))
        if not advances and not middles:
            stats["character_dead_frontiers"] += 1
    return {"stats": dict(stats), "state_depth_distribution": dict(depths), "boundary_depth_distribution": dict(boundary_depths),
            "deepest": deepest, "prune_witnesses": witnesses, "records": records, "states_exhausted": True}


def run(min_letters=100, max_letters=240):
    channels = discover_channels()
    total, depths, boundary_depths, max_depths, rejected = Counter(), Counter(), Counter(), Counter(), Counter()
    rows, pending = [], []
    products = lexical_realizations = 0
    for subject, verb, noun, indexed_pairs in channels:
        row_stats, row_depths = Counter(), Counter()
        deepest = {"depth": -1}
        product_count = long_products = exact = 0
        for form, np_slots in noun_phrase_forms(noun):
            for layout, slots in complete_layouts(subject, verb, np_slots).items():
                result = chart(slots)
                if result["deepest"]["depth"] < 3:
                    raise AssertionError("an indexed edge channel failed to replay three actual letter pairs")
                products += 1
                product_count += 1
                lexical_realizations += prod(map(len, slots))
                long_products += sum(max(map(len, slot)) for slot in slots) >= min_letters
                total.update(result["stats"])
                row_stats.update(result["stats"])
                depths.update(result["state_depth_distribution"])
                boundary_depths.update(result["boundary_depth_distribution"])
                row_depths[result["deepest"]["depth"]] += 1
                max_depths[result["deepest"]["depth"]] += 1
                if result["deepest"]["depth"] > deepest["depth"]:
                    deepest = {**result["deepest"], "layout": layout, "np_form": form}
                exact += len(result["records"])
                for record in result["records"]:
                    words = tuple(record["words"])
                    parses = parse_surface(words)
                    checks = mechanical_admission_checks(" ".join(words), min_letters=min_letters, max_letters=max_letters)
                    codes = [key for key, value in checks.items() if not value]
                    if not parses:
                        codes.append("independent_reparse_failed")
                    if codes:
                        rejected.update(codes)
                    else:
                        # Internal tokens support subsequent provenance queries;
                        # there is deliberately no prose renderer or promotion.
                        pending.append({"tokens": words, "normalized": record["tape"], "letters": record["letters"],
                                        "sha256": sha256(record["tape"].encode()).hexdigest(), "mechanical_checks": checks,
                                        "independent_parses": parses, "external_provenance": "required; unchecked",
                                        "promoted": False})
        rows.append({"subject": subject, "predicate": verb, "final_head": noun.word, "object_type": noun.kind,
                     "object_number": noun.number, "indexed_outer_pairs": indexed_pairs,
                     "actual_eligible_products": product_count, "products_reaching_length_band": long_products,
                     "exact_closures": exact, "max_depth_distribution": dict(row_depths), "deepest": deepest,
                     "stats": dict(row_stats)})
    validator = ROOT / "experiments/semantic_edge_validator_20260913.py"
    return {"status": "lexicon_indexed_semantic_edge_channels", "config": {
        "min_letters": min_letters, "max_letters": max_letters, "min_actual_outer_pairs": 3,
        "fixed_individual_lexemes_not_phrase_pairs": True, "typed_subject_predicate_final_constituent_join": True,
        "all_selected_surface_products_exhausted": True, "free_midpoint": True, "actual_boundary_depth_state": True,
        "online_proper_island_prune": True, "separately_declared_reparser": True,
        "central_and_external_provenance_required_before_promotion": True},
        "inventory_counts": {"plural_human_subjects": len(SUBJECTS), "typed_noun_forms": len(NOUNS), "verbs": len(VERBS),
                             "object_types": len(MODIFIERS)},
        "eligible_channels": len(channels), "selected_products": products, "lexical_realizations": lexical_realizations,
        "stats": dict(total), "state_depth_distribution": dict(depths), "boundary_depth_distribution": dict(boundary_depths),
        "max_depth_distribution": dict(max_depths), "rows": rows, "exact_closures": sum(r["exact_closures"] for r in rows),
        "rejections": dict(rejected), "pending_external_review": pending, "promoted_candidates": [],
        "states_exhausted": True, "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                                                  "independent_validator_sha256": sha256(validator.read_bytes()).hexdigest(),
                                                  "generation_material": "fixed authored lexical role inventories; no source phrase pairs or palindrome seeds"},
        "scope": "Exhaustion covers these selected finite products. Three matched pairs certify edge eligibility only; typed parsing is not reader acceptance, and external provenance is required before any promotion."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-letters", type=int, default=100)
    parser.add_argument("--max-letters", type=int, default=240)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.min_letters, args.max_letters)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: report[key] for key in ("inventory_counts", "eligible_channels", "selected_products", "stats", "max_depth_distribution", "exact_closures")}))
        print(json.dumps({"output": str(args.output), "pending_external_review": len(report["pending_external_review"]), "promoted_candidates": 0}))
    else:
        print(json.dumps(report, indent=2))
