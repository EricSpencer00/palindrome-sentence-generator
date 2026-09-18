"""Independent lexical declarations and surface grammar for edge-channel audit.

This module imports no constructor, frame inventory, slot layout or chart.
Recognition starts again from a complete token sequence. Its duplicated
declarations deliberately prevent a generator-only role annotation from
certifying itself. This parser checks a finite grammar, not human readability.
"""
from __future__ import annotations

HUMANS = frozenset("""
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

# Surface nouns and number are declared independently from the search lexicon.
NOUNS = {
    ("artifact", "singular"): "car cart truck wagon engine motor pump machine tool hammer mallet saw drill file knife needle pin clip net grid map chart book box bag basket bottle cup plate bowl fork spoon chair desk table bed sofa shelf cabinet lamp candle clock watch ring jewel key lock door window gate fence rope rag mat sheet panel prop paddle saddle handle cradle bathtub bucket shovel barrel drum helmet glove shoe boot coat scarf hat bonnet computer phone camera laser eraser browser freezer generator battery magnet",
    ("artifact", "plural"): "cars carts trucks wagons engines motors pumps machines tools hammers mallets saws drills files knives needles pins clips nets grids maps charts books boxes bags baskets bottles cups plates bowls forks spoons chairs desks tables beds sofas shelves cabinets lamps candles clocks watches rings jewels keys locks doors windows gates fences ropes rags mats sheets panels props paddles saddles handles cradles bathtubs buckets shovels barrels drums helmets gloves shoes boots coats scarves hats bonnets computers phones cameras lasers erasers browsers freezers generators batteries magnets steps",
    ("material", "mass"): "wood metal steel iron copper silver gold paper cloth cotton wool silk denim leather plastic rubber glass clay sand soil mud gravel gypsum plaster concrete pulp ink paint glue wax soap oil coal dust",
    ("animal", "singular"): "dog cat bird horse cow pig goat sheep deer duck goose hen rabbit mouse rat bear wolf fox lion tiger elephant monkey ape camel donkey zebra giraffe otter beaver badger mole squirrel hedgehog bat whale dolphin seal walrus shark turtle frog snake lizard iguana gerbil sparrow blackbird bluebird parrot owl eagle falcon swan bee wasp beetle scarab ant butterfly moth spider crab lobster shrimp snail clam mussel oyster octopus squid barracuda",
    ("animal", "plural"): "dogs cats birds horses cows pigs goats ducks geese hens rabbits mice rats bears wolves foxes lions tigers elephants monkeys apes camels donkeys zebras giraffes otters beavers badgers moles squirrels hedgehogs bats whales dolphins seals walruses sharks turtles frogs snakes lizards iguanas gerbils sparrows blackbirds bluebirds parrots owls eagles falcons swans bees wasps beetles scarabs ants butterflies moths spiders crabs lobsters snails clams mussels oysters",
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
LEXICAL_NOUNS = {}
for (kind, number), values in NOUNS.items():
    for value in values.split():
        LEXICAL_NOUNS.setdefault(value, set()).add((kind, number))
for human in HUMANS:
    LEXICAL_NOUNS.setdefault(human, set()).add(("person", "plural"))

ADJECTIVES = {
    "artifact": set("small large old new red black".split()),
    "material": set("clean dry raw".split()),
    "animal": set("young small large calm".split()),
    "food": set("fresh hot cold".split()), "drink": set("fresh cold".split()),
    "text": set("brief detailed old new".split()), "person": set("quiet patient skilled young old tired careful".split()),
    "ensemble": set("small large skilled".split()), "music": set("fine soft loud fast slow".split()),
    "sport": {"competitive"}, "activity": set("brief long".split()),
    "discipline": set("modern advanced basic".split()), "place": set("quiet large old".split()),
    "plant": set("young small large".split()),
}
PHYSICAL = set("artifact material animal food drink text person ensemble place plant".split())
PREDICATE_OBJECTS = {
    "see": PHYSICAL, "observe": PHYSICAL, "inspect": PHYSICAL,
    "find": PHYSICAL, "need": PHYSICAL | {"activity"}, "want": PHYSICAL | {"activity"},
    "choose": PHYSICAL, "read": {"text"},
    "study": set("text discipline animal plant material activity place artifact".split()),
    "buy": set("artifact material animal food drink text plant".split()),
    "sell": set("artifact material animal food drink text plant".split()),
    "use": set("artifact material text".split()), "hire": {"person", "ensemble"}, "help": {"person"},
    "admire": set("animal artifact person ensemble place plant".split()),
    "enjoy": set("food drink sport activity discipline music text".split()),
    "play": {"sport", "music"}, "eat": {"food"}, "drink": {"drink"},
    "collect": set("artifact material text plant".split()), "clean": set("artifact material food".split()),
    "carry": set("artifact material food text plant animal".split()),
    "avoid": set("person animal food drink place activity".split()),
    "visit": {"place", "person"}, "enter": {"place"}, "join": {"ensemble"},
    "hear": set("ensemble music animal person".split()), "cook": {"food"}, "prepare": {"food", "drink"},
    "fix": {"artifact"}, "repair": {"artifact"}, "move": {"artifact", "material"},
    "handle": set("artifact material animal".split()), "write": {"text"}, "recite": {"text"},
    "discuss": set(NOUNS_KIND for NOUNS_KIND, _ in NOUNS),
}
MANNERS = {"carefully", "quietly", "slowly"}
RELATIVE_PAST = {"read", "reviewed", "edited", "printed"}
SOURCES = {"authors", "editors", "writers", "researchers"}


def noun_phrases(words, start, allowed_types=None):
    """Return independently recognized NP endpoints and typed heads."""
    found = []
    for stop in range(start + 1, min(len(words), start + 3) + 1):
        chunk = words[start:stop]
        head = chunk[-1]
        for kind, number in LEXICAL_NOUNS.get(head, ()):
            if allowed_types is not None and kind not in allowed_types:
                continue
            lead = list(chunk[:-1])
            det = lead.pop(0) if lead and lead[0] in {"a", "an", "the", "some"} else None
            if len(lead) > 1 or (lead and lead[0] not in ADJECTIVES[kind]):
                continue
            if number == "singular" and det not in {"a", "an", "the"}:
                continue
            if number != "singular" and det in {"a", "an"}:
                continue
            if det in {"a", "an"}:
                initial = lead[0] if lead else head
                expected = "an" if initial[0] in "aeiou" else "a"
                if det != expected:
                    continue
            found.append((stop, {"kind": kind, "number": number, "head": head, "determiner": det, "modifier": lead[0] if lead else None}))
    return found


def relative_ends(words, start):
    """The optional relative is parsed independently of the main predicate."""
    out = [(start, None)]
    if start >= len(words) or words[start] != "who":
        return out
    pos = start + 1
    if pos < len(words) and words[pos] in MANNERS:
        pos += 1
    if pos >= len(words) or words[pos] not in RELATIVE_PAST:
        return out
    verb = words[pos]
    for after_text, obj in noun_phrases(words, pos + 1, {"text"}):
        source_ends = [(after_text, None)]
        if after_text < len(words) and words[after_text] == "from":
            for end, source in noun_phrases(words, after_text + 1, {"person"}):
                if source["head"] in SOURCES:
                    source_ends.append((end, source))
        for after_source, source in source_ends:
            out.append((after_source, {"verb": verb, "object": obj, "source": source, "location": None}))
            if after_source < len(words) and words[after_source] in {"in", "inside", "near"}:
                for end, location in noun_phrases(words, after_source + 1, {"place"}):
                    out.append((end, {"verb": verb, "object": obj, "source": source, "location": location}))
    return out


def parse_surface(words):
    """Recognize S -> plural-human [relative] [manner] VB NP to end of text."""
    words = tuple(words)
    if not words or words[0] not in HUMANS:
        return []
    parses = []
    for pos, relative in relative_ends(words, 1):
        manner = None
        if pos < len(words) and words[pos] in MANNERS:
            manner, pos = words[pos], pos + 1
        if pos >= len(words) or words[pos] not in PREDICATE_OBJECTS:
            continue
        verb = words[pos]
        for end, obj in noun_phrases(words, pos + 1, PREDICATE_OBJECTS[verb]):
            if end != len(words):
                continue
            parses.append({"subject": words[0], "subject_number": "plural", "subject_type": "adult_human",
                           "verb": verb, "verb_form": "present_base", "valency": "transitive", "object": obj,
                           "manner": manner, "relative": relative, "relative_subject_shared": relative is not None,
                           "complete": True, "agreement_ok": True, "selection_ok": True})
    return parses
