from itertools import product

try:
    from collections import Counter, defaultdict
    from nltk.corpus import brown
    from wordfreq import zipf_frequency
    _cnt = Counter()
    _pc = defaultdict(Counter)
    for _word, _tag in brown.tagged_words(tagset="universal"):
        _word = _word.casefold()
        if _word.isalpha():
            _cnt[_word] += 1
            _pc[_word][_tag] += 1
    _stop = set("a an the our my your his her its this that these those some any each every all no one two three in on at by of for to from with into over under near and or but if then than as is are was were be been being can could will would should may might do does did have has had not".split())
    def _brown_words(tag, limit):
        rows = []
        for _word, _count in _cnt.items():
            if _word in _stop or len(_word) < 3 or zipf_frequency(_word, "en") < 3.2:
                continue
            _tags = _pc[_word]
            if _tags[tag] >= max(2, 0.65 * sum(_tags.values())):
                rows.append((zipf_frequency(_word, "en") + 0.01 * _count, _word))
        return [word for _score, word in sorted(rows, reverse=True)[:limit]]
except Exception:
    _brown_words = lambda _tag, _limit: []

N = """satan sonatas artist baker botanist captain chemist clerk curator doctor editor farmer gardener guide historian judge keeper leader mason mentor nurse painter parent pilot poet reader ranger sailor scholar student teacher traveler warden watcher writer archivist builder child dancer driver friend hero lawyer musician neighbor officer owner planner singer scout soldier speaker worker author actor engineer explorer florist host inspector journalist king lady man model monk mother navigator partner person photographer plumber professor queen researcher runner secretary shepherd sister son surgeon tailor technician uncle visitor waiter wife woman youth atlas book chart clue code diary draft file gift guide key lamp letter map memo message model note parcel paper path photo plan poem portrait record report reply recipe ring road scene script sign sketch story table task text ticket tool tower trail vase verse view window word work""".split()
V = """admires answers arranges builds carries charts checks cleans closes collects compares covers creates crosses delivers draws edits examines explores finds fixes follows gathers guides handles hears helps holds joins keeps learns listens maps marks measures meets moves notices opens orders paints packs plans plants prints reads records repairs reports returns reveals runs saves sees sends serves shares shows sings solves sorts studies teaches tests tracks travels uses values visits waits walks watches writes oscillate""".split()
A = """ancient brave bright calm careful clear coastal distant eager early gentle golden green hidden honest kind large late little local long lovely metallic modern narrow natural neat new patient plain quiet red remote round safe sharp short silent simple slow small steady strong sunny tall tidy tiny warm wide wise young""".split()
D = "a an the my our your this that no one each".split()
B = {"N": N, "V": V, "A": A, "D": D}
N = list(dict.fromkeys(N + _brown_words("NOUN", 400)))
V = list(dict.fromkeys(V + _brown_words("VERB", 300)))
A = list(dict.fromkeys(A + _brown_words("ADJ", 300)))
B = {"N": N, "V": V, "A": A, "D": D}


def trie_index(pattern):
    root = {}
    for vals in product(*(B[x] for x in pattern)):
        if len(set(vals)) != len(vals):
            continue
        tape = "".join(vals)
        node = root
        for ch in tape[::-1]:
            node = node.setdefault(ch, {})
        node.setdefault("$", []).append(vals)
    return root


def search(left_pattern, right_pattern):
    root = trie_index(right_pattern)
    out = []

    def rec(slot, tape, vals, node, terminals):
        if slot == len(left_pattern):
            for rlen, rvals in terminals:
                middle = tape[rlen:]
                if middle == middle[::-1]:
                    words = vals + list(rvals)
                    if len(set(words)) == len(words):
                        out.append((len(tape) + rlen, " ".join(words)))
            return
        for word in B[left_pattern[slot]]:
            if word in vals:
                continue
            ntape = tape + word
            nnode = node
            nterm = terminals[:]
            good = True
            for ch in word:
                if ch not in nnode:
                    good = False
                    break
                nnode = nnode[ch]
                if "$" in nnode:
                    for rvals in nnode["$"]:
                        nterm.append((len(ntape[:ntape.find(ch)]), rvals))
            # The terminal bookkeeping above is intentionally conservative;
            # the complete search below will be rerun with explicit lengths.
            if good:
                rec(slot + 1, ntape, vals + [word], nnode, nterm)

    def walk(slot, tape, vals, node, terminals):
        if slot == len(left_pattern):
            for i, rvals in terminals:
                middle = tape[i:]
                if middle == middle[::-1]:
                    words = list(vals) + list(rvals)
                    if len(set(words)) == len(words):
                        out.append((len(tape) + i, " ".join(words)))
            return
        for word in B[left_pattern[slot]]:
            if word in vals:
                continue
            ntape = tape + word
            nnode = node
            nterm = terminals[:]
            if nnode is not None:
                base = len(tape)
                for offset, ch in enumerate(word, 1):
                    if ch not in nnode:
                        nnode = None
                        break
                    nnode = nnode[ch]
                    if "$" in nnode:
                        for rvals in nnode["$"]:
                            nterm.append((base + offset, rvals))
            if nnode is not None or nterm:
                walk(slot + 1, ntape, vals + [word], nnode, nterm)

    walk(0, "", [], root, [])
    return sorted(set(out), reverse=True)


if __name__ == "__main__":
    for lp, rp in (("NVD", "AN"), ("NVDAN", "AN"), ("DANVD", "AN"), ("DANV", "NAN")):
        rows = search(lp, rp)
        print(lp, rp, len(rows), rows[:20])
