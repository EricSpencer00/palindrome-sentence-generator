"""v3 serves one palindrome. These tests are about what "one palindrome" means.

The endpoint makes four promises and each is checked here rather than trusted:
the thing is a palindrome, the words are words, the punctuation is free, and
the provenance is not overstated. The first is the one that would be worst to
get wrong, so it is checked on the served string and not on the plain one, and
on many draws rather than one.

`grow` is exercised but not endorsed. Two blind annotators preferred the
ungrown seed on 20 of 20 pairs, so the tests assert it stays off by default and
that it does not break the mirror when asked for.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llm_palindrome.shortwords import is_real_short
from llm_palindrome.validator import is_palindrome, normalize
from server import v3


@pytest.fixture(scope="module")
def client():
    from server.app import app
    return TestClient(app)


@pytest.fixture(scope="module")
def bank():
    v3.ensure_loaded()
    assert not v3._load_error, v3._load_error
    return v3._bank


# ------------------------------------------------------------ the one promise

def test_every_bank_entry_is_a_palindrome(bank):
    """The bank is verified on load, so nothing here can be taken on trust."""
    assert bank, "bank is empty"
    for row in bank:
        assert is_palindrome(row["text"]), row["text"]


def test_served_text_is_a_palindrome_across_many_draws(client):
    """Check the PRESENTED string, which is what a caller actually receives.

    Presentation adds capitals and punctuation. Those are invisible to the
    mirror by design, and a bug in `present` would be invisible in the plain
    field and fatal in the served one.
    """
    for seed in range(60):
        r = client.get(f"/api/v3/palindrome?seed={seed}")
        assert r.status_code == 200, r.text
        body = r.json()
        assert is_palindrome(body["text"]), body["text"]
        assert is_palindrome(body["plain"])
        assert normalize(body["text"]) == normalize(body["plain"])


def test_reported_letter_count_matches(client):
    body = client.get("/api/v3/palindrome?seed=7").json()
    assert body["letters"] == len(normalize(body["plain"]))
    assert body["words"] == len(body["plain"].split())


# ------------------------------------------------------------------ real words

def test_no_junk_short_words_are_served(client):
    """The corner where a search cheats: "bn", "cu", "eb" all fit any overhang."""
    for seed in range(40):
        body = client.get(f"/api/v3/palindrome?seed={seed}").json()
        for word in body["plain"].split():
            assert is_real_short(word), f"{word!r} in {body['plain']!r}"


def test_every_word_is_in_the_lexicon(client):
    from llm_palindrome.lexicon import is_real_word
    lex = v3.lexicon()
    for seed in range(40):
        body = client.get(f"/api/v3/palindrome?seed={seed}").json()
        for word in body["plain"].split():
            assert is_real_word(word, lex), f"{word!r} not a word"


def test_real_words_rejects_what_it_should():
    assert v3.real_words(["level", "madam"])
    assert not v3.real_words(["bn", "level"])      # junk two-letter
    assert not v3.real_words(["utc"])              # in frequency lists, not English
    assert not v3.real_words(["ab3"])              # not alphabetic


# -------------------------------------------------------------------- chunks

def test_chunks_reassemble_to_the_text(client):
    """"Assembled from chunks" has to be checkable, not asserted."""
    for seed in range(25):
        body = client.get(f"/api/v3/palindrome?seed={seed}").json()
        joined = " ".join(c["text"] for c in body["chunks"])
        assert normalize(joined) == normalize(body["plain"])


def test_chunk_halves_mirror_each_other(client):
    """The left and right chunks are each other's letters reversed."""
    for seed in range(25):
        body = client.get(f"/api/v3/palindrome?seed={seed}").json()
        roles = {c["role"]: c["text"] for c in body["chunks"]}
        if "left" not in roles or "right" not in roles:
            continue
        left, right = normalize(roles["left"]), normalize(roles["right"])
        assert left == right[::-1], body["plain"]


def test_centre_chunk_is_at_most_one_unit(client):
    for seed in range(25):
        body = client.get(f"/api/v3/palindrome?seed={seed}").json()
        centres = [c for c in body["chunks"] if c["role"] == "centre"]
        assert len(centres) <= 1


# --------------------------------------------------------------- determinism

def test_seed_fixes_the_choice(client):
    a = client.get("/api/v3/palindrome?seed=99").json()
    b = client.get("/api/v3/palindrome?seed=99").json()
    assert a["plain"] == b["plain"]


def test_different_seeds_reach_different_palindromes(client):
    seen = {client.get(f"/api/v3/palindrome?seed={s}").json()["plain"]
            for s in range(30)}
    assert len(seen) > 5, "the endpoint is serving one or two entries"


# --------------------------------------------------------------- provenance

def test_novel_is_the_default_and_excludes_the_catalogue(client):
    """A palindrome that reads because somebody else wrote it is the shortcut
    `docs/NORTH-STAR.md` exists to name. The default must not take it."""
    for seed in range(30):
        body = client.get(f"/api/v3/palindrome?seed={seed}").json()
        assert body["source"] == "generated", body

    got = {client.get(f"/api/v3/palindrome?seed={s}&novel=false").json()["source"]
           for s in range(60)}
    assert "catalogue" in got, "novel=false should reach catalogued entries"


def test_min_letters_is_respected(client):
    for seed in range(20):
        body = client.get(
            f"/api/v3/palindrome?seed={seed}&min_letters=24").json()
        assert body["letters"] >= 24


def test_impossible_constraints_404_rather_than_serving_something_else(client):
    r = client.get("/api/v3/palindrome?min_letters=200")
    assert r.status_code == 404


# ------------------------------------------------------------------- growth

def test_growth_is_off_by_default(client):
    body = client.get("/api/v3/palindrome?seed=3").json()
    assert body["grown"] is False
    assert body["operations"] == []


def test_growth_lengthens_and_still_mirrors(client):
    plain = client.get("/api/v3/palindrome?seed=3").json()
    grown = client.get("/api/v3/palindrome?seed=3&grow=3").json()
    assert is_palindrome(grown["text"])
    assert grown["letters"] >= plain["letters"]
    if grown["operations"]:
        assert grown["grown"] is True


def test_the_response_says_growth_reads_worse(client):
    """The endpoint publishes the finding that argues against its own option."""
    body = client.get("/api/v3/palindrome?seed=1").json()
    assert "20 of 20" in body["notes"]["growth"]


# -------------------------------------------------------------------- health

def test_health_reports_the_bank(client):
    body = client.get("/api/v3/health").json()
    assert body["ok"] is True
    assert body["version"] == 3
    assert body["bank"] == body["generated"] + body["catalogue"]
    assert body["generated"] > 0


def test_bank_file_and_loaded_bank_agree(bank):
    raw = json.loads(Path(v3.BANK_PATH).read_text())
    assert len(bank) <= len(raw)
    assert all(is_palindrome(r["text"]) for r in raw)


# ------------------------------------------------------------- composition
#
# Length is free under mirror-pair nesting, so these tests are about the two
# properties that are NOT free: that nothing repeats, and that the thing is
# still a palindrome at every point on the slider.

LENGTHS = [80, 400, 1200, 3000, 8000, 14000]


@pytest.mark.parametrize("letters", LENGTHS)
def test_composition_is_a_palindrome_at_every_length(client, letters):
    r = client.get(f"/api/v3/composition?seed=4&letters={letters}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert is_palindrome(body["text"]), body["text"][:120]
    assert normalize(body["text"]) == normalize(body["plain"])


@pytest.mark.parametrize("letters", LENGTHS)
def test_no_chunk_ever_repeats(client, letters):
    """The property the whole scheme exists for.

    A sequence of self-palindromic units is a palindrome only when the unit
    sequence is one, forcing unit k to equal unit n+1-k. Mirror-pairs avoid
    that, and this is the assertion that they actually did.
    """
    body = client.get(f"/api/v3/composition?seed=4&letters={letters}").json()
    texts = [c["text"] for c in body["chunks"]]
    assert body["repeats"] == 0
    assert len(set(texts)) == len(texts)


def test_no_repeats_across_many_seeds(client):
    for seed in range(12):
        body = client.get(
            f"/api/v3/composition?seed={seed}&letters=2000").json()
        assert body["repeats"] == 0, body["chunks"]


def test_left_and_right_chunks_mirror_pairwise(client):
    """Chunk i from the left must be chunk i from the right, reversed."""
    body = client.get("/api/v3/composition?seed=6&letters=1200").json()
    left = [c["text"] for c in body["chunks"] if c["role"] == "left"]
    right = [c["text"] for c in body["chunks"] if c["role"] == "right"]
    assert len(left) == len(right) == body["pairs"]
    for a, b in zip(left, reversed(right)):
        assert normalize(a) == normalize(b)[::-1], (a, b)


def test_chunks_reassemble_to_the_composition(client):
    body = client.get("/api/v3/composition?seed=6&letters=1200").json()
    joined = " ".join(c["text"] for c in body["chunks"])
    assert normalize(joined) == normalize(body["plain"])


def test_exactly_one_centre(client):
    body = client.get("/api/v3/composition?seed=6&letters=1200").json()
    assert sum(1 for c in body["chunks"] if c["role"] == "centre") == 1


def test_length_slider_tracks_the_request(client):
    got = []
    for want in (200, 800, 2400, 6000):
        body = client.get(f"/api/v3/composition?seed=2&letters={want}").json()
        assert body["letters"] <= want
        got.append(body["letters"])
    assert got == sorted(got), got
    assert got[-1] > got[0] * 4


def test_capacity_is_reported_and_is_the_ceiling(client):
    cap = client.get("/api/v3/health").json()["capacity"]["novel"]["max_letters"]
    body = client.get(f"/api/v3/composition?seed=2&letters={cap + 5000}").json()
    assert body["letters"] <= cap
    assert body["capacity_letters"] <= cap


def test_chops_caps_the_pair_count(client):
    body = client.get(
        "/api/v3/composition?seed=2&letters=4000&chops=15").json()
    assert body["pairs"] <= 15


def test_longest_first_uses_fewer_pairs_for_the_same_length(client):
    a = client.get("/api/v3/composition?seed=2&letters=1200").json()
    b = client.get(
        "/api/v3/composition?seed=2&letters=1200&longest_first=true").json()
    assert b["pairs"] <= a["pairs"]


def test_every_word_in_a_composition_is_a_real_word(client):
    from llm_palindrome.lexicon import is_real_word
    lex = v3.lexicon()
    body = client.get("/api/v3/composition?seed=8&letters=2000").json()
    for word in body["plain"].split():
        assert is_real_word(word, lex) and is_real_short(word), word


def test_no_degenerate_pair_is_used(client):
    """A half that is itself a palindrome mirrors to itself and would appear
    at both mirrored positions. `no is ice decision` is one such."""
    body = client.get("/api/v3/composition?seed=3&letters=14000").json()
    for c in body["chunks"]:
        if c["role"] == "centre":
            continue
        letters = normalize(c["text"])
        assert letters != letters[::-1], c["text"]


def test_novel_default_excludes_the_catalogue(client):
    body = client.get("/api/v3/composition?seed=5&letters=1200").json()
    assert all(c["source"] == "generated" for c in body["chunks"])


# ----------------------------------------------------------- your own centre

def test_your_palindrome_becomes_the_centre(client):
    """The one slot an arbitrary palindrome can occupy without breaking the
    mirror, because it is the only position not fixed by an opposite number."""
    mine = "a man a plan a canal panama"
    body = client.get(
        f"/api/v3/composition?seed=1&letters=600&centre={mine.replace(' ', '+')}"
    ).json()
    centre = [c for c in body["chunks"] if c["role"] == "centre"]
    assert len(centre) == 1
    assert normalize(centre[0]["text"]) == normalize(mine)
    assert centre[0]["source"] == "yours"
    assert body["centre_is_yours"] is True


def test_your_centre_still_leaves_the_whole_a_palindrome(client):
    for mine in ("racecar", "a man a plan a canal panama", "no lemon no melon"):
        body = client.get(
            f"/api/v3/composition?seed=2&letters=800&centre={mine.replace(' ', '+')}"
        ).json()
        letters = normalize(body["plain"])
        assert letters == letters[::-1], mine


def test_a_non_palindrome_is_refused_not_repaired(client):
    """Trimming a near-miss into a real one would hand back something the
    visitor did not write and present it as theirs."""
    r = client.get("/api/v3/composition?seed=1&letters=400&centre=hello+world")
    assert r.status_code == 400
    assert "not a palindrome" in r.json()["detail"]


def test_a_centre_with_no_letters_is_refused(client):
    r = client.get("/api/v3/composition?seed=1&letters=400&centre=%21%21%21")
    assert r.status_code == 400


def test_an_empty_centre_falls_back_to_the_bank(client):
    body = client.get("/api/v3/composition?seed=1&letters=400&centre=").json()
    assert body["centre_is_yours"] is False
