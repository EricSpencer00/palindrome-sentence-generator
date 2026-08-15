"""Reversal stability: the second half of a word-order paragraph is real text.

Symmetric assembly guarantees the form — "A verb B" in the first half arrives
as "B verb A" in the second. It guarantees nothing about sense. Measured with
GPT-2 over the shipped banks, 67 of 110 units score *worse* reversed (mean
-0.078 logprob per letter, 17 of them below -0.40), and the failures share a
shape: an animacy asymmetry. "Widows lit lamps" reverses into "Lamps lit
widows"; "Priest recited prayers" into "Prayers recited priest". The units
that survive the mirror carry reciprocal or competitive verbs — outlived,
outran, taught, guarded, replaced — where both directions are sayable.

So half of every paragraph was being filled with grammatical falsehoods. These
tests pin the fix: measure the gap once, store it beside the bank, and let
selection prefer units whose mirror still means something.
"""
import json
import re
import statistics
from pathlib import Path

import pytest

from llm_palindrome.reversal import drop_worst, reverse_unit

BANKS = json.loads(Path("data/word_banks.json").read_text())
NAMES = sorted(BANKS)


class TestReverseUnit:
    def test_swaps_word_order(self):
        assert reverse_unit("wind chased dust") == "dust chased wind"

    def test_is_its_own_inverse(self):
        unit = "mothers buried sons"
        assert reverse_unit(reverse_unit(unit)) == unit

    def test_matches_how_the_assembler_mirrors_a_unit(self):
        """The gap must be measured on the string a reader actually sees.

        `word_assemble` reverses the whole word sequence, so a unit's mirror
        image is that unit's own words reversed. Measuring anything else would
        score a sentence the paragraph never contains.
        """
        from llm_palindrome.paragraphs import word_assemble

        unit = "cables fed instruments"
        text = word_assemble([unit], "night follows night")
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        assert sentences[-1].lower() == reverse_unit(unit)


class TestDropWorst:
    GAPS = {
        "a x b": 0.30,
        "c x d": 0.10,
        "e x f": -0.10,
        "g x h": -0.90,
    }

    def test_removes_the_least_stable_first(self):
        kept = drop_worst(list(self.GAPS), self.GAPS, want=2, fraction=0.25)
        assert "g x h" not in kept

    def test_keeps_every_unit_when_nothing_is_dropped(self):
        kept = drop_worst(list(self.GAPS), self.GAPS, want=4, fraction=0.0)
        assert sorted(kept) == sorted(self.GAPS)

    def test_never_starves_the_request(self):
        """Stability is a preference, not a quota — a big ask keeps the bank."""
        kept = drop_worst(list(self.GAPS), self.GAPS, want=4, fraction=0.75)
        assert len(kept) == 4

    def test_an_unmeasured_unit_is_not_silently_dropped(self):
        """A new bank unit lands before the next measurement run.

        Treating "no measurement" as "worst" would make every bank edit
        invisible until someone reran GPT-2. It ranks as neutral instead.
        """
        units = list(self.GAPS) + ["fresh x unit"]
        kept = drop_worst(units, self.GAPS, want=2, fraction=0.5)
        assert "fresh x unit" in kept


@pytest.mark.parametrize("name", NAMES)
class TestBanksCarryMeasurements:
    def test_every_outer_unit_has_a_measured_gap(self, name):
        bank = BANKS[name]
        gaps = bank.get("reversal", {})
        missing = [u for u in bank["outer"] if u.lower() not in gaps]
        assert not missing, missing

    def test_gaps_are_numbers(self, name):
        for unit, gap in BANKS[name].get("reversal", {}).items():
            assert isinstance(gap, (int, float)), (unit, gap)


@pytest.mark.parametrize("name", NAMES)
class TestSelectionPrefersStableUnits:
    def mean_gap(self, name, units):
        gaps = BANKS[name]["reversal"]
        return statistics.mean(gaps[u.lower()] for u in units)

    def test_a_default_request_beats_the_bank_average(self, name):
        """The point of the whole exercise, measured on real requests."""
        from server.v2 import select_units

        bank = BANKS[name]
        baseline = self.mean_gap(name, bank["outer"])
        picked = [
            self.mean_gap(
                name,
                select_units(list(bank["outer"]), 18, nonce, bank["reversal"]))
            for nonce in range(40)
        ]
        assert statistics.mean(picked) > baseline

    def test_the_endpoint_actually_passes_the_measurements_in(self, name):
        """Without this the wiring could be missing and every other test pass.

        Reads the units back out of a rendered paragraph — the first `pairs`
        sentences are the forward half — and holds them to the same bound.
        """
        from server.v2 import word_paragraph

        bank = BANKS[name]
        baseline = self.mean_gap(name, bank["outer"])
        means = []
        for nonce in range(20):
            out = word_paragraph(bank, sentences=18, nonce=nonce)
            said = [s.strip().lower()
                    for s in out["text"].split(".") if s.strip()]
            means.append(self.mean_gap(name, said[:out["pairs"]]))
        assert statistics.mean(means) > baseline

    def test_selection_still_varies_between_requests(self, name):
        """Ranking by stability must not collapse the bank to one paragraph."""
        from server.v2 import select_units

        bank = BANKS[name]
        seen = {
            tuple(select_units(list(bank["outer"]), 18, nonce, bank["reversal"]))
            for nonce in range(20)
        }
        assert len(seen) >= 15

    def test_the_arc_survives_the_filter(self, name):
        """Stakes-bearing units are often the least stable. Keep them anyway."""
        from server.v2 import ARC_WORDS, select_units

        bank = BANKS[name]
        for nonce in range(10):
            picked = select_units(list(bank["outer"]), 18, nonce,
                                  bank["reversal"])
            arc = [u for u in picked if ARC_WORDS & set(u.lower().split())]
            assert arc, (name, nonce)


class TestMeasurementScript:
    def test_it_reproduces_the_stored_keys(self):
        """The stored gaps must be keyed the way the reader looks them up."""
        from training.measure_reversal import gap_key

        for name in NAMES:
            for unit in BANKS[name]["outer"]:
                assert gap_key(unit) in BANKS[name]["reversal"]

    def test_the_key_is_case_and_space_insensitive(self):
        from training.measure_reversal import gap_key

        assert gap_key("  Wind  Chased Dust ") == gap_key("wind chased dust")


def test_shipped_banks_are_mostly_reversal_stable():
    """A regression bound on the banks themselves.

    Measured -0.078 mean, 82/110 above -0.30. The bound sits just below that so
    a bank edit that makes the material worse fails here.

    It deliberately does not demand better. Rewriting the units to raise this
    number is work on the WORD-ORDER mode, which is not this project's
    deliverable — the letters do not mirror there. See docs/training.md on the
    distinction. Selection uses the measurement; nobody should curate toward
    it.
    """
    every = [g for n in NAMES for g in BANKS[n]["reversal"].values()]
    assert statistics.mean(every) > -0.10
    stable = sum(1 for g in every if g > -0.30)
    assert stable / len(every) >= 0.70, f"{stable}/{len(every)}"


def test_units_read_as_sentences_in_both_directions():
    """No unit may reverse into a repeated word — 'night held night' reads as
    a typo rather than a mirror."""
    for name in NAMES:
        for unit in BANKS[name]["outer"]:
            words = re.findall(r"[a-z]+", unit.lower())
            assert len(set(words)) == len(words), (name, unit)
