"""Check bidirectional constraint propagation against a tiny exhaustive oracle."""
import itertools
import unittest
from unittest.mock import patch

from experiments import astra_bidirectional_half_tape_20260920 as pilot


class HalfTapeTests(unittest.TestCase):
    def test_matches_exhaustive_typed_oracle(self):
        e = pilot.Edge
        domains = {
            "S": (e("an aide", "sg", "person"), e("some men", "pl", "person")),
            "V": (e("rips", "sg", "document"), e("inspire", "pl", "person")),
            "O": (e("nine memos", "pl", "document"), e("Diana", "sg", "person", True)),
            "B": (e(";"),),
        }
        frame = ("S0", "V0", "O0", "B", "S1", "V1", "O1")
        oracle = set()
        for path in itertools.product(*(domains[s[0]] for s in frame)):
            if any(path[s].number != path[v].number or path[v].kind != path[o].kind
                   for s, v, o in ((0, 1, 2), (4, 5, 6))):
                continue
            contents = [word for edge in path for word in edge.content]
            if len(contents) != len(set(contents)) or sum(e.proper for e in path) > 1:
                continue
            tape = "".join(e.tape for e in path)
            if len(tape) == 38 and tape == tape[::-1]:
                oracle.add(tape)
        with patch.dict(pilot.DOMAINS, domains, clear=True):
            rows, stats = pilot.solve(frame, 38)
        self.assertEqual(len(oracle), 2)
        self.assertEqual({pilot.normalize_letters(r["text"]) for r in rows}, oracle)
        self.assertFalse(stats["budget_exhausted"])
        self.assertTrue(all(r["two_pointer_exact"] and r["mechanically_admitted"] for r in rows))

    def test_audit_rejects_changed_letter(self):
        row = pilot.audit("An aide rips nine memos; some men inspire Dianb.", minimum=38)
        self.assertFalse(row["two_pointer_exact"])
        self.assertNotEqual(row["sha256_forward"], row["sha256_reverse"])
        self.assertFalse(row["mechanically_admitted"])

    def test_budget_is_a_hard_bound(self):
        _, stats = pilot.solve(pilot.FRAMES["joined"], 40, budget=10)
        self.assertEqual(stats["edge_attempts"], 10)
        self.assertTrue(stats["budget_exhausted"])


if __name__ == "__main__":
    unittest.main()
