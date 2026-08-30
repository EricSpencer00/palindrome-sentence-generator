"""Score an arm of oss_judge.py against the blind human verdicts.

Reported per test set and per arm:

  accuracy    agreement with the key, pooling both orientations, so a
              27-item set contributes 54 decisions exactly as the two human
              judges did
  self-agree  the same item answered the same way with the sides swapped.
              This is the position control: 0/n means the model is answering
              by position and its accuracy is meaningless; n/n means it is
              reading the text
  left rate   how often it picked whichever passage was printed first
"""
import json, math, sys

FLIP = {"A": "B", "B": "A"}


def binom(c, n):
    return sum(math.comb(n, i) for i in range(c, n + 1)) / 2 ** n


def load_key(path, field):
    return {k["id"]: (k["kind"], k[field]) for k in json.load(open(path))}


# The key stores the side of one named condition; the side the HUMAN judges
# actually chose is that side for the seam set (they preferred the single
# chunk, unanimously) and the opposite side for the grow set (they preferred
# the seed, 20/20). Scoring against the stored side rather than the human
# consensus reads a perfect agreement as a perfect disagreement, so the
# inversion is recorded here once and applied everywhere.
SETS = {
    "seam": ("runs/punct/seam_key.json", "single_side", False),
    "grow": ("runs/punct/grow_key.json", "grown_side", True),
}


def main(tag):
    picks = json.load(open(f"runs/punct/oss_{tag}.json"))
    print(f"\n### {tag}")
    for setname, (keypath, field, invert) in SETS.items():
        key = load_key(keypath, field)
        if invert:
            key = {i: (k, FLIP[v] if k != "calibration" else v)
                   for i, (k, v) in key.items()}
        fwd = {int(i): v for i, v in picks[f"{setname}.fwd"].items()}
        rev = {int(i): v for i, v in picks[f"{setname}.rev"].items()}
        print(f"\n  {setname}")
        print(f"    {'arm':<16} {'n':>3}  {'agrees w/ human':>16}  "
              f"{'p':>6}  {'self-agree':>10}  {'left':>7}")
        for kind in dict.fromkeys(k for k, _ in key.values()):
            ids = [i for i in key if key[i][0] == kind]
            # rev picks are flipped back into the forward frame before scoring,
            # so both orientations are counted in the same coordinate system.
            dec = [(i, fwd[i]) for i in ids if fwd[i]] + \
                  [(i, FLIP[rev[i]]) for i in ids if rev[i]]
            hit = sum(p == key[i][1] for i, p in dec)
            same = sum(1 for i in ids if fwd[i] and rev[i]
                       and fwd[i] == FLIP[rev[i]])
            left = sum(1 for i in ids if fwd[i] == "A") + \
                   sum(1 for i in ids if rev[i] == "A")
            print(f"    {kind:<16} {len(ids):>3}  "
                  f"{hit:>7}/{len(dec):<8}  {binom(hit, len(dec)):>6.3f}  "
                  f"{same:>6}/{len(ids):<3}  {left:>3}/{2*len(ids):<3}")
        miss = sum(1 for d in (fwd, rev) for v in d.values() if v is None)
        if miss:
            print(f"    unparsed replies: {miss}")


if __name__ == "__main__":
    for t in sys.argv[1:]:
        main(t)
