"""Score the presenter-cost arm. Same letters throughout; only marks vary."""
import json, re, statistics as st, sys


def norm(s):
    return re.sub(r"[^a-z]", "", s.lower())

ORDER = ["hand", "perword", "bare", "present"]


def main(tag):
    rows = json.load(open(f"runs/punct/pcost_{tag}.json"))
    by = {}
    for r in rows:
        by.setdefault(r["kind"], []).append(r["score"])
    print(f"\n### {tag}")
    print(f"  {'variant':<9} {'n':>3} {'mean':>6} {'sd':>5}   {'0':>3} {'1':>3} {'2':>3} {'3':>3}")
    for k in ORDER:
        v = [x for x in by.get(k, []) if x is not None]
        if not v:
            continue
        h = [sum(1 for x in v if x == d) for d in range(4)]
        print(f"  {k:<9} {len(v):>3} {st.mean(v):>6.2f} "
              f"{st.stdev(v) if len(v)>1 else 0:>5.2f}   "
              f"{h[0]:>3} {h[1]:>3} {h[2]:>3} {h[3]:>3}")
    m = {k: [x for x in by.get(k, []) if x is not None] for k in ORDER}
    if m["hand"] and m["present"]:
        print(f"\n  hand - present = {st.mean(m['hand']) - st.mean(m['present']):+.2f}"
              "   what our punctuation costs on this material")
    if m["hand"] and m["bare"]:
        print(f"  hand - bare    = {st.mean(m['hand']) - st.mean(m['bare']):+.2f}"
              "   what good punctuation is worth at all")
    if m["present"] and m["bare"]:
        print(f"  present - bare = {st.mean(m['present']) - st.mean(m['bare']):+.2f}"
              "   whether ours beats doing nothing")
    # Paired, since every text appears in all three variants. The three
    # variants of one palindrome have identical LETTERS and different marks,
    # so the letters are the join key.
    if m["hand"] and m["present"]:
        idx = {}
        for r in rows:
            idx.setdefault(r["kind"], []).append(r)
        print("\n  paired by text (hand vs present):")
        hs = {norm(r["text"]): r for r in idx["hand"]}
        won = tie = lost = 0
        for r in idx["present"]:
            k = norm(r["text"])
            o = hs.get(k)
            if o is None or o["score"] is None or r["score"] is None:
                continue
            won += o["score"] > r["score"]
            tie += o["score"] == r["score"]
            lost += o["score"] < r["score"]
        print(f"    hand better {won}, tie {tie}, present better {lost}")


if __name__ == "__main__":
    for t in sys.argv[1:]:
        main(t)
