"""Score an absolute-coherence arm: does the scale have power, and where does
our output land relative to the palindromes people wrote?"""
import json, statistics as st, sys

ORDER = ["prose", "prose_pres", "human", "chunk", "k2", "k4", "k8", "shuffle", "shuf_pres"]


def main(tag):
    rows = json.load(open(f"runs/punct/scale_{tag}.json"))
    by = {}
    for r in rows:
        by.setdefault(r["kind"], []).append(r["score"])
    print(f"\n### {tag}")
    print(f"  {'arm':<9} {'n':>3} {'mean':>6} {'sd':>5}   "
          f"{'0':>3} {'1':>3} {'2':>3} {'3':>3}   miss")
    for k in ORDER:
        v = [x for x in by.get(k, []) if x is not None]
        miss = sum(1 for x in by.get(k, []) if x is None)
        if not v:
            continue
        hist = [sum(1 for x in v if x == d) for d in range(4)]
        sd = st.stdev(v) if len(v) > 1 else 0.0
        print(f"  {k:<9} {len(v):>3} {st.mean(v):>6.2f} {sd:>5.2f}   "
              f"{hist[0]:>3} {hist[1]:>3} {hist[2]:>3} {hist[3]:>3}   {miss}")
    p = [x for x in by.get("prose", []) if x is not None]
    s = [x for x in by.get("shuffle", []) if x is not None]
    if p and s:
        # The whole arm rests on this gap. If prose and its own shuffle score
        # the same, nothing else in the table is evidence about anything.
        print(f"\n  POWER  prose - shuffle = {st.mean(p) - st.mean(s):+.2f}")
    pp = [x for x in by.get("prose_pres", []) if x is not None]
    if p and pp:
        print(f"  PRESENTER COST  prose - prose presented = "
              f"{st.mean(p) - st.mean(pp):+.2f}")
    h = [x for x in by.get("human", []) if x is not None]
    c = [x for x in by.get("chunk", []) if x is not None]
    if h and c:
        print(f"  human  - our chunk    = {st.mean(h) - st.mean(c):+.2f}")
    k1 = c
    k2 = [x for x in by.get("k2", []) if x is not None]
    if k1 and k2:
        print(f"  chunk  - k2 (one seam)= {st.mean(k1) - st.mean(k2):+.2f}"
              "   <- the seam effect, if the scale sees it")


if __name__ == "__main__":
    for t in sys.argv[1:]:
        main(t)
