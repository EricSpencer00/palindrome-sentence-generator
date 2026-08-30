"""Score the punctuate-afterwards arms across judges.

Columns are judges, rows are punctuation schemes. The cross matters: `llm_20b`
scored by the 20b judge is a model marking its own homework, and the gap
between that and the same text scored by another family is the size of the
favour it does itself.
"""
import json, statistics as st, sys, os

ORDER = ["hand", "llm_120b", "llm_20b", "bare", "perword", "present"]


def main(tags):
    cols, data = [], {}
    for t in tags:
        p = f"runs/punct/after_{t}.json"
        if not os.path.exists(p):
            continue
        cols.append(t)
        by = {}
        for r in json.load(open(p)):
            if r["score"] is not None:
                by.setdefault(r["kind"], []).append(r["score"])
        data[t] = by
    if not cols:
        print("no arms scored yet")
        return
    print(f"  {'scheme':<10} " + "".join(f"{c:>10}" for c in cols) + f"{'n':>5}")
    for k in ORDER:
        cells, n = [], 0
        for c in cols:
            v = data[c].get(k, [])
            n = max(n, len(v))
            cells.append(f"{st.mean(v):>10.2f}" if v else f"{'-':>10}")
        if n:
            print(f"  {k:<10} " + "".join(cells) + f"{n:>5}")
    base = "bare"
    print(f"\n  against {base}, per judge:")
    for k in ORDER:
        if k == base:
            continue
        cells = []
        for c in cols:
            v, b = data[c].get(k, []), data[c].get(base, [])
            cells.append(f"{st.mean(v) - st.mean(b):>+10.2f}" if v and b
                         else f"{'-':>10}")
        if any(x.strip() != "-" for x in cells):
            print(f"  {k:<10} " + "".join(cells))


if __name__ == "__main__":
    main(sys.argv[1:])
