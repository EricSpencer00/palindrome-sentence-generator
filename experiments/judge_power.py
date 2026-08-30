"""Does a candidate judge separate prose from its own shuffle?

A judge with no discriminative power agrees with everything, and an arm scored
by one reads as "all schemes are equal" rather than as "this judge is useless".
llama3.1:8b scored six punctuation schemes at 2.00, 2.00, 2.00, 2.00, 2.04 and
2.08, which is the second thing and looks like the first.

So candidate judges are gated here before being used: 20 WikiText spans against
those same 20 spans shuffled, raw. A judge that cannot put prose above word
salad cannot be asked about anything subtler.
"""
import random, statistics as st, sys

sys.path.insert(0, ".")
from experiments.coherence_scale import ask, prose_spans_raw

if __name__ == "__main__":
    rng = random.Random(3)
    spans, raw = prose_spans_raw(20, 24, rng)
    shuf = []
    for w in spans:
        s = list(w); rng.shuffle(s)
        shuf.append(" ".join(s))
    for model in sys.argv[1:]:
        p = [ask(model, t)[0] for t in raw]
        q = [ask(model, t)[0] for t in shuf]
        p = [x for x in p if x is not None]
        q = [x for x in q if x is not None]
        gap = st.mean(p) - st.mean(q)
        # 0.5 of a point on a 0-3 scale is a judgement call, not a derived
        # threshold, and floating point puts a nominal 0.50 gap on either side
        # of it. Anything within 0.05 is reported as borderline rather than
        # decided by rounding.
        # The mean gap is the weaker criterion. Claude Haiku 4.5 passed it at
        # +0.60 and then reversed a result that blind pairwise judging settled
        # 52/52 the other way, because it rates 70% of the shuffled salad at 2
        # or above while gpt-oss rates 0% of it that way. A judge that calls
        # word salad English has no room left to see a difference. Both are
        # reported; the salad rate is the one to believe.
        salad = sum(1 for x in q if x >= 2) / len(q)
        verdict = ("USABLE" if gap >= 0.55 and salad <= 0.25 else
                   "LENIENT" if gap >= 0.45 else "NO POWER")
        print(f"{model:<24} prose {st.mean(p):.2f}  shuffle {st.mean(q):.2f}"
              f"  gap {gap:+.2f}  salad-called-English {salad:.2f}"
              f"  {verdict}", flush=True)
