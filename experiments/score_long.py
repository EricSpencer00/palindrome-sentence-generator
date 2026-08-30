"""Score every 37+ letter find with the gated coherence judge, best-first."""
import json, re, sys
sys.path.insert(0, ".")
from experiments.coherence_scale import ask

texts = []
for line in open("runs/graph_long.jsonl"):
    d = json.loads(line)
    texts += d["texts"]
rows = []
for t in texts:
    s, _ = ask("gpt-oss:20b", t)
    rows.append((s if s is not None else -1, len(re.sub("[^a-z]", "", t)), t))
    print(f"[{rows[-1][0]}] {rows[-1][1]}L {t}", flush=True)
rows.sort(key=lambda r: (-r[0], -r[1]))
json.dump(rows, open("runs/graph_long_scored.json", "w"), indent=1)
print("\n=== best ===")
for s, L, t in rows[:12]:
    print(f"[{s}] {L}L {t}")
