"""Score every 37+ letter find with the gated coherence judge, best-first."""
import json, re, sys
sys.path.insert(0, ".")
from experiments.coherence_scale import ask


def main() -> None:
    texts = []
    with open("runs/graph_long.jsonl") as fh:
        for line in fh:
            texts += json.loads(line)["texts"]
    rows = []
    for text in texts:
        score, _ = ask("gpt-oss:20b", text)
        rows.append((score if score is not None else -1,
                     len(re.sub("[^a-z]", "", text)), text))
        print(f"[{rows[-1][0]}] {rows[-1][1]}L {text}", flush=True)
    rows.sort(key=lambda row: (-row[0], -row[1]))
    with open("runs/graph_long_scored.json", "w") as fh:
        json.dump(rows, fh, indent=1)
    print("\n=== best ===")
    for score, letters, text in rows[:12]:
        print(f"[{score}] {letters}L {text}")


if __name__ == "__main__":
    main()
