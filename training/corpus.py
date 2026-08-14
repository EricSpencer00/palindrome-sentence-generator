"""Build a word-aligned GPT-2 token stream, stored once and read in either
direction.

Two decisions here are load-bearing for everything downstream.

**Word-aligned tokenization.** Every word is tokenized independently as
`" " + word`, and the per-word token lists are concatenated. This is not what
GPT-2's BPE does to running text — merges cross word boundaries — but it makes
the token sequence a concatenation of whole-word blocks. That is what lets the
search score one word at a time against a cached context, and what makes
reversal exact: reversing a word-aligned stream yields a stream whose blocks
are still whole words. Reversing a natively-tokenized stream does not.

**One stream, read both ways.** A window of the reversed stream is exactly the
reverse of a window of the forward stream, so the backward model trains on
flipped windows of the same file. The forward and backward runs therefore see
identical data, which is the point: any difference between them is direction,
not corpus.

Documents are separated by the end-of-text token. Windows may straddle that
boundary, as they do in ordinary LM training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EOT = 50256  # GPT-2 <|endoftext|>
DTYPE = np.uint16  # GPT-2 vocab is 50257, so ids fit


def word_aligned_ids(tok, text: str) -> list[int]:
    """Token ids for `text`, tokenizing each whitespace word on its own."""
    return word_aligned_batch(tok, [text])[0]


def word_aligned_batch(tok, texts: Sequence[str]) -> list[list[int]]:
    """Word-aligned ids for many texts in one tokenizer call.

    The fast tokenizer's per-call overhead dominates when it is handed a few
    words at a time, and there are ~1.8M rows in the corpus. Pooling every
    word from a block of rows into a single call and cutting the result back
    apart by word count turns hours of tokenization into minutes.
    """
    words_per_text = [t.split() for t in texts]
    flat = [" " + w for ws in words_per_text for w in ws]
    if not flat:
        return [[] for _ in texts]
    encoded = tok(flat, add_special_tokens=False)["input_ids"]

    out, at = [], 0
    for ws in words_per_text:
        blocks = encoded[at:at + len(ws)]
        at += len(ws)
        out.append([i for block in blocks for i in block])
    return out


def build(dataset: str, split: str, out_dir: Path, limit: int | None,
          model: str = "gpt2") -> dict:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    name = "wikitext-103-raw-v1" if dataset == "wikitext" else None
    ds = load_dataset(dataset, name, split=split) if name else load_dataset(dataset, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / f"{split}.bin"

    total = 0
    words = 0
    pending: list[str] = []
    rows_per_call = 2000

    def flush(fh) -> tuple[int, int]:
        if not pending:
            return 0, 0
        blocks = word_aligned_batch(tok, pending)
        ids = [i for block in blocks if block for i in block + [EOT]]
        if not ids:
            return 0, 0
        np.array(ids, dtype=DTYPE).tofile(fh)
        return len(ids), sum(len(t.split()) for t in pending)

    with open(bin_path, "wb") as fh:
        for row in ds:
            text = row["text"].strip()
            if not text or text.startswith("="):  # wikitext section headers
                continue
            pending.append(text)
            if len(pending) >= rows_per_call:
                t, w = flush(fh)
                total += t
                words += w
                pending = []
        t, w = flush(fh)
        total += t
        words += w

    meta = {"tokens": total, "words": words, "dtype": "uint16",
            "model": model, "dataset": dataset, "split": split,
            "tokenization": "word-aligned, each word encoded as ' '+word"}
    (out_dir / f"{split}.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_stream(path: Path) -> np.ndarray:
    return np.memmap(path, dtype=DTYPE, mode="r")


def sample_window(stream: np.ndarray, start: int, length: int,
                  direction: str) -> np.ndarray:
    """A training window, flipped for the backward direction.

    Flipping the window is equivalent to taking the same window from a
    fully-reversed stream, so the two directions see the same text.
    """
    w = np.asarray(stream[start:start + length], dtype=np.int64)
    return w[::-1].copy() if direction == "backward" else w


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="wikitext")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", type=Path, default=Path("data/tokens"))
    ap.add_argument("--limit", type=int, default=None,
                    help="cap on source rows, for a quick local build")
    ap.add_argument("--model", default="gpt2")
    args = ap.parse_args()

    meta = build(args.dataset, args.split, args.out, args.limit, args.model)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
