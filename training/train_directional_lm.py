"""Fine-tune GPT-2 to read English in one direction or the other.

`--direction forward` is the control and `--direction backward` is the model
under test. They differ in exactly one line — whether the training window is
flipped — so a difference in the palindrome results cannot be attributed to the
corpus, the tokenization, the schedule, or the step count.

The control matters more than it looks. Word-aligned tokenization (see
corpus.py) is already a shift away from what GPT-2 was pretrained on, so an
off-the-shelf GPT-2 is not a fair baseline for a model fine-tuned this way.
The forward run is.

Single process or torchrun DDP; the latter is what the cluster job uses.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from corpus import load_stream, sample_window


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", 1)) > 1


def setup() -> tuple[torch.device, int, int]:
    if is_distributed():
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        return torch.device("cuda", local), rank, dist.get_world_size()
    if torch.cuda.is_available():
        return torch.device("cuda"), 0, 1
    if torch.backends.mps.is_available():
        return torch.device("mps"), 0, 1
    return torch.device("cpu"), 0, 1


def batches(stream, batch_size, seq_len, direction, seed, rank, world):
    """Infinite stream of (batch, seq_len) windows, disjoint across ranks."""
    rng = np.random.default_rng(seed + rank)
    high = len(stream) - seq_len - 1
    while True:
        starts = rng.integers(0, high, size=batch_size)
        yield np.stack([sample_window(stream, int(s), seq_len + 1, direction)
                        for s in starts])


def causal_loss(model, w: torch.Tensor) -> torch.Tensor:
    """Next-token cross-entropy, shifted exactly once.

    Passing `labels=` to a Hugging Face causal head makes it shift internally,
    so handing it labels that were already shifted trains the model to predict
    two tokens ahead. That does not crash — it just starts near chance and
    converges to something that is not a language model. Doing the shift here
    makes it visible and version-independent.
    """
    logits = model(input_ids=w[:, :-1]).logits
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), w[:, 1:].reshape(-1))


def evaluate(model, stream, args, device, batches_n: int = 20) -> float:
    model.eval()
    gen = batches(stream, args.batch_size, args.seq_len, args.direction,
                  seed=99, rank=0, world=1)
    total = 0.0
    with torch.no_grad():
        for _ in range(batches_n):
            w = torch.from_numpy(next(gen)).to(device)
            total += causal_loss(model, w).item()
    model.train()
    return total / batches_n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--direction", choices=["forward", "backward"], required=True)
    ap.add_argument("--tokens", type=Path, required=True, help="train .bin")
    ap.add_argument("--val-tokens", type=Path, default=None)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="stop early and still checkpoint, for queue walltime")
    args = ap.parse_args()

    device, rank, world = setup()
    torch.manual_seed(args.seed)
    main_proc = rank == 0

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.train()
    raw = model
    if is_distributed():
        model = DistributedDataParallel(model, device_ids=[device.index])

    stream = load_stream(args.tokens)
    val = load_stream(args.val_tokens) if args.val_tokens else None
    gen = batches(stream, args.batch_size, args.seq_len, args.direction,
                  args.seed, rank, world)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                            betas=(0.9, 0.95))

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    if main_proc:
        print(f"direction={args.direction} device={device} world={world} "
              f"tokens={len(stream):,} steps={args.steps}", flush=True)

    t0 = time.time()
    history = []
    stopped_at = args.steps
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        w = torch.from_numpy(next(gen)).to(device)
        loss = causal_loss(model, w)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if main_proc and (step % args.log_every == 0 or step == args.steps - 1):
            mins = (time.time() - t0) / 60
            print(f"step {step:6d}  loss {loss.item():.4f}  "
                  f"ppl {math.exp(min(20, loss.item())):8.2f}  {mins:5.1f}m",
                  flush=True)
            history.append({"step": step, "loss": loss.item(), "minutes": mins})

        if args.max_minutes and (time.time() - t0) / 60 > args.max_minutes:
            stopped_at = step + 1
            if main_proc:
                print(f"stopping at step {step}: hit --max-minutes", flush=True)
            break

    result = {"direction": args.direction, "steps_run": stopped_at,
              "steps_requested": args.steps, "world": world,
              "minutes": (time.time() - t0) / 60, "history": history,
              "tokens_seen": stopped_at * args.batch_size * args.seq_len * world}
    if val is not None and main_proc:
        result["val_loss"] = evaluate(raw, val, args, device)
        result["val_ppl"] = math.exp(min(20, result["val_loss"]))
        print(f"val loss {result['val_loss']:.4f} "
              f"ppl {result['val_ppl']:.2f}", flush=True)

    if main_proc:
        args.out.mkdir(parents=True, exist_ok=True)
        raw.save_pretrained(args.out)
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(args.model).save_pretrained(args.out)
        (args.out / "training.json").write_text(json.dumps(result, indent=2))
        print(f"saved {args.out}", flush=True)

    if is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
