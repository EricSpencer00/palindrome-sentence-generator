"""Admission for v5 under load: a few searches at a time, everyone else gets a
paragraph the engine already wrote.

A search is ~15 s of pure-Python CPU. Threads share one core under the GIL, so
searches run in WORKERS separate processes. When every worker is busy, or one
visitor asks too often, the request is answered at once from a pool of earlier
results: seeded from POOL_PATH (the site's replay cache) and grown by every
fresh search. Nothing waits in a queue, so a traffic spike costs the box at
most WORKERS cores.
"""
from __future__ import annotations

import asyncio
import collections
import json
import os
import random
import threading
import time
from concurrent.futures import ProcessPoolExecutor

WORKERS = int(os.environ.get("PALINDROME_V5_WORKERS", "2"))
POOL_PATH = os.environ.get("PALINDROME_V5_POOL", "data/pool.json")
POOL_MAX = 300
PER_IP = int(os.environ.get("PALINDROME_V5_PER_IP", "6"))  # fresh searches per window
WINDOW = 600.0

_executor: ProcessPoolExecutor | None = None
_busy = 0
_lock = threading.Lock()
_pool: collections.deque = collections.deque(maxlen=POOL_MAX)
_inflight: set[str] = set()
_recent: dict[str, collections.deque] = {}


def _init_worker() -> None:
    os.environ["PALINDROME_NO_WARM"] = "1"  # server.app must not spawn a pool of its own
    from server import v5
    v5.warm()


def _search(prompt: str):
    from server.app import _search_v5
    return _search_v5(prompt)


def start() -> None:
    """Load the pool and spawn the workers (each warms its own models)."""
    global _executor
    try:
        with open(POOL_PATH) as fh:
            _pool.extend(x for x in json.load(fh) if x.get("left") and x.get("right"))
    except (OSError, ValueError):
        pass
    import multiprocessing as mp
    _executor = ProcessPoolExecutor(WORKERS, mp_context=mp.get_context("spawn"),
                                    initializer=_init_worker)
    for _ in range(WORKERS):  # warm now, not on the first visitor
        _executor.submit(time.sleep, 2)


def pooled() -> dict | None:
    if not _pool:  # every pooled entry carries both halves (checked on load and on add)
        return None
    return {**random.choice(list(_pool)), "cached": True}


def _admit(ip: str) -> bool:
    global _busy
    now = time.time()
    with _lock:
        hits = _recent.setdefault(ip, collections.deque())
        while hits and now - hits[0] > WINDOW:
            hits.popleft()
        if _busy >= WORKERS or ip in _inflight or len(hits) >= PER_IP:
            return False
        _busy += 1
        _inflight.add(ip)
        hits.append(now)
        return True


def _release(ip: str) -> None:
    global _busy
    with _lock:
        _busy -= 1
        _inflight.discard(ip)


def _remember(result: dict) -> None:
    keep = {k: result[k] for k in ("type", "left", "right", "center", "centerDisplay", "pivot",
                                   "pivotOdd", "promptCenter", "letters", "words") if k in result}
    _pool.append(keep)
    try:
        tmp = POOL_PATH + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(list(_pool), fh)
        os.replace(tmp, POOL_PATH)
    except OSError:
        pass


async def generate(prompt: str, ip: str) -> dict | None:
    """A fresh result when a worker is free for this visitor, else a pooled one.
    A typed prompt is only served fresh: a pooled paragraph would ignore it."""
    if _executor is None or not _admit(ip):
        return pooled() if not prompt.strip() else {"type": "busy"}
    try:
        result = await asyncio.get_running_loop().run_in_executor(_executor, _search, prompt)
    except Exception:
        result = None
    finally:
        _release(ip)
    if result:
        if not prompt.strip():
            _remember(result)
        return result
    return pooled() if not prompt.strip() else None
