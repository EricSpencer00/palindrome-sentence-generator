"""Fire N simultaneous /api/generate requests and report what each got.

usage: python tools/load_test.py BASE_URL N [SAME_IP]
Distinct visitors are simulated with the cf-connecting-ip header, which the
API trusts only because the tunnel sets it; SAME_IP extra requests share one.
"""
import concurrent.futures as cf
import json
import re
import sys
import time
import urllib.request

base, n = sys.argv[1].rstrip("/"), int(sys.argv[2])
same = int(sys.argv[3]) if len(sys.argv) > 3 else 0


def hit(ip: str):
    t0 = time.time()
    req = urllib.request.Request(f"{base}/api/generate?prompt=&budget=16", headers={"cf-connecting-ip": ip})
    last = None
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            for line in r:
                line = line.decode().strip()
                if line.startswith("data: "):
                    last = json.loads(line[6:])
    except Exception as e:
        return ("fail", time.time() - t0, type(e).__name__)
    if not last or last.get("type") != "result":
        return ("error", time.time() - t0, (last or {}).get("message"))
    mid = last.get("centerDisplay") or last.get("center") or ""
    s = re.sub("[^a-z]", "", " ".join(last["left"] + [mid] + last["right"]).lower())
    kind = "cached" if last.get("cached") else "fresh"
    return (kind, time.time() - t0, s == s[::-1])


ips = [f"10.0.0.{i}" for i in range(n)] + ["10.9.9.9"] * same
with cf.ThreadPoolExecutor(len(ips)) as ex:
    out = list(ex.map(hit, ips))
for kind in ("fresh", "cached", "error", "fail"):
    got = [o for o in out if o[0] == kind]
    if got:
        ts = sorted(o[1] for o in got)
        print(f"{kind:6} {len(got):3}  median {ts[len(ts) // 2]:5.1f}s  max {ts[-1]:5.1f}s  "
              f"{'exact ' + str(sum(o[2] is True for o in got)) if kind in ('fresh', 'cached') else set(o[2] for o in got)}")
