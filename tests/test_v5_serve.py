import json

from server import v5_serve


def test_pool_keeps_both_halves_and_drops_broken_entries(tmp_path, monkeypatch):
    path = tmp_path / "pool.json"
    path.write_text(json.dumps([{"type": "result", "left": ["A"], "right": ["a"]},
                                {"type": "result", "left": ["B"]}]))  # missing its right half
    monkeypatch.setattr(v5_serve, "POOL_PATH", str(path))
    v5_serve._pool.clear()
    v5_serve._pool.extend(x for x in json.loads(path.read_text()) if x.get("left") and x.get("right"))
    assert len(v5_serve._pool) == 1
    v5_serve._remember({"type": "result", "left": ["No"], "right": ["on"], "center": "",
                        "letters": 4, "words": 2, "seconds": 9.9})
    saved = json.loads(path.read_text())[-1]
    assert saved["right"] == ["on"] and "seconds" not in saved
    assert v5_serve.pooled()["cached"] is True


def test_one_search_per_visitor_and_a_global_cap(monkeypatch):
    monkeypatch.setattr(v5_serve, "WORKERS", 2)
    v5_serve._busy, v5_serve._inflight, v5_serve._recent = 0, set(), {}
    assert v5_serve._admit("a") and not v5_serve._admit("a")  # one at a time per visitor
    assert v5_serve._admit("b") and not v5_serve._admit("c")  # two workers, both busy
    v5_serve._release("a")
    assert v5_serve._admit("c")
