from experiments.packed_temporal_paragraph_20260927 import compile_temporal, run


def test_temporal_fact_is_exactly_once_on_every_accepting_path():
    g, _ = compile_temporal()
    successors = {}
    for a, b, _, text, role in g.edges:
        successors.setdefault(a, []).append((b, int(bool(text) and role == 'time:fact0:event0')))
    for a, targets in g.epsilon.items():
        successors.setdefault(a, []).extend((b, 0) for b in targets)
    pending, seen, counts = [(g.start, 0)], set(), set()
    while pending:
        node, count = pending.pop()
        if (node, count) in seen:
            continue
        seen.add((node, count))
        if node == g.finish:
            counts.add(count)
        pending.extend((nxt, count+delta) for nxt, delta in successors.get(node, ()))
    assert counts == {1}


def test_controls_and_bounded_search():
    result = run()
    assert len(result['conditions'][0]['candidates']) == 2
    assert all(not row['cap_reached'] for row in result['conditions'])
    assert len(result['complete_prose_controls']) == 2
    assert all(row['rendered'].count('.') == 3 for row in result['complete_prose_controls'])
