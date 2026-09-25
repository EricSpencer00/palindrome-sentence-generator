from experiments.score_length_stratified_readability import (
    select_heldout_span,
    split_fileids,
)


def test_brown_document_split_is_deterministic_and_disjoint():
    fileids = [f"news-{index:03d}" for index in range(100)]
    train, heldout = split_fileids(fileids)
    train_again, heldout_again = split_fileids(list(reversed(fileids)))
    assert train == train_again
    assert heldout == heldout_again
    assert set(train).isdisjoint(heldout)
    assert set(train) | set(heldout) == set(fileids)
    assert 10 <= len(heldout) <= 30


def test_heldout_controls_are_complete_contiguous_and_nonoverlapping():
    documents = {
        "a": (
            [["one", "two"], ["three"], ["four", "five"], ["six"]],
            [(0, ["one", "two"]), (1, ["three"]),
             (2, ["four", "five"]), (3, ["six"])],
        ),
        "b": (
            [["red", "blue"], ["green"], ["gold", "white"]],
            [(0, ["red", "blue"]), (1, ["green"]),
             (2, ["gold", "white"])],
        ),
    }
    occupied = {}
    first = select_heldout_span(documents, 3, occupied, "control-a")
    fileid, start, end, count, surface = first
    occupied.setdefault(fileid, []).append((start, end))
    second = select_heldout_span(documents, 3, occupied, "control-b")
    fileid2, start2, end2, count2, _ = second
    assert count == 3
    assert count2 == 3
    assert fileid != fileid2 or end <= start2 or end2 <= start
    assert surface
