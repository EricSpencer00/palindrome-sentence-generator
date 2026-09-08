from llm_palindrome.semantic_graph import cosine, paired_path


def test_cosine_handles_identical_orthogonal_and_empty_vectors():
    assert cosine([1, 0], [1, 0]) == 1
    assert cosine([1, 0], [0, 1]) == 0
    assert cosine([], []) == 0


def test_paired_path_requires_both_mirrored_transitions():
    pairs = [
        {"left": "outer l", "right": "outer r"},
        {"left": "inner l", "right": "inner r"},
        {"left": "bad l", "right": "bad r"},
    ]
    emb = {
        "centre": [1, 0], "inner l": [1, 0], "inner r": [1, 0],
        "outer l": [1, 0], "outer r": [1, 0],
        "bad l": [1, 0], "bad r": [0, 1],
    }
    path = paired_path(pairs, "centre", emb, threshold=0.9, want=2, seed=1,
                       max_template_uses=10)
    assert len(path) == 2
    assert all("bad" not in pair["left"] for pair in path)


def test_paired_path_rejects_embedding_coherence_made_of_repetition():
    pairs = [
        {"left": "deep speed", "right": "trade man"},
        {"left": "deep speed", "right": "trade man"},
        {"left": "quiet river", "right": "water rests"},
    ]
    emb = {text: [1, 0] for pair in pairs for text in pair.values()}
    emb["centre words"] = [1, 0]
    path = paired_path(pairs, "centre words", emb, threshold=0.9,
                       want=3, seed=3, max_bigram_uses=1)
    assert len(path) <= 2
    assert sum(pair["left"] == "deep speed" for pair in path) <= 1


def test_paired_path_does_not_repeat_the_centre_as_a_sentence():
    pairs = [
        {"left": "items draw award", "right": "award draws items"},
        {"left": "other words remain", "right": "meaning stays clear"},
    ]
    emb = {text: [1, 0] for pair in pairs for text in pair.values()}
    emb["items draw award"] = [1, 0]
    path = paired_path(pairs, "items draw award", emb, threshold=0.9,
                       want=2, seed=1, max_template_uses=10)
    assert all(pair["left"] != "items draw award" for pair in path)
