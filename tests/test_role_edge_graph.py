from llm_palindrome.role_edge_graph import RoleEdge, build_graph, novelty_preflight, search_paths, tape


def test_edges_and_role_path_are_exact_before_rendering():
    edges = [
        RoleEdge("stressed deliver", "reviled desserts", "subject", "verb"),
        RoleEdge("deliver a", "a reviled", "verb", "object"),
        RoleEdge("deliver stressed", "desserts reviled", "object", "end"),
    ]
    paths = search_paths(build_graph(edges), ["subject"], max_edges=3)
    assert paths
    assert all(path.exact() for path in paths)
    assert paths[-1].roles == ("subject", "verb", "object")
    assert tape(paths[-1].render()) == tape(paths[-1].render())[::-1]


def test_malformed_edges_are_not_repaired_into_graph():
    graph = build_graph([RoleEdge("left", "wrong", "subject", "end")])
    assert graph == {}


def test_novelty_preflight_uses_normalized_render():
    edge = RoleEdge("stressed deliver", "reviled desserts", "subject", "end")
    path = search_paths(build_graph([edge]), ["subject"])[0]
    assert novelty_preflight([path], [path.render().upper()]) == []
