# Reverse-transition authored boundary search (2026-09-16)

As a final same-family repair, this run replaces frequency reranking with a
small hand-authored role bank and an exact bilateral boundary walker. It keeps
the corrected complete grammar `DET SUBJ VERB DET ADJ OBJ PREP DET NOUN` on
both sides and never promotes a fragment.

The search is independently lexicalized and uses only character equality at
the two active boundaries. Evidence records the full state audit, rendered
complete probes, and exact/admission counts.

Artifact: [`reverse_transition_svo_authored_search_20260916.py`](../experiments/reverse_transition_svo_authored_search_20260916.py)

Evidence: [`reverse-transition-svo-authored-search-20260916.json`](../runs/reverse-transition-svo-authored-search-20260916.json)
