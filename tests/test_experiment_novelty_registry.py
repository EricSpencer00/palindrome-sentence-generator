from pathlib import Path
import json

from experiments.validate_experiment_novelty_20260915 import validate
from experiments.preflight_experiment_novelty import preflight


def test_registered_experiments_have_unique_signatures_and_artifacts():
    result = validate()
    assert result["missing"] == []
    assert result["entries"] == result["unique_signatures"]
    assert result["entries"] == result["unique_artifacts"]
    assert result["excluded"] == 24
    # The ledger is append-only: parallel construction routes may add entries
    # without making this invariant stale.  The validator still requires every
    # registered artifact to resolve and every signature to be unique.
    assert result["entries"] >= 113
    assert result["run_artifacts"] >= 80


def test_seed_probes_are_explicitly_excluded_as_overlapping_repairs():
    path = Path(__file__).parents[1] / "docs/experiment-novelty-registry.json"
    data = json.loads(path.read_text())
    excluded = {row["id"]: row for row in data["excluded"]}
    assert set(excluded) == {
        "eliot-structural-control-preflight-20260916",
        "orthographic-derivation-preflight-20260916",
        "heteropalindromic-clause-pair-preflight-20260916",
        "bidirectional-phrase-pair-preflight-20260916",
        "semantic-word-pair-cross-boundary-preflight-20260916",
        "event-extension-transducer-preflight-20260916",
        "semantic-involution-frame-excluded",
        "seed-symmetric-mutation-excluded",
        "seed-boundary-shift-excluded",
        "thematic-grid-seam-composition-excluded",
        "semantic-pairing-typed-clauses-excluded",
        "multiset-balanced-grammar-invalid-excluded",
        "reversible-phrase-chain-preflight-20260916",
        "coordination-ellipsis-preflight-20260916",
        "modal-scope-preflight-20260916",
        "nested-quotation-preflight-20260916",
        "quantified-comparison-preflight-20260916",
        "centerout-reversible-pair-preflight-20260916",
        "gpt2-reverse-rerank-preflight-20260916",
        "b3-corpus-weighted-reverse-preflight-20260916-excluded",
        "center-insertion-outer-tape-preflight-20260916",
            "cumulative-boundary-profile-fixedpoint-20260916",
            "semantic-phrase-lattice-automata-20260916-excluded",
            "semantic-scene-stack-machine-20260916-excluded",
    }
    assert excluded["semantic-involution-frame-excluded"]["overlaps"] == []
    assert excluded["bidirectional-phrase-pair-preflight-20260916"]["overlaps"] == [
        "manual-endpoint-engineering", "attested-phrase-pair-wrapper"
    ]
    assert all(
        excluded[name]["overlaps"] == ["internal-center-window-repair"]
        for name in ("seed-symmetric-mutation-excluded", "seed-boundary-shift-excluded")
    )
    assert excluded["thematic-grid-seam-composition-excluded"]["overlaps"] == [
        "character-ledger-promptbank", "semantic-sentence-pair-alignment"
    ]
    assert excluded["semantic-pairing-typed-clauses-excluded"]["overlaps"] == [
        "typed-semordnilap", "character-ledger-promptbank", "semantic-sentence-pair-alignment"
    ]
    assert excluded["multiset-balanced-grammar-invalid-excluded"]["overlaps"] == [
        "global-brown-pos", "evolutionary-prose-genome"
    ]


def test_preflight_checks_registered_and_excluded_routes():
    result = preflight(
        "new-test-route",
        "test-only-state-space|independent-construction-dimension",
        "runs/test-only-route.json",
    )
    assert result["status"] == "novel"
    assert result["registered_families_checked"] == len(
        json.loads((Path(__file__).parents[1] / "docs/experiment-novelty-registry.json").read_text())["entries"]
    )
    assert result["excluded_routes_checked"] == 24
    assert result["manual_review_required"] is False
    assert result["conceptual_near_pairs"] == []


def test_preflight_surfaces_conceptual_near_pair_before_execution():
    result = preflight(
        "near-test-route",
        "semantic-event-pair|independent-right-lexicalization|new-test-state",
        "runs/near-test-route.json",
        near_threshold=0.30,
    )
    assert result["manual_review_required"] is True
    assert result["conceptual_near_pairs"][0]["id"] == "connective-bearing-event-pair"
    assert result["conceptual_near_pairs"][0]["jaccard"] >= 0.30


def test_latest_experiments_are_registered_as_distinct_families():
    path = Path(__file__).parents[1] / "docs" / "experiment-novelty-registry.json"
    ids = {row["id"] for row in __import__("json").loads(path.read_text())["entries"]}
    assert "brown-coreferent-variable-pp" in ids
    assert "event-frame-independent-relexicalization" in ids
    assert "fresh-paired-clause-ledger" in ids
    assert "dialogue-shared-topic-elliptical-residual" in ids
    assert "morphological-derivational-seam" in ids
    assert "cp-semantic-grammar-palindrome" in ids
    assert "clause-lattice-joint-dp" in ids
    assert "character-clause-fst-joint-emission" in ids
    assert "two-bank-word-equation-seam-dp" in ids
    assert "seam-first-complete-clause-authoring" in ids
    assert "semantic-dependency-outside-in" in ids
    assert "synchronous-semantic-parse-equations" in ids
    assert "internal-center-window-repair" in ids
    assert "discourse-plan-coupled-expansion" in ids
    assert "collocation-synchronous-grammar" in ids
    assert "human-compositional-center-window" in ids
    assert "global-semantic-paraphrase-rewrite" in ids
    assert "collocation-graph-path" in ids
    assert "semantic-sentence-pair-alignment" in ids
    assert "template-analogy-semantic-lexicalization" in ids
    assert "neural-dual-prefix-beam-v2" in ids
    assert "grammar-intersection-chart" in ids
    assert "dependency-attribute-grammar-chart" in ids
    assert "evolutionary-prose-genome" in ids
    assert "bpe-dual-continuation" in ids
    assert "lexicalized-tag-yield-equation" in ids
    assert "mined-phrase-chunk-clause-composition" in ids
    assert "morphosemantic-product-delay-20260916" in ids
    assert "morphology-first-dependency-lattice-20260916" in ids
    assert "weighted-cfg-sync-dp-20260916" in ids
    assert "reversible-grammar-insertion-20260916" in ids
    assert "corpus-neural-frame-realizer-20260916" in ids
    assert "variable-boundary-tape-ilp" in ids
    assert "reverse-complement-eulerian" in ids
    assert "prosodic-foot-surface-realizer" in ids
    assert "global-tied-masked-denoising" in ids
    assert "character-lm-half-tape" in ids
    assert "corpus-sentence-gram-fst" in ids
    assert "lexical-admission-centerout" in ids
    assert "grammar-boundary-resegmentation-repair" in ids
    assert "fixed-tape-valency-chart-repair" in ids
    assert "compound-derivational-scene-csp-20260916" in ids
    assert "paraphrase-graph-debt-paths-20260916" in ids
    assert "parse-tree-exact-cover-20260916" in ids
    assert "bilateral-semantic-cfg-20260916" in ids
    assert "human-scene-equation-frames-20260916" in ids
    assert "typed-edit-program-repair-20260916" in ids
    assert "event-graph-character-sat-20260916" in ids
    assert "syntax-stack-semantic-role-decoder-20260916" in ids
    assert "phrase-equation-inventory-solver-20260916" in ids
    assert "semantic-center-sat-20260916" in ids
    assert "reverse-segmentation-cfg-valency-20260916" in ids
    assert "dependency-mirror-pair-constructor-20260916" in ids
    assert "dependency-mirror-pair-repair-20260916" in ids
    assert "online-grammar-state-char-decoder-20260916" in ids
    assert "online-grammar-state-slot-repair-20260916" in ids
    assert "online-grammar-state-outer-frame-repair-20260916" in ids
    assert "reverse-lexicon-synthesis-20260916" in ids
    assert "centerout-grammar-boundary-dp-20260916" in ids
    assert "authored-clause-template-sat-20260916" in ids
    assert "reverse-lexicon-inflection-repair-20260916" in ids
    assert "centerout-grammar-boundary-repair-20260916" in ids
    assert "authored-clause-template-sat-repair-20260916" in ids
    assert "live-slot-equation-cfg-resegmentation-20260916" in ids
    assert "clause-growth-frame-repair-20260916" in ids
    assert "proper-name-caption-crossword" in ids
    assert "information-structure-focus-scope" in ids
    assert "anaphoric-scene-chain-composition" in ids
    assert "multiset-balanced-pair-sampling" in ids
    assert "proper-name-reverse-grammar" in ids
    assert "adaptive-crossword-span-cover" in ids
    assert "context-template-crossword-repair" in ids
    assert "syntactic-mirror-template-repair" in ids
    assert "semantic-selectional-prefix-automaton" in ids
    assert "model-authored-clause-bank-index" in ids
    assert "semantic-scene-seam-growth" in ids
    assert "live-seam-intent-continuation" in ids
    assert "fixed-tape-gpt2-boundary-decoder-20260915" in ids
    assert "pos-template-centerout-longform-repair-20260915" in ids
    assert "pos-template-centerout-longform-center-residual-repair-20260915" in ids
    assert "role-aware-reversible-reservoir-centerout-20260915" in ids
    assert "variable-length-role-reservoir-centerout-20260915" in ids
    assert "asymmetric-template-reservoir-centerout-20260915" in ids
    assert "attested-phrase-pair-wrapper-20260915" in ids
    assert "homograph-sense-lattice-20260915" in ids
    assert "interrogative-quantifier-fsm-20260915" in ids
    assert "terminal-aware-grammar-intersection-20260916" in ids
    assert "semantic-mcts-derivation-20260916" in ids
    assert "paired-obligation-astar-20260916" in ids
    assert "paired-obligation-astar-debt-repair-20260916" in ids
    assert "recursive-obligation-clause-growth-20260916" in ids
    assert "corpus-span-boundary-dp-20260916" in ids
    assert "recursive-grammar-residual-dp-20260916" in ids
    assert "dialogue-speech-act-residual-20260916" in ids
    assert "brown-attested-residual-lattice-20260916" in ids
    assert "seedless-semantic-cfg-bilateral-20260916" in ids
    assert "whole-sentence-semordnilap-clauses-20260916" in ids
    assert "semantic-mutation-residual-20260916" in ids
    assert "rhetorical-plan-lattice-20260916" in ids
    assert "inflectional-fst-clitic-tape-20260916" in ids
    assert "induced-pcfg-character-equation-20260916" in ids
    assert "graph-to-prose-path-20260916" in ids
    assert "voice-alternation-residual-20260916" in ids
    assert "ccg-semantic-solver-20260916" in ids
    assert "dependency-completion-csp-20260916" in ids
    assert "lexical-word-equation-inventory-20260916" in ids
    assert "direct-constrained-authoring-20260916" in ids
    assert "lexical-chain-palindrome-20260916" in ids
    assert "morphology-semantic-template-csp-20260916" in ids
    assert "pivot-paragraph-beam-20260916" in ids
    assert "prosodic-foot-scene-constructor-20260916" in ids
    assert "induced-grammar-reverse-decoder-20260916" in ids
    assert "semantic-frame-tape-solver-20260916" in ids
    assert "maxsat-semantic-grammar-20260916" in ids
    assert "heldout-boundary-decoder-20260916" in ids
    assert "lexical-trie-segmentation-repair-20260916" in ids
    assert "residual-lexical-decoder-20260916" in ids
    assert "evidential-scene-planner-20260916" in ids
    assert "paragraph-paraphrase-obligation-20260916" in ids
    assert "heteropalindromic-clause-composer-20260916" in ids
    assert "function-word-boundary-balance-preflight-20260916-excluded" in ids


def test_rhythmai_probe_is_registered_as_direct_authoring_repair_evidence():
    path = Path(__file__).parents[1] / "docs" / "experiment-novelty-registry.json"
    data = json.loads(path.read_text())
    direct = next(row for row in data["entries"] if row["id"] == "direct-constrained-authoring-20260916")
    assert "runs/rhythmai-authoring-probe-20260916.json" in direct["run_artifacts"]
    assert direct["repair_artifacts"] == ["runs/rhythmai-authoring-probe-20260916.json"]


def test_new_centerout_repairs_keep_failure_evidence_and_reader_gate_closed():
    root = Path(__file__).parents[1]
    model_run = json.loads((root / "runs/fixed-tape-gpt2-boundary-decoder-20260915.json").read_text())
    assert model_run["stats"]["mechanically_admitted"] == 0
    assert model_run["stats"]["reader_eligible"] == 0
    assert model_run["provenance"]["programmatic_readability_claim"] is False
    center_run = json.loads((root / "runs/pos-template-centerout-longform-center-residual-repair-20260915.json").read_text())
    assert center_run["stats"]["mechanically_admitted"] == 0
    assert center_run["stats"]["reader_eligible"] == 0
    assert center_run["repair"] == "allow only a palindromic residual at the final character centre"


def test_asymmetric_reservoir_route_is_preflighted_and_keeps_reader_gate_closed():
    root = Path(__file__).parents[1]
    for name in (
        "role-aware-reversible-reservoir-centerout-20260915.json",
        "variable-length-role-reservoir-centerout-20260915.json",
        "asymmetric-template-reservoir-centerout-20260915.json",
    ):
        run = json.loads((root / "runs" / name).read_text())
        assert run["novelty_audit"]["registry_entries_read_before_run"] == 78
        assert run["novelty_audit"]["conceptual_near_pairs"] == []
        assert run["stats"]["mechanically_admitted"] == 0
        assert run["stats"]["reader_eligible"] == 0
        assert run["provenance"]["programmatic_readability_claim"] is False

    phrase_run = json.loads((root / "runs/attested-phrase-pair-wrapper-20260915.json").read_text())
    assert phrase_run["novelty_audit"]["registry_entries_read_before_run"] == 78
    assert phrase_run["stats"]["reverse_segment_hits"] == 0
    assert phrase_run["stats"]["exact"] == 0
    assert phrase_run["stats"]["reader_eligible"] == 0
    assert phrase_run["rendered_probes"]

    homograph_run = json.loads((root / "runs/homograph-sense-lattice-20260915.json").read_text())
    assert homograph_run["novelty_audit"]["registry_entries_read_before_run"] == 82
    assert homograph_run["stats"]["mechanically_admitted"] == 0
    assert homograph_run["stats"]["reader_eligible"] == 0
    assert homograph_run["rendered_probes"]

    indexed = json.loads((root / "runs/interrogative-quantifier-indexed-repair-20260916.json").read_text())
    assert indexed["repair_of"] == "interrogative-quantifier-fsm-20260915"
    assert indexed["novelty_preflight"]["manual_review_required"] is True
    assert indexed["stats"]["target_index_hits"] == 1
    assert indexed["stats"]["known_or_duplicate_reject"] == 1
    assert indexed["stats"]["mechanically_admitted"] == 0
    assert indexed["stats"]["reader_eligible"] == 0


def test_terminal_aware_intersection_records_boundary_repair_and_closes_reader_gate():
    root = Path(__file__).parents[1]
    run = json.loads((root / "runs/terminal-aware-grammar-intersection-20260916.json").read_text())
    assert run["novelty_preflight"]["registry_entries_before_run"] == 90
    assert run["novelty_preflight"]["signature_overlap"] == []
    assert run["config"]["terminal_epsilon_closure"] is True
    assert run["stats"]["short_closures"] == 9
    assert run["stats"]["mechanically_admitted"] == 0
    assert run["stats"]["reader_eligible"] == 0


def test_semantic_mcts_records_uct_rollouts_and_closes_reader_gate():
    root = Path(__file__).parents[1]
    run = json.loads((root / "runs/semantic-mcts-derivation-20260916.json").read_text())
    assert run["novelty_preflight"]["registry_entries_before_run"] == 91
    assert run["novelty_preflight"]["signature_overlap"] == []
    assert run["stats"]["rollouts"] == 30_000
    assert run["stats"]["longest_probe_letters"] == 29
    assert run["stats"]["mechanically_admitted"] == 0
    assert run["stats"]["reader_eligible"] == 0


def test_semantic_mcts_reverse_prior_is_recorded_as_a_concrete_repair():
    root = Path(__file__).parents[1]
    run = json.loads((root / "runs/semantic-mcts-reverse-prior-repair-20260916.json").read_text())
    assert run["repair_of"] == "semantic-mcts-derivation-20260916"
    assert run["novelty_preflight"]["manual_review_required"] is True
    assert run["config"]["reverse_conditioned_prior"] is True
    assert run["stats"]["rollouts"] == 30_000
    assert run["stats"]["exact_complete"] == 0
    assert run["stats"]["mechanically_admitted"] == 0
    assert run["stats"]["reader_eligible"] == 0


def test_scene_growth_records_preflight_before_self_registration():
    root = Path(__file__).parents[1]
    run = json.loads((root / "runs/semantic-scene-seam-growth-20260915.json").read_text())
    assert run["novelty_preflight"]["registry_entries_before_run"] == 73
    assert run["novelty_preflight"]["manual_review_required"] is False
    assert run["exact_count"] == 0
    assert run["reader_eligible_count"] == 0


def test_live_seam_records_timeout_repairs_and_preflight():
    root = Path(__file__).parents[1]
    run = json.loads((root / "runs/live-seam-intent-continuation-20260915.json").read_text())
    assert run["novelty_preflight"]["registry_entries_before_run"] == 74
    assert run["novelty_preflight"]["manual_review_required"] is False
    assert run["exact"] == 0
    assert run["reader_eligible"] == 0
    assert "bounded short continuation" in run["repair_operator"]


def test_latest_artifacts_were_preflighted_before_self_registration():
    root = Path(__file__).parents[1]
    registry_count = len(json.loads((root / "docs/experiment-novelty-registry.json").read_text())["entries"])
    for name in ("evolutionary-prose-genome-20260915.json", "dependency-attribute-grammar-chart-20260915.json", "bpe-dual-continuation-20260915.json", "lexicalized-tag-yield-equation-20260915.json"):
        audit = json.loads((root / "runs" / name).read_text())["novelty_audit"]
        # Later distinct experiments may register after this artifact was
        # produced; the preflight count must be a prior snapshot, not an
        # unstable equality with today's registry size.
        assert 0 < audit["registry_entries_read_before_run"] < registry_count
        assert not audit.get("self_entry_present", [])
        assert not audit.get("replay_of_registered_family", False)
    chart = json.loads((root / "runs" / "grammar-intersection-chart-20260915.json").read_text())
    assert 0 < chart["registry_entry_count_before_run"] < registry_count
    assert chart["prior_signatures_overlap"] == []
    phrase_run = json.loads((root / "runs" / "mined-phrase-chunk-clause-composition-v3-20260915.json").read_text())
    assert phrase_run["novelty_audit"]["registry_entries_read_before_run"] < registry_count
    assert phrase_run["novelty_audit"]["signature_overlap"] == []
    assert phrase_run["stats"]["exact_candidates"] == 0
    assert len(phrase_run["rendered_candidates_and_probes"]) == 40
    ilp_run = json.loads((root / "runs" / "variable-boundary-tape-ilp-20260915.json").read_text())
    assert ilp_run["novelty_audit"]["registry_entries_read_before_run"] < registry_count
    assert ilp_run["novelty_audit"]["prior_signatures_overlap"] == []
    assert ilp_run["solver_summary"]["exact_solution_count"] == 0
    assert ilp_run["results"][0]["rendered_probe"]["independent_exact_audit"]["exact"] is False
    euler_run = json.loads((root / "runs" / "reverse-complement-eulerian-20260915.json").read_text())
    assert euler_run["novelty_audit"]["registry_entries_read_before_run"] == 52
    assert euler_run["novelty_audit"]["signature_overlap"] == []
    assert euler_run["stats"]["rendered_probes"] == 40
    assert euler_run["stats"]["reader_eligible"] == 0
    assert euler_run["repair_run"]["balanced_trails"] == 0
    prosodic_run = json.loads((root / "runs" / "prosodic-foot-surface-realizer-repair4-20260915.json").read_text())
    assert prosodic_run["novelty_audit"]["registry_entries_read_before_run"] == 54
    assert prosodic_run["novelty_audit"]["signature_overlap"] == []
    assert prosodic_run["novelty_audit"]["self_entry_present"] == ["prosodic-foot-surface-realizer"]
    assert prosodic_run["novelty_audit"]["repair_of_registered_family"] is True
    assert prosodic_run["stats"]["exact_candidates"] == 0
    assert prosodic_run["stats"]["reader_eligible"] == 0
    assert len(prosodic_run["rendered_candidates_and_probes"]) == 40
    assert prosodic_run["rendered_candidates_and_probes"][0]["rendered"]
    assert prosodic_run["rendered_candidates_and_probes"][0]["independent_exact_audit"]["exact"] is False
    masked_run = json.loads((root / "runs" / "global-tied-masked-denoising-20260915.json").read_text())
    assert masked_run["signature"] == "global-tied-character-mask|parallel-word-denoising|bidirectional-position-ledger|whole-tape-assignment|symbolic-fallback"
    assert len(masked_run["proposals"]) == 6
    assert sum(row["admitted"] for row in masked_run["proposals"]) == 0
    assert all("rendered" in row and "ledger" in row and "checks" in row for row in masked_run["proposals"])
    for name in (
        "character-lm-half-tape-20260915.json",
        "character-lm-half-tape-repair0-20260915.json",
        "character-lm-half-tape-repair1-20260915.json",
        "character-lm-half-tape-repair2-20260915.json",
    ):
        char_run = json.loads((root / "runs" / name).read_text())
        assert char_run["signature"] == "character-lm-half-tape|joint-forward-reverse-ngram-score|viterbi-word-boundary-recovery|independent-full-tape-audit"
        assert char_run["stats"]["exact_probes"] == 40
        assert char_run["stats"]["mechanically_admitted"] == 0
        assert char_run["stats"]["reader_eligible"] == 0
        assert char_run["rendered_candidates_and_probes"][0]["rendered"]
    seam_run = json.loads((root / "runs" / "semordnilap-template-inventory-20260915.json").read_text())
    assert seam_run["signature"] == "finite semordnilap pair inventory fills typed syntactic templates with bilateral character equations and seam-preserving repair substitutions"
    assert seam_run["probes"] == 350
    assert seam_run["exact"] == 0
    assert seam_run["mechanically_admitted"] == 0
    assert seam_run["reader_eligible"] == 0
    assert all(row["method"] == "typed-template-seam" for row in seam_run["rows"])
    semantic = __import__("experiments.semantic_involution_frame_20260915", fromlist=["run"]).run()
    assert semantic["status"] == "preflight_only_no_promotion"
    assert semantic["probes"][0]["text"] == "Deliver no evil. Live on, reviled."
    assert semantic["probes"][0]["exact_letter_palindrome"] is True


def test_latest_constructive_repairs_record_distinct_preflight_and_failure_frontier():
    root = Path(__file__).parents[1]
    centerout = json.loads((root / "runs/lexical-admission-centerout-20260915.json").read_text())
    assert centerout["novelty_audit"]["manual_review_required"] is False
    assert centerout["stats"]["mechanically_admitted"] == 4
    assert max(row["letters"] for row in centerout["rendered_candidates_and_probes"]) == 116
    valency = json.loads((root / "runs/fixed-tape-valency-chart-repair-20260915.json").read_text())
    assert valency["novelty_audit"]["repair_of_registered_family"] is True
    assert valency["stats"]["mechanically_admitted"] == 2
    assert valency["stats"]["complete_clause_parses"] == 0
    thematic = json.loads((root / "runs/thematic-grid-seam-composition-20260915.json").read_text())
    assert thematic["stats"]["tested"] == 16
    assert thematic["stats"]["exact"] == 0
    semantic = json.loads((root / "runs/semantic-pairing-repair-final-2026-09-15.json").read_text())
    assert semantic["config"]["frame_count"] == 17280
    assert semantic["exact_survivors"] == []
    proper = json.loads((root / "runs/proper-name-caption-crossword-20260915.json").read_text())
    assert proper["independent_audit"]["probes_checked"] == 25
    assert proper["independent_audit"]["primary_exact_count"] == 0
    assert proper["independent_audit"]["independent_exact_count"] == 0
    assert proper["independent_audit"]["disagreements"] == []
    info = json.loads((root / "runs/information-structure-focus-scope-2026-09-15.json").read_text())
    assert info["preflight"]["registry_entries"] == 63
    assert len(info["rows"]) == 16
    assert info["independent_audit"]["primary_exact"] == 0
    assert info["independent_audit"]["independent_exact"] == 0
    assert info["independent_audit"]["disagreements"] == []
    repair = json.loads((root / "runs/information-structure-terminal-seam-repair-2026-09-15.json").read_text())
    assert repair["preflight"]["manual_review_required"] is True
    assert repair["preflight"]["disposition"] == "repair, not a retained family"
    assert repair["independent_audit"]["probes_checked"] == 16
    assert repair["independent_audit"]["primary_exact"] == 0
    assert repair["independent_audit"]["independent_exact"] == 0
    anaphoric = json.loads((root / "runs/anaphoric-scene-chain-20260915.json").read_text())
    assert anaphoric["preflight"]["registry_entries"] == 63
    assert anaphoric["stats"]["scenes"] == 4
    assert anaphoric["independent_audit"]["primary_exact"] == 0
    assert anaphoric["independent_audit"]["independent_exact"] == 0
    assert anaphoric["independent_audit"]["disagreements"] == []
    multiset = json.loads((root / "runs/multiset-balanced-pair-sampling-20260915.json").read_text())
    assert multiset["novelty_preflight"]["registry_entries"] == 65
    assert multiset["parity_survivors"] == 0
    assert multiset["independent_audit"]["primary_exact"] == 0
    assert multiset["independent_audit"]["independent_exact"] == 0
    lattice = json.loads((root / "runs/multiset-parity-lattice-repair-20260915.json").read_text())
    assert lattice["preflight"]["disposition"] == "repair, not a retained family"
    assert lattice["clause_count"] == 2304
    assert lattice["parity_joined_pairs"] == 0
    assert lattice["rendered_probe_count"] == 25
    assert lattice["independent_audit"]["primary_exact"] == 0
    agreement = json.loads((root / "runs/multiset-agreement-lattice-repair-20260915.json").read_text())
    assert agreement["preflight"]["disposition"] == "repair, not a retained family"
    assert agreement["clause_count"] == 98304
    assert agreement["parity_joined_pairs"] == 2720
    assert agreement["independent_audit"]["primary_exact"] == 0
    assert agreement["independent_audit"]["independent_exact"] == 0
    proper_reverse = json.loads((root / "runs/proper-name-reverse-grammar-20260915.json").read_text())
    assert proper_reverse["preflight"]["registry_entries"] == 66
    assert proper_reverse["grammar_clause_count"] == 140368
    assert proper_reverse["raw_exact_pairs"] == 0
    assert proper_reverse["independent_audit"]["primary_exact"] == 0
