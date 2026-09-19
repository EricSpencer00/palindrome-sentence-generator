"""v4 evidence and evaluation API.

v4 is deliberately an evidence surface, not another unblinded generator.  It
exposes the strongest independently constructed candidate, its provenance,
and two independent exactness checks.  The evaluation endpoint is a
Shakespearean/RLAIF-inspired diagnostic: it can rank what to repair next, but
it cannot certify that a candidate reads as English.  That claim remains a
blinded-reader decision.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from llm_palindrome.admission import mechanical_admission_checks

router = APIRouter(prefix="/api/v4", tags=["v4 evidence"])

GATE_MESSAGE = (
    "v4 generation is gated: exactness and programmatic diagnostics do not "
    "establish readable English. A candidate must first pass a randomized "
    "blinded reader study."
)

BEST_KNOWN_TEXT = "An aide rips nine memos; some men inspire Diana."
BEST_KNOWN_PROVENANCE = {
    "run_id": "character-trie-grammar-decoder-20260919",
    "method": "character-trie grammar decoder with POS/inflection terminals and live half-tape assignments",
    "source": "project construction run; not catalogue text",
    "novelty_preflight": "passed local catalogue and construction-shortcut exclusions",
    "search_summary": {
        "exact_candidates": 1,
        "longest_exact_letters": 38,
        "mechanically_admitted_candidates": 1,
        "pilot_lengths": "38–60; one independently recovered 38-letter anchor, no >38 closure",
        "shared_participant_temporal_repair": "no exact closure in bounded pilot",
        "latest_dream_rsi_repair": "2 mechanically admitted exact rows; longest 50 letters; reader gate closed",
        "latest_semantic_shell_repair": "7,272 intact scene-shell renderings; longest 183 letters; zero exact closure",
        "latest_indexed_path_repair": "1,205 typed-path nodes across 38–56 letters; two exact 38-letter admissions; no improvement",
        "latest_connector_product_repair": "100 typed connector frontier witnesses; longest 65 letters; zero exact closure",
        "latest_residual_seam_repair": "246 grammatical clause combinations across 151 seam states; longest 66 letters; zero exact closure",
        "latest_terminal_repair": "252 state-selected held-out terminals; longest 67 letters; zero exact closure",
        "latest_relative_repair": "12,588 indexed relative-path nodes across 40–90 letters; zero exact closure",
        "latest_shared_relative_repair": "4,042 indexed nodes with shared participant and finite complement; zero exact closure",
        "latest_residual_prefix_repair": "258 candidates across 56 two-character seam-prefix states; longest 77 letters; zero exact closure",
        "latest_typed_bank_repair": "703,172 deterministic CSP nodes over a one-time model-authored bank; 390 frontier controls; zero exact closure",
        "latest_three_beat_repair": "1,344 finite SVO³ renderings; longest frontier 81 letters; zero exact closure",
        "latest_character_trie_relative_repair": "306,725 nodes across 84 targets from 39–80 letters; zero exact closure",
        "latest_authored_scene_lattice": "45 exact 42-letter diagnostics; 10 prior-tape collisions; zero mechanically admitted",
        "latest_compositional_shell_repair": "21,660 complete-shell compositions; 164 intact controls to 75 letters; zero exact closure",
        "latest_shell_inflection_repair": "11,340 role-preserving variants; 420 controls; zero exact closure",
        "latest_center_out_orbit_product": "12,000 independent paths per side; 34 product states; 4 intact 41-letter controls; zero exact closure",
        "latest_setting_frame_orbit_product": "2,000 independent paths per side; 37 product states; 4 intact 59-letter controls; zero exact closure",
        "reader_study": "not run",
    },
}

# The most recent constructive lane is kept alongside the exact frontier.  It
# is evidence about a search method, not a candidate being quietly promoted:
# the longest shell rendering is deliberately recorded as non-palindromic.
SEMANTIC_SHELL_RUN = {
    "run_id": "semantic-shell-growth-20260919",
    "method": "incremental typed semantic shells with live mirrored-edge character debt",
    "status": "completed_no_exact_closure",
    "rendered_candidates": 7272,
    "longest_rendered_letters": 183,
    "exact_candidates": 0,
    "mechanically_admitted_candidates": 0,
    "longest_rendered_example": (
        "some patient poets praise the sonnet a young herald guides Diana "
        "the actor guards the tent; the actors meet in the harbor; "
        "the harbor guards the quiet captain a new tale reads the sailor "
        "the sonnet praises some patient poets."
    ),
    "provenance": "fresh typed event bank; no finished-tape reversal; no catalogue import",
    "independent_validation": ["literal two-pointer audit", "forward/reverse SHA-256"],
    "reader_status": "not_run; no exact candidate reached the reader gate",
    "next_repair": (
        "replace the unclosed event shell with indexed character-debt states and a grammatical connector lattice; "
        "reader-test only an exact, intact-prose closure against shuffled controls"
    ),
}

# One post-hoc local-model pass is recorded separately from the deterministic
# rubric.  It is an AI-feedback diagnostic, not a reward used during search.
AI_FEEDBACK_RUN = {
    "experiment_id": "v4-rlaif-frontier-eval-20260919",
    "model": "gpt-oss:20b",
    "status": "diagnostic_only",
    "search_uses_feedback": False,
    "readability_certified": False,
    "blind_human_readers_required": True,
    "rubric": "0-3 intact English, scene coherence, Shakespearean cadence; length ignored",
    "scores": [
        {
            "run_id": "character-trie-grammar-decoder-20260919",
            "rendered": BEST_KNOWN_TEXT,
            "letters": 38,
            "exact": True,
            "intact_english": 2,
            "scene_coherence": 2,
            "shakespearean_cadence": 1,
            "repair": "Preserve the compact scene, then add a consequential second beat only through a new exact construction.",
        },
        {
            "run_id": "dream-rsi-strict-phrase-bank-20260919",
            "rendered": "To new one post is an evening. Is sign in even as its open owe. Not.",
            "letters": 50,
            "exact": True,
            "intact_english": 0,
            "scene_coherence": 0,
            "shakespearean_cadence": 0,
            "repair": "The passage is incoherent; return to a typed scene construction rather than preserving this closure.",
        },
        {
            "run_id": "dream-rsi-strict-phrase-bank-20260919",
            "rendered": "Erased on forever event is an evening. Is sign in even as it never ever. Of nodes are.",
            "letters": 66,
            "exact": True,
            "intact_english": 0,
            "scene_coherence": 0,
            "shakespearean_cadence": 0,
            "repair": "The passage is incoherent and is already rejected for a hidden proper palindrome span.",
        },
    ],
    "next_test": "randomized blinded intact-prose versus shuffled-control rating",
}

# Fresh Dream-RSI closures are exposed as a repair frontier, never folded into
# ``best_known``: the 50-letter row clears mechanical checks but has not been
# read by blinded humans, while the longer row is explicitly rejected for a
# hidden proper palindrome span.
DREAM_RSI_REPAIR_FRONTIER = [
    {
        "rendered": "To new one post is an evening. Is sign in even as its open owe. Not.",
        "letters": 50,
        "normalized": "tonewonepostisaneveningissigninevenasitsopenowenot",
        "sha256_forward": "855d51fa2b5cb8b4f63b9e043494f066702f8abbb329671ee5c68b82ca7788e3",
        "sha256_reverse": "855d51fa2b5cb8b4f63b9e043494f066702f8abbb329671ee5c68b82ca7788e3",
        "exact": True,
        "mechanically_admitted": True,
        "reader_status": "not_run",
        "promotion_status": "gated_pending_blinded_readers",
        "provenance": "dream-rsi-strict-phrase-bank-20260919; fresh model phrase proposals; not catalogue text",
    },
    {
        "rendered": "Erased on forever event is an evening. Is sign in even as it never ever. Of nodes are.",
        "letters": 66,
        "normalized": "erasedonforevereventisaneveningissigninevenasitnevereverofnodesare",
        "sha256_forward": "a4ebb3e316fb257c50e3eb17a9b35d94cf5796c43075cdf552cfa92c93492469",
        "sha256_reverse": "a4ebb3e316fb257c50e3eb17a9b35d94cf5796c43075cdf552cfa92c93492469",
        "exact": True,
        "mechanically_admitted": False,
        "rejection": "hidden proper palindrome span",
        "reader_status": "not_run",
        "provenance": "dream-rsi-strict-phrase-bank-20260919; fresh model phrase proposals; not catalogue text",
    },
]

INDEXED_PATH_RUN = {
    "run_id": "half-tape-indexed-path-csp-20260919",
    "method": "typed grammar path with inverted half-tape seam index",
    "status": "completed_exact_anchor_only",
    "search_nodes": 1205,
    "target_lengths": "38–56",
    "exact_candidates": 2,
    "mechanically_admitted_candidates": 2,
    "longest_exact_letters": 38,
    "longest_exact_example": "an aide rips nine memos some men inspire Diana.",
    "provenance": "typed ordinary grammar path; no phrase reversal; no catalogue import",
    "independent_validation": ["outside-in two-pointer", "forward/reverse SHA-256"],
    "reader_status": "not_run; anchor recovery is not human readability evidence",
    "next_repair": "agreement-carrying relative-complement path with indexed word-boundary offsets",
}

CONNECTOR_PRODUCT_RUN = {
    "run_id": "connector-character-product-20260919",
    "method": "typed connector CFG intersected with incremental outside-in character product",
    "status": "completed_no_exact_closure",
    "frontier_witnesses": 100,
    "longest_rendered_letters": 65,
    "exact_candidates": 0,
    "mechanically_admitted_candidates": 0,
    "longest_frontier_example": "a quiet player reads the old sonnet, while , while the patient poet praises Diana.",
    "provenance": "fresh typed clauses; no finished-tape reversal; no catalogue import",
    "independent_validation": ["literal two-pointer", "forward/reverse SHA-256"],
    "reader_status": "not_run; frontier witness is malformed and not a candidate",
    "next_repair": "index residual seam by remaining length, next character, and agreement state",
}

RESIDUAL_SEAM_RUN = {
    "run_id": "residual-seam-scene-lattice-20260919",
    "method": "complete grammatical clauses indexed by residual seam state",
    "status": "completed_no_exact_closure",
    "rendered_candidates": 246,
    "indexed_states": 151,
    "target_range": "40–70 letters",
    "longest_rendered_letters": 66,
    "exact_candidates": 0,
    "mechanically_admitted_candidates": 0,
    "longest_rendered_example": "the player greets the king as the bells sound a quiet poet praises the sonnet.",
    "provenance": "fresh grammatical clauses and connectors; no finished-tape reversal; no catalogue import",
    "independent_validation": ["literal outside-in two-pointer", "forward/reverse SHA-256"],
    "reader_status": "not_run; no exact candidate reached the reader gate",
    "next_repair": "replace one residual-bearing clause terminal with a held-out same-valency Shakespearean realization",
}

RESIDUAL_TERMINAL_RUN = {
    "run_id": "residual-terminal-repair-lattice-20260919",
    "method": "held-out same-valency terminal replacement selected by residual seam state",
    "status": "completed_no_exact_closure",
    "selected_by_state": 252,
    "rendered_candidates": 252,
    "target_range": "40–90 letters",
    "longest_rendered_letters": 67,
    "exact_candidates": 0,
    "mechanically_admitted_candidates": 0,
    "longest_rendered_example": "a quiet singer praises the court while the player greets the king.",
    "provenance": "held-out same-valency terminal bank; no finished-tape reversal; no catalogue import",
    "independent_validation": ["literal two-pointer", "forward/reverse SHA-256"],
    "reader_status": "not_run; no exact candidate reached the reader gate",
    "next_repair": "add a second held-out terminal tier keyed by a two-character residual prefix",
}

RELATIVE_INDEXED_RUN = {
    "run_id": "relative-indexed-boundary-csp-20260919",
    "method": "agreement-carrying relative-complement path with indexed word-boundary offsets",
    "status": "completed_no_exact_closure",
    "search_nodes": 12588,
    "target_range": "40–90 letters",
    "exact_candidates": 0,
    "mechanically_admitted_candidates": 0,
    "longest_exact_letters": 0,
    "longest_frontier_example": "a bard reads a letter who some men inspire Diana.",
    "frontier_letters": 39,
    "provenance": "agreement-carrying relative path; no finished-tape reversal; no catalogue import",
    "independent_validation": ["outside-in two-pointer", "forward/reverse SHA-256"],
    "reader_status": "not_run; no exact candidate reached the reader gate",
    "next_repair": "permit a shared participant across the relative seam and add a finite verb-complement marker",
}

SHARED_RELATIVE_RUN = {
    "run_id": "shared-relative-complement-csp-20260919",
    "method": "shared-participant relative seam with finite complement markers",
    "status": "completed_no_exact_closure",
    "search_nodes": 4042,
    "target_range": "40–100 letters",
    "exact_candidates": 0,
    "mechanically_admitted_candidates": 0,
    "longest_exact_letters": 0,
    "provenance": "shared participant and finite that/and markers; no finished-tape reversal; no catalogue import",
    "independent_validation": ["outside-in two-pointer", "forward/reverse SHA-256"],
    "reader_status": "not_run; no exact candidate reached the reader gate",
    "next_repair": "carry two finite markers plus a bounded adjunct slot",
}

RESIDUAL_PREFIX_RUN = {
    "run_id": "residual-prefix2-attachment-lattice-20260919",
    "method": "two-character residual-prefix lattice with attachment, agreement, and valency state",
    "status": "completed_no_exact_closure",
    "selected_by_prefix": 258,
    "indexed_states": 56,
    "target_range": "40–100 letters",
    "longest_rendered_letters": 77,
    "exact_candidates": 0,
    "mechanically_admitted_candidates": 0,
    "longest_rendered_example": "the player guards the hearth while a quiet poet praises the sonnet at dawn.",
    "provenance": "fresh attachment lattice; no finished-tape reversal; no catalogue import",
    "independent_validation": ["literal two-pointer", "forward/reverse SHA-256"],
    "reader_status": "not_run; no exact candidate reached the reader gate",
    "next_repair": "replace the one-character edge policy with indexed two-character word transitions",
}

LLM_TYPED_BANK_RUN = {
    "run_id": "llm-authored-typed-bank-csp-20260919",
    "method": "one-time typed Shakespearean lexical bank intersected with deterministic half-tape CSP",
    "status": "completed_no_exact_closure",
    "authoring_model": "gpt-oss:20b",
    "search_nodes": 703172,
    "frontier_controls": 390,
    "target_range": "40–90 letters",
    "exact_candidates": 0,
    "mechanically_admitted_candidates": 0,
    "longest_frontier_example": "the herald orders the tapestry in the great hall.",
    "provenance": "model authored lexical alternatives once; no candidate reward loop; no catalogue import",
    "independent_validation": ["literal outside-in two-pointer", "forward/reverse SHA-256"],
    "reader_status": "not_run; frontier controls are not exact candidates",
    "next_repair": "add a held-out lexical bank keyed by the first two residual characters",
}

THREE_BEAT_RUN = {
    "run_id": "three-beat-alias-grammar-20260919",
    "method": "three-beat finite typed grammar with live boundary aliases",
    "status": "completed_no_exact_closure",
    "rendered_candidates": 1344,
    "frontier_controls": 10,
    "target_range": "40–120 letters",
    "longest_rendered_letters": 81,
    "exact_candidates": 0,
    "mechanically_admitted_candidates": 0,
    "longest_frontier_example": "an aide rips nine memos and some men inspire Diana and the bard writes a sonnet.",
    "provenance": "three independent finite SVO beats with boundary aliases; no finished-tape reversal; no catalogue import",
    "independent_validation": ["literal two-pointer", "forward/reverse SHA-256"],
    "reader_status": "not_run; frontier controls are not exact candidates",
    "next_repair": "carry the first two-character seam obligation across the second conjunction before selecting the third beat",
}

CHARACTER_RELATIVE_RUN = {
    "run_id": "character-trie-relative-decoder-20260920",
    "method": "character-trie grammar decoder with typed relative marker/subject/verb/object transitions",
    "status": "completed_no_exact_closure",
    "search_nodes": 306725,
    "target_runs": 84,
    "target_range": "39–80 letters",
    "exact_candidates": 0,
    "mechanically_admitted_candidates": 0,
    "longest_exact_letters": 0,
    "provenance": "live character-trie boundaries and agreement state; no finished-tape reversal; no catalogue import",
    "independent_validation": ["literal outside-in two-pointer", "forward/reverse SHA-256"],
    "reader_status": "not_run; no exact candidate reached the reader gate",
    "next_repair": "add a shared-participant/anaphor state plus one adjunct edge after a valid relative closure",
}

AUTHORED_SCENE_LATTICE_RUN = {
    "run_id": "authored-scene-lattice-20260920",
    "method": "human-authored semantic scene lattice with live reverse-tape edge equations",
    "status": "completed_exact_diagnostics_with_novelty_collisions",
    "nodes": 45,
    "exact_candidates": 45,
    "novel_exact_candidates": 35,
    "prior_exact_collisions": 10,
    "mechanically_admitted_candidates": 0,
    "longest_exact_letters": 42,
    "longest_exact_example": "Was Noel an item stressed? Desserts met in a leon saw.",
    "provenance": "fresh lattice code, but reverse-compatible edge family overlaps prior run artifacts; no finished-tape reversal",
    "independent_validation": ["literal outside-in two-pointer", "forward/reverse SHA-256"],
    "reader_status": "withdrawn; exact diagnostics are not coherent prose and the longest row is a prior-run collision",
    "next_repair": "replace reverse-compatible witness fragments with fresh grammatical clause pairs and rerun novelty preflight",
}

COMPOSITIONAL_SHELL_RUN = {
    "run_id": "compositional-shell-seam-dp-20260920",
    "method": "compositional shell seam DP over independently authored complete SVO clauses",
    "status": "completed_no_exact_closure",
    "visited": 21660,
    "retained_intact_controls": 164,
    "exact_candidates": 0,
    "longest_retained_letters": 75,
    "best_seam_match_chars": 4,
    "longest_retained_example": "Some singers praise the dawn, while a friend follows the road, while an aide reads nine memos.",
    "provenance": "complete grammatical shells joined with live seam obligations; no finished-tape reversal; no catalogue import",
    "independent_validation": ["literal outside-in two-pointer", "forward/reverse SHA-256"],
    "reader_status": "not_run; controls are intact English but not exact candidates",
    "next_repair": "apply seam-conditioned inflectional substitutions within one complete shell while preserving semantic roles",
}

SHELL_INFLECTION_RUN = {
    "run_id": "shell-inflection-seam-repair-20260920",
    "method": "seam-conditioned role-preserving lexical and inflectional substitution inside one complete SVO shell",
    "status": "completed_no_exact_closure",
    "visited": 11340,
    "retained_controls": 420,
    "exact_candidates": 0,
    "longest_retained_letters": 27,
    "best_seam_match_chars": 4,
    "longest_retained_example": "Some scribe inspires some memos.",
    "provenance": "complete SVO shell with agreement-aware substitutions; no finished-tape reversal; no catalogue import",
    "independent_validation": ["literal outside-in two-pointer", "forward/reverse SHA-256"],
    "reader_status": "not_run; controls are intact but below the exact reader gate",
    "next_repair": "add agreement-carrying adjunct slots while preserving one-shell semantics",
}

AUTHORED_GRAMMATICAL_PAIR_RUN = {
    "run_id": "authored-grammatical-clause-pairs-20260920",
    "method": "fresh authored grammatical clause-pair lattice with live character equations",
    "status": "completed_exact_diagnostics",
    "nodes": 49,
    "exact_candidates": 49,
    "mechanically_admitted_candidates": 0,
    "longest_exact_letters": 30,
    "longest_exact_example": "Stressed, Deliver; reviled, desserts.",
    "provenance": "fresh reversible lexical pairs; no catalogue import; no finished-tape reversal",
    "independent_validation": ["literal outside-in two-pointer", "forward/reverse SHA-256"],
    "reader_status": "withdrawn; exact rows are fragmentary and below the target length",
    "next_repair": "require complete finite subject/verb clauses before reverse-compatible pairs enter the lattice",
}

ANCHOR_OVERHANG_RUN = {
    "run_id": "anchor-preserving-overhang-20260919",
    "method": "center-out overhang expansion with cached character debt around the 38-letter anchor",
    "status": "exact_frontier_not_admissible",
    "letters": 132,
    "exact_candidates": 1,
    "mechanically_admitted_candidates": 0,
    "rendered": "Name not left onto her a. Set add new one last one. Can all its an aide rips nine memos some men inspire Diana still an. Ace not sale now end dates. Are hot not felt one man.",
    "rejection": "the complete 38-letter anchor is a proper self-palindromic span and a content word repeats",
    "provenance": "center-out extension; no claim of readable prose; no reader package",
    "independent_validation": ["literal outside-in two-pointer", "forward/reverse SHA-256"],
    "reader_status": "withdrawn shortcut; not reader-worthy",
    "next_repair": "forbid anchor embedding before any overhang expansion and move substantive search to a new sentence structure",
}

BIDIRECTIONAL_HALF_TAPE_RUN = {
    "run_id": "astra-bidirectional-half-tape-20260920",
    "method": "bidirectional typed half-tape CSP with constrained-edge expansion",
    "status": "completed_no_exact_closure_above_anchor",
    "target_range": "39–52 letters",
    "grammar_families": 5,
    "cells_completed": 70,
    "edge_attempts": 2421192,
    "search_nodes": 10460,
    "budget_exhausted_cells": 0,
    "exact_candidates_above_38": 0,
    "mechanically_admitted_candidates_above_38": 0,
    "regression": "An aide rips nine memos; some men inspire Diana.",
    "provenance": "remote deterministic replay; no model calls; no finished-tape reversal; no catalogue import",
    "independent_validation": ["literal outside-in two-pointer", "forward/reverse SHA-256", "miniature exhaustive oracle"],
    "reader_status": "not_run; no new exact candidate reached the reader gate",
    "next_repair": "change sentence structure or lexical boundary possibilities instead of allocating more time to exhausted cells",
}

AGREEMENT_ADJUNCT_RUN = {
    "run_id": "agreement-carrying-adjunct-center-csp-20260920",
    "method": "agreement-carrying semantic SVO shell with one temporal/locative slot",
    "status": "completed_no_exact_closure",
    "target_range": "39–70 letters",
    "semantic_shell_specs": 16800,
    "outer_equation_pruned": 15120,
    "inner_states": 1680,
    "rendered_controls": 24,
    "exact_candidates": 0,
    "mechanically_admitted_candidates": 0,
    "longest_control_letters": 47,
    "best_control": "The quiet poet guards an open journal while he waits.",
    "provenance": "remote deterministic semantic domains; no finished-tape reversal; no catalogue import; no RLAIF search reward",
    "independent_validation": ["literal outside-in two-pointer", "forward/reverse SHA-256", "shared mechanical admission gate"],
    "reader_status": "not_run; controls are intact prose but not exact candidates",
    "next_repair": "carry one held-out agreement-compatible verb/object edge into the residual character state",
}

FINITE_CLAUSE_ORBIT_RUN = {
    "run_id": "finite-clause-character-orbits-20260919",
    "method": "finite SVO character-orbit product over two complete clause tries",
    "status": "completed_no_exact_closure",
    "expanded_orbit_states": 8,
    "matched_orbit_transitions": 7,
    "rejected_orbit_transitions": 5,
    "exact_candidates": 0,
    "mechanically_admitted_candidates": 0,
    "intact_controls": 3,
    "longest_control_letters": 48,
    "longest_control": "A patient keeper guards charts; The baker records a sonnet.",
    "provenance": "remote deterministic complete finite-SVO tries; no reversible lexical pairs; no finished-tape reversal; no catalogue import",
    "independent_validation": ["literal outside-in two-pointer", "forward/reverse SHA-256", "remote replay SHA-256"],
    "reader_status": "not_run; controls are intact prose but not exact candidates",
    "next_repair": "add one held-out subject/object noun bundle at the first live orbit frontier",
}

TWO_SIDED_SEMANTIC_ORBIT_RUN = {
    "run_id": "two-sided-semantic-orbit-product-20260920",
    "method": "fresh center-out semantic story grammar with a live two-sided character-orbit product",
    "status": "completed_no_exact_closure",
    "target_range": "39–60 letters",
    "path_budget_per_side": 12000,
    "left_paths": 12000,
    "right_paths": 10200,
    "expanded_product_states": 34,
    "matched_orbit_transitions": 20,
    "rejected_orbit_transitions": 140,
    "max_orbit_depth": 3,
    "exact_candidates": 0,
    "mechanically_admitted_candidates": 0,
    "intact_controls": 4,
    "longest_control_letters": 41,
    "controls": [
        "A baker carries a map; an artist answers some bells.",
        "A baker carries a map; an artist follows an archive.",
        "A baker carries a map; an artist follows some roads.",
        "A baker carries a map; an artist hears some ledgers.",
    ],
    "provenance": "remote deterministic independent semantic path banks; grammar boundaries and mirrored character orbits selected before rendering; no repair, finished-tape reversal, catalogue import, or RLAIF reward",
    "independent_validation": [
        "live two-sided orbit product",
        "literal outside-in two-pointer",
        "forward/reverse SHA-256",
        "mechanical admission gate",
    ],
    "reader_status": "not_run; controls are intact prose but no exact candidate reached the reader gate",
    "next_construction_discriminator": "add one held-out complete story frame at the first dead frontier and rerun the same center-out product; do not edit a rendered tape",
}

TWO_SIDED_SETTING_FRAME_RUN = {
    "run_id": "two-sided-semantic-orbit-product-setting-frame-20260920",
    "method": "held-out initial-setting semantic story frame with the same live center-out character-orbit product",
    "status": "completed_no_exact_closure",
    "target_range": "39–60 letters",
    "path_budget_per_side": 2000,
    "left_paths": 2000,
    "right_paths": 2000,
    "expanded_product_states": 37,
    "matched_orbit_transitions": 20,
    "rejected_orbit_transitions": 201,
    "max_orbit_depth": 3,
    "exact_candidates": 0,
    "mechanically_admitted_candidates": 0,
    "intact_controls": 4,
    "longest_control_letters": 59,
    "controls": [
        "At the gate a baker carries a memo; over the river an artist answers a bell.",
        "At the gate a baker carries a memo; on the shore an artist answers a ledger.",
        "At the gate a baker carries a memo; on the tower an artist answers a ledger.",
        "At the gate a baker carries a memo; on the river an artist answers a ledger.",
    ],
    "provenance": "remote deterministic held-out initial-setting frame; grammar boundaries, setting determiner, semantic roles, and mirrored character orbits selected before rendering; no repair, finished-tape reversal, catalogue import, or RLAIF reward",
    "independent_validation": [
        "live two-sided center-out orbit product",
        "literal outside-in two-pointer",
        "forward/reverse SHA-256",
        "mechanical admission gate",
    ],
    "reader_status": "not_run; controls are intact prose but no exact candidate reached the reader gate",
    "next_construction_discriminator": "retain the precise setting-preposition/OBJECT frontier; add no further frame until this boundary's support is independently falsified",
}

READER_PACKAGE = {
    "experiment_id": "reader-package-v4-20260919",
    "status": "blinded_package_ready_human_ratings_pending",
    "seed": 20260919,
    "pair_count": 5,
    "conditions": ["intact", "word-shuffled_control"],
    "randomized_blinded_order": True,
    "answer_key_separated": True,
    "programmatic_metrics_certify_readability": False,
    "next_action": "collect independent ratings and report pairwise preference with rater IDs and exclusions",
}

OPTIMIZATION_SPEC = {
    "objective_order": [
        "exact letter-level closure",
        "intact grammatical constituents with a concrete scene",
        "longer rendered tape",
        "human-rated readability and dramatic cadence",
    ],
    "hard_exclusions": [
        "word-order-only symmetry",
        "repeated or self-palindromic units",
        "borrowed catalogue text",
        "fragmentary or gibberish output",
    ],
    "promotion_rule": "Historical diagnostics cannot certify readability or drive post-hoc edits; each active candidate must be generated on the exact character orbit, and promotion requires randomized blinded intact-prose versus shuffled-control readers.",
    "active_construction_policy": "Exact-by-construction orbit generation: choose grammar boundaries and mirrored character pairs together; residual repair lanes are historical diagnostics, not the primary search.",
    "current_search": "two-sided-semantic-orbit-product-setting-frame-20260920",
    "search_history": [
        "half-tape-grammar-csp-20260919",
        "dream-rsi-strict-phrase-bank-20260919",
        "semantic-shell-growth-20260919",
        "half-tape-indexed-path-csp-20260919",
        "connector-character-product-20260919",
        "residual-seam-scene-lattice-20260919",
        "residual-terminal-repair-lattice-20260919",
        "relative-indexed-boundary-csp-20260919",
        "shared-relative-complement-csp-20260919",
        "residual-prefix2-attachment-lattice-20260919",
        "llm-authored-typed-bank-csp-20260919",
        "three-beat-alias-grammar-20260919",
        "finite-clause-character-orbits-20260919",
        "character-trie-relative-decoder-20260920",
        "authored-scene-lattice-20260920",
        "compositional-shell-seam-dp-20260920",
        "shell-inflection-seam-repair-20260920",
        "authored-grammatical-clause-pairs-20260920",
        "anchor-preserving-overhang-20260919",
        "astra-bidirectional-half-tape-20260920",
        "agreement-carrying-adjunct-center-csp-20260920",
        "two-sided-semantic-orbit-product-20260920",
        "two-sided-semantic-orbit-product-setting-frame-20260920",
    ],
}


class EvaluationRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2_000)
    use_lm: bool = False


def _letter_tape(text: str) -> str:
    """Normalize independently of the project's validator implementation."""
    if any(ch.isalpha() and not ch.isascii() for ch in text):
        raise ValueError("only ASCII alphabetic characters are supported")
    return "".join(ch for ch in text.casefold() if "a" <= ch <= "z")


def _two_pointer_palindrome(tape: str) -> bool:
    left, right = 0, len(tape) - 1
    while left < right:
        if tape[left] != tape[right]:
            return False
        left += 1
        right -= 1
    return bool(tape)


def independent_audit(text: str) -> dict[str, Any]:
    """Return an audit that does not call ``validator.is_palindrome``."""
    try:
        tape = _letter_tape(text)
        exact = _two_pointer_palindrome(tape)
        forward_sha = hashlib.sha256(tape.encode("ascii")).hexdigest()
        reverse_sha = hashlib.sha256(tape[::-1].encode("ascii")).hexdigest()
    except ValueError as exc:
        return {
            "exact": False,
            "independent_two_pointer": False,
            "letters": 0,
            "error": str(exc),
        }
    return {
        "exact": exact,
        "independent_two_pointer": exact,
        "letters": len(tape),
        "normalized": tape,
        "sha256_forward": forward_sha,
        "sha256_reverse": reverse_sha,
        "sha256_match": forward_sha == reverse_sha,
    }


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.casefold())


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _rlaif_diagnostic(text: str, checks: Mapping[str, bool], audit: Mapping[str, Any]) -> dict[str, Any]:
    """Give repair-oriented language feedback without pretending to be human data.

    The rubric deliberately prefers a concrete scene, grammatical cadence,
    and varied syntax—the qualities a Shakespearean line needs—while keeping
    every score explicitly diagnostic.  It is not a trained reward model and
    never promotes an item to readable output.
    """
    words = _word_tokens(text)
    content = [
        word for word in words
        if word not in {"a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "is", "are", "was", "were", "some"}
    ]
    lengths = [float(len(word)) for word in words]
    lexical = bool(words) and bool(checks.get("lexicon_words"))
    scene = min(1.0, len(set(content)) / 6.0) if content else 0.0
    cadence = min(1.0, (_std(lengths) / 3.0) + (0.15 if re.search(r"[,;:!?]", text) else 0.0))
    grammar = 1.0 if checks.get("word_form") and lexical else 0.35 if words else 0.0
    exactness = 1.0 if audit.get("exact") else 0.0
    # These axes are deliberately interpretable Shakespearean craft prompts:
    # image, agency, turn, and cadence.  They are a repair rubric, not a
    # reward model and not a substitute for a reader response.
    concrete = {"aide", "memos", "men", "diana", "bard", "rose", "shore", "moon", "river"}
    image = min(1.0, len(set(content) & concrete) / 3.0) if content else 0.0
    agency = 1.0 if re.search(r"\b(?:a|an|the|some)\s+\w+\s+\w+", text.casefold()) else 0.0
    turn = 1.0 if re.search(r"[;:!?]", text) and len(words) >= 7 else 0.35 if words else 0.0
    dramatic = round((image + agency + turn + cadence) / 4.0, 3)

    if not audit.get("exact"):
        feedback = "Repair the letter tape first; language judgements are premature until closure is exact."
    elif scene < 0.5:
        feedback = (
            "Exact closure is mechanically real, but the line is semantically thin: replace abstract or repeated slots "
            "with a named actor, a concrete object, and one consequential Shakespearean action before any reader test."
        )
    elif len(audit.get("normalized", "")) < 80:
        feedback = (
            "Compact dramatic image, but not yet a full Shakespearean movement: preserve the aide/memos/Diana scene "
            "while extending it with a subject-led clause, a strong verb, and a consequential second beat."
        )
    else:
        feedback = "The scene and cadence are promising; test this intact prose against blinded readers before promotion."

    return {
        "status": "diagnostic_only",
        "framework": "RLAIF-inspired Shakespearean repair rubric",
        "certifies_readability": False,
        "human_evidence_required": True,
        "scores": {
            "exactness": round(exactness, 3),
            "lexical_surface": round(1.0 if lexical else 0.0, 3),
            "scene_specificity": round(scene, 3),
            "cadence": round(cadence, 3),
            "grammatical_surface": round(grammar, 3),
            "shakespearean_image": round(image, 3),
            "shakespearean_agency": round(agency, 3),
            "shakespearean_turn": round(turn, 3),
            "dramatic_cadence_diagnostic": dramatic,
        },
        "strengths": [
            "concrete actors and objects" if image >= 0.67 else "some concrete imagery",
            "subject-led action" if agency else "no stable subject-led action yet",
            "a visible turn or beat boundary" if turn >= 0.75 else "no clear dramatic turn yet",
        ],
        "repairs": [
            "extend the scene with a second consequential beat while preserving the live character seam",
            "keep any added clause independently grammatical and reader-testable",
        ],
        "feedback": feedback,
        "next_reader_facing_test": "randomized blinded intact-prose versus shuffled-control rating",
    }


def _evaluate(text: str, *, use_lm: bool = False) -> dict[str, Any]:
    audit = independent_audit(text)
    try:
        checks = mechanical_admission_checks(text, min_letters=30, max_letters=2_000)
    except (TypeError, ValueError):
        checks = {}
    result: dict[str, Any] = {
        "candidate": {
            "rendered": text,
            "provenance": "submitted_to_diagnostic; not accepted output",
            "audit": audit,
            "mechanical_checks": checks,
        },
        "rlaif": _rlaif_diagnostic(text, checks, audit),
        "promotion": {
            "status": "gated",
            "reader_status": "not_run",
            "reason": GATE_MESSAGE,
        },
    }
    if use_lm:
        try:
            from llm_palindrome.lm_scoring import GPT2Scorer

            result["language_model"] = {
                "status": "diagnostic_only",
                "model": "gpt2",
                "score": GPT2Scorer("gpt2", device="cpu").score_texts([text])[0],
                "certifies_readability": False,
            }
        except Exception as exc:  # model download/runtime is optional
            result["language_model"] = {
                "status": "unavailable",
                "error": f"{type(exc).__name__}: {exc}",
                "certifies_readability": False,
            }
    return result


def _best_known_record() -> dict[str, Any]:
    evaluation = _evaluate(BEST_KNOWN_TEXT)
    return {
        "rendered": BEST_KNOWN_TEXT,
        "letters": evaluation["candidate"]["audit"]["letters"],
        "provenance": BEST_KNOWN_PROVENANCE,
        "audit": evaluation["candidate"]["audit"],
        "mechanical_checks": evaluation["candidate"]["mechanical_checks"],
        "rlaif": evaluation["rlaif"],
        "promotion_status": "gated_pending_blinded_readers",
    }


def _rlaif_frontier() -> list[dict[str, Any]]:
    """Compare actual rendered rows without turning a proxy into a gate."""
    rows = [
        {
            "run_id": BEST_KNOWN_PROVENANCE["run_id"],
            "role": "best_known_reader_plausible",
            "rendered": BEST_KNOWN_TEXT,
            "provenance": BEST_KNOWN_PROVENANCE["method"],
        },
        *[
            {
                "run_id": row["provenance"].split(";", 1)[0],
                "role": "repair_frontier",
                "rendered": row["rendered"],
                "provenance": row["provenance"],
            }
            for row in DREAM_RSI_REPAIR_FRONTIER
        ],
        {
            "run_id": AUTHORED_SCENE_LATTICE_RUN["run_id"],
            "role": "withdrawn_exact_diagnostic",
            "rendered": AUTHORED_SCENE_LATTICE_RUN["longest_exact_example"],
            "provenance": AUTHORED_SCENE_LATTICE_RUN["provenance"],
        },
        {
            "run_id": COMPOSITIONAL_SHELL_RUN["run_id"],
            "role": "longest_intact_control",
            "rendered": COMPOSITIONAL_SHELL_RUN["longest_retained_example"],
            "provenance": COMPOSITIONAL_SHELL_RUN["provenance"],
        },
        {
            "run_id": AUTHORED_GRAMMATICAL_PAIR_RUN["run_id"],
            "role": "fragmentary_exact_diagnostic",
            "rendered": AUTHORED_GRAMMATICAL_PAIR_RUN["longest_exact_example"],
            "provenance": AUTHORED_GRAMMATICAL_PAIR_RUN["provenance"],
        },
        {
            "run_id": SHELL_INFLECTION_RUN["run_id"],
            "role": "short_intact_control",
            "rendered": SHELL_INFLECTION_RUN["longest_retained_example"],
            "provenance": SHELL_INFLECTION_RUN["provenance"],
        },
        {
            "run_id": ANCHOR_OVERHANG_RUN["run_id"],
            "role": "withdrawn_anchor_embedded_frontier",
            "rendered": ANCHOR_OVERHANG_RUN["rendered"],
            "provenance": ANCHOR_OVERHANG_RUN["provenance"],
        },
        {
            "run_id": AGREEMENT_ADJUNCT_RUN["run_id"],
            "role": "agreement_carrying_intact_control",
            "rendered": AGREEMENT_ADJUNCT_RUN["best_control"],
            "provenance": AGREEMENT_ADJUNCT_RUN["provenance"],
        },
        {
            "run_id": FINITE_CLAUSE_ORBIT_RUN["run_id"],
            "role": "finite_clause_intact_control",
            "rendered": FINITE_CLAUSE_ORBIT_RUN["longest_control"],
            "provenance": FINITE_CLAUSE_ORBIT_RUN["provenance"],
        },
        {
            "run_id": TWO_SIDED_SEMANTIC_ORBIT_RUN["run_id"],
            "role": "center_out_intact_control",
            "rendered": TWO_SIDED_SEMANTIC_ORBIT_RUN["controls"][0],
            "provenance": TWO_SIDED_SEMANTIC_ORBIT_RUN["provenance"],
        },
        {
            "run_id": TWO_SIDED_SETTING_FRAME_RUN["run_id"],
            "role": "center_out_setting_frame_control",
            "rendered": TWO_SIDED_SETTING_FRAME_RUN["controls"][0],
            "provenance": TWO_SIDED_SETTING_FRAME_RUN["provenance"],
        },
    ]
    comparison = []
    for row in rows:
        evaluation = _evaluate(row["rendered"])
        comparison.append(
            {
                **row,
                "letters": evaluation["candidate"]["audit"].get("letters", 0),
                "exact": evaluation["candidate"]["audit"].get("exact", False),
                "mechanically_admitted": all(evaluation["candidate"]["mechanical_checks"].values()),
                "rlaif": evaluation["rlaif"],
                "next_reader_facing_test": evaluation["rlaif"]["next_reader_facing_test"],
                "promotion": evaluation["promotion"],
            }
        )
    return comparison


@router.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "version": "v4",
        "mode": "evidence_and_diagnostics",
        "gate": {
            "generation": "gated",
            "reader_evidence": False,
            "human_certification_required": True,
        },
        "best_known_letters": 38,
        "optimization": OPTIMIZATION_SPEC,
    }


@router.get("/evidence")
def evidence() -> dict[str, Any]:
    """Expose the current frontier without presenting it as certified output."""
    return {
        "version": "v4",
        "status": "evidence_only",
        "gate": {
            "generation": "gated",
            "reader_evidence": False,
            "human_certification_required": True,
        },
        "best_known": _best_known_record(),
        "repair_frontier": DREAM_RSI_REPAIR_FRONTIER,
        "method_runs": [SEMANTIC_SHELL_RUN, INDEXED_PATH_RUN, CONNECTOR_PRODUCT_RUN, RESIDUAL_SEAM_RUN, RESIDUAL_TERMINAL_RUN, RELATIVE_INDEXED_RUN, SHARED_RELATIVE_RUN, RESIDUAL_PREFIX_RUN, LLM_TYPED_BANK_RUN, THREE_BEAT_RUN, FINITE_CLAUSE_ORBIT_RUN, CHARACTER_RELATIVE_RUN, AUTHORED_SCENE_LATTICE_RUN, COMPOSITIONAL_SHELL_RUN, SHELL_INFLECTION_RUN, AUTHORED_GRAMMATICAL_PAIR_RUN, ANCHOR_OVERHANG_RUN, BIDIRECTIONAL_HALF_TAPE_RUN, AGREEMENT_ADJUNCT_RUN, TWO_SIDED_SEMANTIC_ORBIT_RUN, TWO_SIDED_SETTING_FRAME_RUN],
        "rlaif_frontier": _rlaif_frontier(),
        "ai_feedback_run": AI_FEEDBACK_RUN,
        "reader_package": READER_PACKAGE,
        "optimization": OPTIMIZATION_SPEC,
    }


@router.get("/candidate")
def candidate() -> dict[str, Any]:
    """Alias for clients that call the evidence item a candidate."""
    return evidence()


@router.get("/method")
def method() -> dict[str, Any]:
    """Expose the constructive objective and its evidence gate."""
    return {
        "version": "v4",
        "status": "constructive_search_in_progress",
        "optimization": OPTIMIZATION_SPEC,
        "current_best": _best_known_record(),
        "repair_frontier": DREAM_RSI_REPAIR_FRONTIER,
        "method_runs": [SEMANTIC_SHELL_RUN, INDEXED_PATH_RUN, CONNECTOR_PRODUCT_RUN, RESIDUAL_SEAM_RUN, RESIDUAL_TERMINAL_RUN, RELATIVE_INDEXED_RUN, SHARED_RELATIVE_RUN, RESIDUAL_PREFIX_RUN, LLM_TYPED_BANK_RUN, THREE_BEAT_RUN, FINITE_CLAUSE_ORBIT_RUN, CHARACTER_RELATIVE_RUN, AUTHORED_SCENE_LATTICE_RUN, COMPOSITIONAL_SHELL_RUN, SHELL_INFLECTION_RUN, AUTHORED_GRAMMATICAL_PAIR_RUN, ANCHOR_OVERHANG_RUN, BIDIRECTIONAL_HALF_TAPE_RUN, AGREEMENT_ADJUNCT_RUN, TWO_SIDED_SEMANTIC_ORBIT_RUN, TWO_SIDED_SETTING_FRAME_RUN],
        "rlaif_frontier": _rlaif_frontier(),
        "ai_feedback_run": AI_FEEDBACK_RUN,
        "reader_package": READER_PACKAGE,
    }


@router.get("/frontier-evaluation")
def frontier_evaluation() -> dict[str, Any]:
    """Return the repair comparison used by the next construction decision."""
    return {
        "version": "v4",
        "status": "diagnostic_only",
        "certifies_readability": False,
        "human_evidence_required": True,
        "rows": _rlaif_frontier(),
        "method_run": TWO_SIDED_SETTING_FRAME_RUN,
        "method_runs": [SEMANTIC_SHELL_RUN, INDEXED_PATH_RUN, CONNECTOR_PRODUCT_RUN, RESIDUAL_SEAM_RUN, RESIDUAL_TERMINAL_RUN, RELATIVE_INDEXED_RUN, SHARED_RELATIVE_RUN, RESIDUAL_PREFIX_RUN, LLM_TYPED_BANK_RUN, THREE_BEAT_RUN, FINITE_CLAUSE_ORBIT_RUN, CHARACTER_RELATIVE_RUN, AUTHORED_SCENE_LATTICE_RUN, COMPOSITIONAL_SHELL_RUN, SHELL_INFLECTION_RUN, AUTHORED_GRAMMATICAL_PAIR_RUN, ANCHOR_OVERHANG_RUN, BIDIRECTIONAL_HALF_TAPE_RUN, AGREEMENT_ADJUNCT_RUN, TWO_SIDED_SEMANTIC_ORBIT_RUN, TWO_SIDED_SETTING_FRAME_RUN],
        "ai_feedback_run": AI_FEEDBACK_RUN,
        "reader_package": READER_PACKAGE,
        "next_reader_facing_test": "randomized blinded intact-prose versus shuffled-control rating",
    }


@router.get("/best-evaluation")
def best_evaluation(use_lm: bool = Query(False)) -> dict[str, Any]:
    """Run the deterministic Shakespearean repair rubric on the frontier item."""
    return _evaluate(BEST_KNOWN_TEXT, use_lm=use_lm)


@router.post("/evaluate")
def evaluate(payload: EvaluationRequest) -> dict[str, Any]:
    return _evaluate(payload.text, use_lm=payload.use_lm)


@router.get("/evaluate")
def evaluate_query(
    text: str = Query(..., min_length=1, max_length=2_000),
    use_lm: bool = Query(False),
) -> dict[str, Any]:
    return _evaluate(text, use_lm=use_lm)


@router.api_route("/generate", methods=["GET", "POST"])
def generate() -> None:
    raise HTTPException(status_code=503, detail=GATE_MESSAGE)
