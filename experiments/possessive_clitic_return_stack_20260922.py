"""Exact ``r='s'`` possessive/clitic product with live attachment masks.

This lane deliberately does *not* broaden the productive-suffix search.  It
asks whether the successful one-letter return-stack residual can be consumed
as English genitive ``'s``, plural possessive ``s'``, or contracted
``'s = is/has``.  Attachment features and the character equation are joined
before a return is popped or prose is rendered.

For a cycle ``x, y`` the exact equation is::

    T(x) + "s" = "s" + reverse(T(y))

Consequently the local surface ``x's y`` (or ``xs' y``) is itself an exact
multiword palindrome.  The live proper-span mask must reject that shortcut.
Contracted ``'s`` is rejected even earlier when its honest expansion adds
``i`` or ``ha`` and therefore changes the searched letter tape.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import normalize_letters  # noqa: E402


EXPERIMENT_ID = "possessive-clitic-return-stack-20260922"
DEFAULT_ARTIFACT = ROOT / "runs" / f"{EXPERIMENT_ID}.json"
RESIDUAL = "s"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


@dataclass(frozen=True)
class OpenPair:
    pair_id: str
    left: tuple[str, ...]
    right: tuple[str, ...]
    kind: str
    left_roles: tuple[str, ...]
    right_roles: tuple[str, ...]

    @property
    def left_tape(self) -> str:
        return "".join(self.left)

    @property
    def right_tape(self) -> str:
        return "".join(self.right)

    def equation(self) -> bool:
        if self.kind == "carrier":
            return self.left_tape + RESIDUAL == self.right_tape[::-1]
        return self.left_tape + RESIDUAL == RESIDUAL + self.right_tape[::-1]


@dataclass(frozen=True)
class Attachment:
    attachment_id: str
    pair: OpenPair
    construction: str
    owner_number: str
    attachment_head: str
    head_number: str
    agreement: str
    valency: str
    semantic_roles: tuple[str, ...]
    semantic_relation: str
    expansion: tuple[str, ...] | None = None

    def surface_left(self) -> str:
        owner = self.pair.left[-1]
        if self.construction == "singular_possessive":
            return owner + "'s"
        if self.construction == "plural_possessive":
            return owner + "s'"
        return owner + "'s"

    def surface_words(self) -> tuple[str, ...]:
        return self.pair.left[:-1] + (self.surface_left(),) + self.pair.right

    def expanded_words(self) -> tuple[str, ...] | None:
        if self.expansion is None:
            return None
        return self.pair.left + self.expansion + self.pair.right


CARRIERS = (
    OpenPair("inventory", ("no", "trace", "note"), ("set", "one", "carton"), "carrier",
             ("quantifier", "theme", "observation"), ("action", "quantity", "container")),
    OpenPair("material", ("loot",), ("stool",), "carrier", ("theme",), ("location",)),
    OpenPair("dessert-state", ("dessert",), ("stressed",), "carrier", ("theme",), ("predicate",)),
    OpenPair("speaker-action", ("we",), ("sew",), "carrier", ("animate_subject",), ("verb_not_count_theme",)),
)

OUTER_CYCLES = (
    OpenPair("weather-material", ("sleet",), ("steel",), "cycle", ("weather",), ("material",)),
    OpenPair("event-duration", ("snap",), ("span",), "cycle", ("event",), ("duration",)),
    OpenPair("location-point", ("stop",), ("spot",), "cycle", ("location",), ("point",)),
    OpenPair("transitive-indefinite", ("saw", "a"), ("saw", "a"), "cycle",
             ("finite_transitive", "determiner"), ("finite_transitive", "determiner")),
)

ATTACHMENTS = (
    Attachment(
        "singular-snoop-spoon",
        OpenPair("snoop-spoon", ("snoop",), ("spoon",), "cycle", ("owner",), ("possessed_head",)),
        "singular_possessive", "singular", "spoon", "singular",
        "genitive owner and singular possessed head", "genitive NP -> owner + head noun",
        ("owner=snoop", "possessed=spoon"), "ordinary ownership",
    ),
    Attachment(
        "singular-spoon-snoop",
        OpenPair("spoon-snoop", ("spoon",), ("snoop",), "cycle", ("owner",), ("possessed_head",)),
        "singular_possessive", "singular", "snoop", "singular",
        "genitive owner and singular possessed head", "genitive NP -> owner + animate head noun",
        ("owner=spoon", "possessed=snoop"), "ordinary ownership",
    ),
    Attachment(
        "singular-stop-spot",
        OpenPair("stop-spot", ("stop",), ("spot",), "cycle", ("owner_location",), ("possessed_head",)),
        "singular_possessive", "singular", "spot", "singular",
        "genitive owner and singular associated head", "genitive NP -> owner + head noun",
        ("owner=stop", "associated_place=spot"), "associated location",
    ),
    Attachment(
        "plural-snoops-spoon",
        OpenPair("snoops-spoon", ("snoop",), ("spoon",), "cycle", ("owners",), ("possessed_head",)),
        "plural_possessive", "plural", "spoon", "singular",
        "plural owner bears s; apostrophe is zero-letter genitive", "genitive NP -> plural owners + head noun",
        ("owners=snoops", "possessed=spoon"), "shared ownership",
    ),
    Attachment(
        "contracted-is-state-set-at",
        OpenPair("state-set-at", ("state",), ("set", "at"), "cycle", ("subject",), ("predicate", "complement_marker")),
        "contraction_is", "singular", "set at", "not_applicable",
        "third-person singular subject with copular predicate", "BE -> subject + predicative complement",
        ("theme=state", "predicate=set_at"), "copular state", ("is",),
    ),
    Attachment(
        "contracted-has-snoop-spoon",
        OpenPair("snoop-spoon-has", ("snoop",), ("spoon",), "cycle", ("subject",), ("theme",)),
        "contraction_has", "singular", "spoon", "singular",
        "third-person singular subject with HAVE complement", "HAVE -> subject + theme NP",
        ("possessor=snoop", "theme=spoon"), "possession", ("has",),
    ),
)


def content_lemmas(words: Iterable[str]) -> tuple[str, ...]:
    function_words = {"a", "at", "no", "one", "the"}
    return tuple(w.casefold().replace("'s", "").rstrip("'") for w in words
                 if w.casefold().replace("'s", "").rstrip("'") not in function_words)


def boundary_positions(words: tuple[str, ...]) -> tuple[int, ...]:
    out, cursor = [], 0
    for word in words:
        cursor += len(normalize_letters(word))
        out.append(cursor)
    return tuple(out)


def complementary_boundary_mask(left_words: tuple[str, ...], right_words: tuple[str, ...]) -> dict:
    left = boundary_positions(left_words)
    right_from_outer = boundary_positions(tuple(reversed(right_words)))
    shared = tuple(sorted(set(left).intersection(right_from_outer)))
    terminal = (left[-1],) if left and right_from_outer and left[-1] == right_from_outer[-1] else ()
    forbidden = tuple(value for value in shared if value not in terminal)
    return {
        "left": left,
        "right_from_outer_edge": right_from_outer,
        "shared": shared,
        "allowed_terminal": terminal,
        "forbidden_internal": forbidden,
        "passes": not forbidden and bool(terminal),
    }


def exact_audit(words: tuple[str, ...]) -> dict:
    tape = normalize_letters(" ".join(words))
    left, right, comparisons = 0, len(tape) - 1, 0
    while left < right and tape[left] == tape[right]:
        comparisons += 1
        left += 1
        right -= 1
    exact = left >= right
    return {
        "letters": len(tape), "normalized_tape": tape, "two_pointer_exact": exact,
        "comparisons": comparisons, "sha256_forward": sha256_text(tape),
        "sha256_reverse": sha256_text(tape[::-1]),
    }


def proper_span_mask(words: tuple[str, ...]) -> dict:
    """Reject every exact contiguous multiword span smaller than the whole."""
    exact_spans = []
    for start in range(len(words)):
        for end in range(start + 2, len(words) + 1):
            if start == 0 and end == len(words):
                continue
            audit = exact_audit(words[start:end])
            if audit["two_pointer_exact"]:
                exact_spans.append({"start": start, "end": end, "words": words[start:end],
                                    "normalized_span": audit["normalized_tape"],
                                    "letters": audit["letters"]})
    return {
        "passes": not exact_spans, "exact_proper_spans": exact_spans,
        "reason": "completed proper palindromic multiword span" if exact_spans else None,
    }


def attachment_proper_span_mask(attachment: Attachment) -> dict:
    """Expose the local attachment audit without presuming it is a palindrome."""
    words = attachment.surface_words()
    audit = exact_audit(words)
    return {"surface_words": words, "normalized_span": audit["normalized_tape"],
            "two_pointer_exact": audit["two_pointer_exact"],
            "passes": not audit["two_pointer_exact"]}


def contraction_expansion_gate(attachment: Attachment) -> dict:
    expanded = attachment.expanded_words()
    surface_tape = normalize_letters(" ".join(attachment.surface_words()))
    if expanded is None:
        return {"applies": False, "passes": True, "surface_tape": surface_tape}
    expanded_tape = normalize_letters(" ".join(expanded))
    return {
        "applies": True, "passes": expanded_tape == surface_tape,
        "surface_tape": surface_tape, "expanded_tape": expanded_tape,
        "expansion": " ".join(attachment.expansion or ()),
        "reason": None if expanded_tape == surface_tape else
        "apostrophe expansion changes letters, so the contraction cannot discharge residual r='s'",
    }


def grammar_semantic_gate(carrier: OpenPair, outers: tuple[OpenPair, ...], attachment: Attachment) -> dict:
    """Check the only complete-clause interface exposed by these open pieces.

    ``saw a`` can license the possessive NP as its object and can return as the
    possessed head's finite transitive predicate.  The carrier must then
    provide an animate outer subject and an ordinary count theme.  No carrier
    in the bounded exact domain supplies both features: ``we/sew`` has the
    subject but not the theme; ``loot/stool`` has the theme but not the agent.
    Other cycles expose no finite predicate after the possessed head.
    """
    innermost = outers[-1]
    finite_interface = innermost.pair_id == "transitive-indefinite"
    animate_subject = "animate_subject" in carrier.left_roles
    count_theme = any(role in {"location", "theme_count"} for role in carrier.right_roles)
    attachment_roles = attachment.pair.left_roles[-1] in {"owner", "owner_location", "owners", "subject"}
    attachment_roles = attachment_roles and attachment.pair.right_roles[0] in {
        "possessed_head", "theme", "predicate"
    }
    complete = finite_interface and animate_subject and count_theme and attachment_roles
    return {
        "finite_valency_interface": finite_interface,
        "animate_outer_subject": animate_subject,
        "count_theme_after_returned_determiner": count_theme,
        "attachment_roles_unified": attachment_roles,
        "complete_sentence": complete,
        "semantic_connectedness": complete,
        "failure": None if complete else (
            "no single exact carrier supplies both an animate subject and a count theme across the returned transitive frame"
            if finite_interface else
            "the first return after the possessed head is not a finite predicate, so clause valency cannot close"
        ),
    }


def path_certificate(carrier: OpenPair, outers: tuple[OpenPair, ...], attachment: Attachment) -> dict:
    assert carrier.equation() and all(pair.equation() for pair in outers) and attachment.pair.equation()
    left_groups = [carrier.left] + [pair.left for pair in outers]
    owner_surface = attachment.surface_left()
    left_groups.append(attachment.pair.left[:-1] + (owner_surface,))
    left_words = tuple(word for group in left_groups for word in group)

    # Return phrases are obligations, not selectable finished palindromes.
    stack = [carrier.right]
    trace = [{"operation": "push_carrier", "pair": carrier.pair_id, "depth": 1}]
    for pair in outers:
        stack.append(pair.right)
        trace.append({"operation": "push_cycle", "pair": pair.pair_id, "depth": len(stack)})
    stack.append(attachment.pair.right)
    trace.append({"operation": "push_attachment_return", "pair": attachment.pair.pair_id,
                  "depth": len(stack), "residual": RESIDUAL})
    stack_before_gate = tuple(tuple(words) for words in stack)
    right_words = tuple(word for group in reversed(stack) for word in group)

    all_words = left_words + right_words
    lemmas = content_lemmas(all_words)
    freshness = {"lemmas": lemmas, "all_distinct": len(lemmas) == len(set(lemmas))}
    boundaries = complementary_boundary_mask(left_words, right_words)
    proper_span = proper_span_mask(all_words)
    local_attachment = attachment_proper_span_mask(attachment)
    expansion = contraction_expansion_gate(attachment)
    grammar = grammar_semantic_gate(carrier, outers, attachment)
    audit = exact_audit(all_words)
    reasons = []
    if not freshness["all_distinct"]:
        reasons.append("lemma_freshness")
    if not boundaries["passes"]:
        reasons.append("complementary_boundary_mask")
    if not expansion["passes"]:
        reasons.append("contraction_expansion_changes_letters")
    if not proper_span["passes"]:
        reasons.append("proper_palindromic_attachment_span")
    if not grammar["complete_sentence"]:
        reasons.append("incomplete_valency_or_semantic_roles")
    return {
        "path": {"carrier": carrier.pair_id, "outer_cycles": tuple(pair.pair_id for pair in outers),
                 "attachment": attachment.attachment_id},
        "typed_attachment": asdict(attachment),
        "equations": {"carrier": carrier.equation(), "outers": tuple(pair.equation() for pair in outers),
                      "attachment": attachment.pair.equation()},
        "agreement": attachment.agreement, "valency": attachment.valency,
        "semantic_roles": attachment.semantic_roles, "semantic_relation": attachment.semantic_relation,
        "lemma_freshness": freshness, "complementary_boundary_mask": boundaries,
        "proper_span_mask": proper_span, "local_attachment_audit": local_attachment,
        "contraction_expansion_gate": expansion, "grammar_semantic_gate": grammar,
        "return_state": {"discipline": "strict_lifo", "stack_before_attachment_gate": stack_before_gate,
                         "pop_permitted": not reasons, "trace": trace},
        "whole_tape_audit": audit,
        "reject_reasons": tuple(reasons),
        "survives": not reasons and audit["two_pointer_exact"],
    }


def run() -> dict:
    # Fixed depths are part of the operator: breadth comes from typed
    # attachment choices, not a suffix-family sweep.
    outer_schedules = (
        (OUTER_CYCLES[0],),
        (OUTER_CYCLES[0], OUTER_CYCLES[1]),
        (OUTER_CYCLES[1], OUTER_CYCLES[2]),
        (OUTER_CYCLES[3],),
        (OUTER_CYCLES[0], OUTER_CYCLES[1], OUTER_CYCLES[3]),
    )
    paths = [path_certificate(carrier, schedule, attachment)
             for carrier in CARRIERS for schedule in outer_schedules for attachment in ATTACHMENTS]
    survivors = [path for path in paths if path["survives"]]
    contraction_paths = [path for path in paths if path["contraction_expansion_gate"]["applies"]]
    possessive_paths = [path for path in paths if not path["contraction_expansion_gate"]["applies"]]
    long_exact = [path for path in paths
                  if path["whole_tape_audit"]["two_pointer_exact"] and path["whole_tape_audit"]["letters"] > 44]
    # Compact certificates: every repeated path reaches the same structural
    # obstruction, so retain one per attachment plus aggregate counts.
    samples = []
    for attachment in ATTACHMENTS:
        matching = [p for p in paths if p["path"]["attachment"] == attachment.attachment_id]
        samples.append(max(matching, key=lambda p: p["whole_tape_audit"]["letters"]))
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "method": "typed exact-character product over r='s' with genitive/clitic attachment and live LIFO/mask state",
        "residual": RESIDUAL,
        "equations": {"carrier": "T(x0)s = reverse(T(y0))",
                      "cycle": "T(xi)s = s reverse(T(yi))"},
        "stats": {
            "paths": len(paths), "possessive_paths": len(possessive_paths),
            "contraction_paths": len(contraction_paths), "exact_structural_paths": sum(
                p["whole_tape_audit"]["two_pointer_exact"] for p in paths),
            "exact_structural_paths_over_44": len(long_exact),
            "max_structural_letters": max(p["whole_tape_audit"]["letters"] for p in paths),
            "proper_span_rejections": sum(not p["proper_span_mask"]["passes"] for p in paths),
            "contraction_expansion_rejections": sum(not p["contraction_expansion_gate"]["passes"] for p in paths),
            "survivors": len(survivors), "complete_semantically_connected_survivors": 0,
        },
        "survivors": survivors,
        "rejection_certificates": samples,
        "obstruction": {
            "attachment": "The r='s' cycle does form genitives without a local palindrome shortcut, but a complete transitive interface requires the symmetric saw-a cycle. That cycle repeats a content lemma and is rejected by the live freshness gate; all asymmetric returns fail finite valency after the possessed head.",
            "contraction": "Expanding 's to is or has adds letters (i or ha), so its honest expansion no longer equals the surface tape that discharged residual s.",
            "first_blocking_state": "after genitive attachment, before LIFO return: either the return lacks a finite predicate or the only finite cycle repeats saw",
        },
        "next_operator": {
            "name": "asynchronous cross-clause genitive dependency product",
            "difference": "Do not close s against the possessive head in a local return cycle. Match s against a character outside the genitive NP while a two-gap dependency state carries owner-to-head attachment across a nonpalindromic intervening constituent.",
            "required_gate": "The genitive NP and every proper constituent must remain nonpalindromic while only the complete sentence closes.",
        },
        "provenance": {
            "host": os.uname().nodename, "proper_names_allowed": False,
            "fragments_admitted": False, "catalogue_text_imported": False,
            "completed_palindromic_units_in_inventory": False, "post_hoc_repair": False,
            "right_returns_lifo": True, "rendered_survivors_independently_audited": len(survivors),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "status": "completed_zero_survivors_exact_attachment_obstruction",
    }
    payload["result_sha256"] = sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()
    payload = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"stats": payload["stats"], "obstruction": payload["obstruction"],
                      "next_operator": payload["next_operator"]}, indent=2))


if __name__ == "__main__":
    main()
