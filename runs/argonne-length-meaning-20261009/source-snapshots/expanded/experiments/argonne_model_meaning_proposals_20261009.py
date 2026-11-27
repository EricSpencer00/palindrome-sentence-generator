"""Bounded CUDA model proposals, verifier feedback, and whole-text exact filters.

This is a proposed-surface-plus-exact-filter baseline, not a constrained
decoder or a meaning measurement. It differs from the previously studied
GPT-2 coupled word beam: the model here generates complete surfaces before
the independent verifier checks them. Each fresh proposal has at most one
immediate feedback retry so that a finite run covers all three proposal arms.

Only public repository examples and a credited public Reddit demonstration
are loaded. No private reader ratings, model server, or external inference API
are used. Torch and Transformers are imported only by the CUDA runner.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
DEMONSTRATION_ID = "t3_1uphsc9"
PUBLIC_POSTS = "data/reddit-palindromes/2026-10-09/posts.json"
SOURCE_SPECS = (
    {"id": "568-public", "path": "runs/incumbent-560-outer-causal-scene-20261002.json",
     "row_id": "outer-causal-scene-568-working-incumbent", "letters": 568,
     "sha256": "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"},
    {"id": "666-public", "path": "runs/incumbent-666-comparison-alternative-20260922.json",
     "row_id": "comparison-alternative-nora-sees-666", "letters": 666,
     "sha256": "cafd77235f82d9ff4f68814dc7e03d196bf719bf1ec5d541e172073502e12297"},
    {"id": "752-public", "path": "runs/repair-736-center-lexical-event-path-20260923.json",
     "row_id": None, "letters": 752,
     "sha256": "109c758ecfa0c0a2049c1dfb925d3820eb453a92836c0835d80f5106f222de3c"},
)
METHOD_NOTE = (
    "Whole-surface language-model proposal plus independent exact filter and "
    "one verifier-feedback retry; not constrained decoding. The previously "
    "studied GPT-2 coupled word beam is a different method."
)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized_letters(text: str) -> tuple[str, list[str]]:
    """Do not turn unsupported alphabetic characters into ASCII letters."""
    unsupported = sorted({char for char in text if char.isalpha() and not char.isascii()})
    return "".join(char.lower() for char in text if char.isascii() and char.isalpha()), unsupported


def audit_candidate(text: str, case: dict, references: dict[str, list[str]],
                    seen: dict[str, int]) -> dict:
    tape, unsupported = normalized_letters(text)
    mismatches = []
    for left in range(len(tape) // 2):
        right = len(tape) - 1 - left
        if tape[left] != tape[right] and len(mismatches) < 12:
            mismatches.append({
                "left_offset": left, "right_offset": right,
                "left_char": tape[left], "right_char": tape[right],
                "left_context": tape[max(0, left - 6):left + 7],
                "right_context": tape[max(0, right - 6):right + 7],
            })
    pointer_exact = bool(tape) and not mismatches
    reversal_exact = bool(tape) and tape == tape[::-1]
    supported = not unsupported
    exact = supported and pointer_exact and reversal_exact
    matched = references.get(tape, []) if tape and supported else []
    duplicate = seen.get(tape) if tape and supported else None
    length_ok = case["min_letters"] <= len(tape) <= case["max_letters"]
    return {
        "normalization": "ASCII A-Z/a-z only, lowercase; ignore case, spaces, punctuation and digits; reject unsupported alphabetic characters",
        "letters": len(tape), "unsupported_alphabetic": unsupported,
        "ascii_alphabet_supported": supported,
        "two_pointer_exact_on_ascii_projection": pointer_exact,
        "full_reversal_exact_on_ascii_projection": reversal_exact,
        "whole_text_exact": exact,
        "normalized_sha256": sha256(tape), "reverse_sha256": sha256(tape[::-1]),
        "first_mismatches": mismatches,
        "target_letters": case["target_letters"],
        "target_length_delta": len(tape) - case["target_letters"],
        "allowed_letter_range": [case["min_letters"], case["max_letters"]],
        "length_pass": length_ok,
        "reference_matches": matched, "duplicate_of_attempt": duplicate,
        "eligible_exact_proposal": exact and length_ok and not matched and duplicate is None,
        "meaning_measurement": "not_measured",
    }


def parse_response(raw: str) -> tuple[dict | None, str | None]:
    """Retain malformed responses; do not extract or repair a preferred line."""
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None, "invalid_json"
    if not isinstance(value, dict) or set(value) != {"text", "plot"}:
        return None, "expected_exact_text_plot_object"
    if not isinstance(value["text"], str) or not isinstance(value["plot"], str):
        return None, "text_and_plot_must_be_strings"
    return value, None


def load_public_inputs(root: Path = ROOT) -> tuple[list[dict], dict, dict[str, list[str]], dict]:
    sources, references, file_hashes = [], {}, {}
    for spec in SOURCE_SPECS:
        path = root / spec["path"]
        content = path.read_bytes()
        payload = json.loads(content)
        row = (next(item for item in payload["rows"] if item["id"] == spec["row_id"])
               if spec["row_id"] else payload["candidate"])
        text = row["rendered"]
        tape, unsupported = normalized_letters(text)
        if unsupported or len(tape) != spec["letters"] or sha256(tape) != spec["sha256"] or tape != tape[::-1]:
            raise ValueError(f"public_source_changed_or_invalid:{spec['id']}")
        source = {**spec, "text": text, "file_sha256": hashlib.sha256(content).hexdigest(),
                  "lineage": "existing exact construction; no human meaning validation"}
        sources.append(source)
        references.setdefault(tape, []).append(spec["id"])
        file_hashes[spec["path"]] = source["file_sha256"]
    content = (root / PUBLIC_POSTS).read_bytes()
    posts = json.loads(content)["rows"]
    file_hashes[PUBLIC_POSTS] = hashlib.sha256(content).hexdigest()
    for post in posts:
        tape, unsupported = normalized_letters(post["text"])
        if tape and not unsupported:
            references.setdefault(tape, []).append("reddit:" + post["post_id"])
    post = next(item for item in posts if item["post_id"] == DEMONSTRATION_ID)
    tape, unsupported = normalized_letters(post["text"])
    if unsupported or len(tape) < 80 or tape != tape[::-1]:
        raise ValueError("credited_demonstration_not_supported_exact_80_plus")
    demo = {key: post[key] for key in ("text", "author", "author_url", "source_url", "post_id")}
    demo.update({"letters": len(tape), "normalized_sha256": sha256(tape),
                 "source_path": PUBLIC_POSTS,
                 "credit_limit": "Credit identifies the Reddit poster; original authorship and meaning ratings are not established.",
                 "lineage": "credited demonstration; conditioning/adaptation is not independent discovery"})
    return sources, demo, references, file_hashes


def proposal_cases(sources: list[dict], demo: dict) -> list[dict]:
    plots = (
        "A courier finds a route on a map and delivers a parcel to its recipient.",
        "A keeper finds a lost animal, guides it home, and reunites it with its owner.",
        "A nurse checks a medicine label, notices a problem, and sends a corrective message.",
    )
    cases = [{"id": f"direct-{letters}", "arm": "direct",
              "target_letters": letters, "min_letters": letters, "max_letters": 2 * letters - 1,
              "intent": plots[index], "reference": None,
              "independent_novelty_claim": False}
             for index, letters in enumerate((120, 240, 480))]
    for source in sources:
        cases.append({"id": "reconstruct-" + str(source["letters"]), "arm": "reconstruction",
                      "target_letters": source["letters"], "min_letters": source["letters"],
                      "max_letters": 2 * source["letters"] - 1,
                      "intent": "Nora supplies a map to Aron; he uses it to stop dangerous water flow, with help from the others. Tell this single connected plot with stable participants.",
                      "reference": source, "independent_novelty_claim": False})
    cases.append({"id": "phrase-overhang-short", "arm": "phrase_overhang",
                  "target_letters": 96, "min_letters": 40, "max_letters": 160,
                  "intent": "A person discovers a concrete problem, acts on it, and explains the result in one connected short scene.",
                  "reference": demo, "independent_novelty_claim": False})
    return cases


def build_prompt(case: dict, variant: int, previous: dict | None = None) -> str:
    length = (f"Aim for at least {case['min_letters']} normalized ASCII letters, within the {case['min_letters']} to {case['max_letters']} letter band. Use the available length to develop the event, its relationships, and its consequences; do not add filler merely to meet the minimum."
              if case["arm"] in {"direct", "reconstruction"} else
              f"Aim for {case['target_letters']} normalized ASCII letters; use {case['min_letters']} to {case['max_letters']} without filler.")
    pieces = [
        "Return only a JSON object with exactly two string fields: text (the complete proposed passage) and plot (your description of its intended event).",
        "Propose an exact whole-passage letter palindrome in natural connected English. Ignoring nonletters and case, its full ASCII letter stream must equal its reversal. Use ASCII alphabetic characters. Spaces, punctuation, case and digits do not contribute letters.",
        length, "Intended plot: " + case["intent"],
        "Express who did what to whom and how the events relate. Avoid unrelated lists, names as filler, invented words, repeated palindromic blocks, and famous copied palindromes. Do not explain or list alternative passages. Your plot field is a proposal claim, not an evaluation.",
        f"Independent proposal variant: {variant}.",
    ]
    if case["arm"] == "reconstruction":
        pieces += [
            "The following public source is exact but has disconnected and awkward discourse. Fully reconstruct its wording and boundaries around the intended plot; retaining its letters or sentence list is unnecessary. The intended plot above is an experimental hypothesis, not a claim that this source already expresses it.",
            "Public source passage: " + case["reference"]["text"],
        ]
    elif case["arm"] == "phrase_overhang":
        ref = case["reference"]
        pieces += [
            "Phrase/overhang demonstration: opposite word and sentence boundaries can differ. Design the forced opposite phrase in its actual sentence context, carry unmatched letters across boundaries, and produce one complete exact short passage.",
            f"Credited public demonstration by {ref['author']} ({ref['source_url']}): {ref['text']}",
            "Do not copy the demonstration. This arm is demonstration-conditioned; any adaptation must retain its credit and cannot claim independent novelty.",
        ]
    if previous is not None:
        pieces.append("VERIFIER FEEDBACK: revise the previous proposed surface, fixing the measured constraints while preserving the SAME intended plot. This is one bounded feedback retry, not a constrained decoder.")
        if previous.get("candidate") is not None:
            pieces.append("Previous complete text: " + previous["candidate"])
        if previous.get("audit") is not None:
            audit = previous["audit"]
            feedback = {key: audit[key] for key in (
                "letters", "target_letters", "target_length_delta", "allowed_letter_range",
                "whole_text_exact", "unsupported_alphabetic", "first_mismatches",
                "reference_matches", "duplicate_of_attempt")}
            pieces.append("Measured verifier result (zero-based offsets in normalized ASCII letters): " + json.dumps(feedback, ensure_ascii=False))
        else:
            pieces.append("The previous response was invalid: " + str(previous.get("parse_error", "no_complete_candidate")) + ". Return the required complete JSON object.")
    return "\n\n".join(pieces)


def atomic_checkpoint(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def summarize(rows: list[dict]) -> dict:
    audits = [row["audit"] for row in rows if row.get("audit") is not None]
    return {"responses": len(rows), "parsed_candidates": len(audits),
            "invalid_responses": sum(row.get("parse_error") is not None for row in rows),
            "feedback_retries": sum(row["proposal_stage"] == "verifier_feedback_retry" for row in rows),
            "whole_text_exact": sum(item["whole_text_exact"] for item in audits),
            "eligible_exact_proposals": sum(item["eligible_exact_proposal"] for item in audits),
            "longest_eligible_exact_letters": max((item["letters"] for item in audits if item["eligible_exact_proposal"]), default=0),
            "meaning_measurements": 0}


class BudgetExpired(Exception):
    pass


def configure_public_cache(path: Path) -> None:
    """Configure before importing either Hugging Face package."""
    os.environ.update({"HF_HOME": str(path), "HF_HUB_CACHE": str(path / "hub"),
                       "HF_TOKEN_PATH": str(path / "unused-token"),
                       "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
                       "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1",
                       "HF_ENDPOINT": "https://huggingface.co"})
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        os.environ.pop(name, None)


def choose_cache(output_parent: Path, supplied: Path | None) -> tuple[Path, bool]:
    if supplied is None:
        return Path(tempfile.mkdtemp(prefix="proposal-hf-cache-", dir=output_parent)).resolve(), False
    path = supplied.resolve(strict=True)
    if not path.is_dir():
        raise ValueError("supplied_cache_dir_must_be_an_existing_directory")
    return path, True


def run(args: argparse.Namespace) -> dict:
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing_to_overwrite:{out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    sources, demo, references, hashes = load_public_inputs()
    cases = proposal_cases(sources, demo)
    cache, cache_reused = choose_cache(out.parent, args.cache_dir)
    started = time.monotonic()
    deadline = started + args.seconds
    payload: dict[str, Any] = {
        "experiment": "argonne_model_meaning_proposals_20261009",
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "prepared",
        "method": METHOD_NOTE, "meaning_measurement": "not_measured",
        "human_validation": "not_collected",
        "config": {"model": args.model, "revision": args.revision, "attempts": args.attempts,
                   "max_new_tokens": args.max_new_tokens, "seconds": args.seconds,
                   "temperature": 0.7, "top_p": 0.8, "top_k": 20,
                   "enable_thinking": False, "trust_remote_code": False,
                   "implicit_authentication": False, "explicit_hf_token": False,
                   "cache": str(cache), "cache_reused": cache_reused,
                   "cache_policy": "supplied_existing_task_cache" if cache_reused else "fresh_task_cache",
                   "cuda_only": True,
                   "schedule": "initial + at most one immediate feedback retry; seven cases cycle across pairs",
                   "budget_includes_model_loading": True},
        "public_sources": sources, "credited_demonstration": demo,
        "source_file_sha256": hashes, "reference_tapes": len(references), "cases": cases,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": [], "stats": summarize([]),
    }
    atomic_checkpoint(out, payload)
    configure_public_cache(cache)

    def expire(_signum, _frame):
        raise BudgetExpired("wall_clock_budget_exhausted")

    old_handler = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, deadline - time.monotonic()))
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

        version = re.match(r"(\d+)\.(\d+)", transformers.__version__)
        if version is None or tuple(map(int, version.groups())) < (4, 51):
            raise RuntimeError("transformers_4_51_or_newer_required")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA_required_no_CPU_or_MPS_fallback")
        payload["runtime"] = {"torch": torch.__version__, "transformers": transformers.__version__,
                              "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0)}
        payload["status"] = "loading_model"
        atomic_checkpoint(out, payload)
        common = {"revision": args.revision, "cache_dir": str(cache / "hub"),
                  "token": False, "trust_remote_code": False}
        tokenizer = AutoTokenizer.from_pretrained(args.model, **common)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            args.model, **common, torch_dtype=dtype, use_safetensors=True,
            attn_implementation="sdpa",
        ).to("cuda:0").eval()
        payload["runtime"]["dtype"] = str(dtype)
        resolved_revision = getattr(model.config, "_commit_hash", None)
        payload["runtime"]["resolved_model_revision"] = resolved_revision
        if resolved_revision != args.revision:
            raise RuntimeError("resolved_model_revision_does_not_match_pinned_revision")
        payload["status"] = "running"
        atomic_checkpoint(out, payload)

        class DeadlineCriteria(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return time.monotonic() >= deadline

        seen: dict[str, int] = {}
        initial_row = None
        for attempt in range(args.attempts):
            if time.monotonic() >= deadline:
                raise BudgetExpired("wall_clock_budget_exhausted")
            case = cases[(attempt // 2) % len(cases)]
            needs_retry = bool(attempt % 2 and initial_row is not None and
                               (initial_row.get("audit") is None or
                                not initial_row["audit"]["eligible_exact_proposal"]))
            previous = initial_row if needs_retry else None
            prompt = build_prompt(case, attempt // 2, previous)
            messages = [{"role": "user", "content": prompt}]
            chat = tokenizer.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False)
            inputs = tokenizer([chat], return_tensors="pt", truncation=False).to("cuda:0")
            seed = 2026100900 + attempt
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            call_started = time.monotonic()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens, do_sample=True,
                    temperature=0.7, top_p=0.8, top_k=20,
                    pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria=StoppingCriteriaList([DeadlineCriteria()]),
                    max_time=max(0.001, deadline - time.monotonic()),
                )
            completion = generated[0][inputs.input_ids.shape[-1]:]
            raw = tokenizer.decode(completion, skip_special_tokens=True)
            row = {"attempt": attempt, "case_id": case["id"], "arm": case["arm"],
                   "proposal_stage": "verifier_feedback_retry" if needs_retry else "initial_proposal",
                   "feedback_parent_attempt": previous["attempt"] if previous else None,
                   "intended_plot": case["intent"], "prompt": prompt, "prompt_sha256": sha256(prompt),
                   "seed": seed, "raw_response": raw, "candidate": None, "audit": None,
                   "parse_error": None, "generated_tokens": int(len(completion)),
                   "token_limit_reached": len(completion) >= args.max_new_tokens,
                   "elapsed_seconds": time.monotonic() - call_started,
                   "independent_novelty_claim": False,
                   "lineage": case["reference"] if case["reference"] else {"kind": "model_proposal; originality not established"},
                   "meaning_measurement": "not_measured"}
            payload["rows"].append(row)
            payload["stats"] = summarize(payload["rows"])
            # Save the unmodified response before parsing or validating it.
            atomic_checkpoint(out, payload)
            parsed, row["parse_error"] = parse_response(raw)
            if parsed is not None:
                row["candidate"] = parsed["text"]
                row["model_claims"] = {"plot": parsed["plot"], "status": "unmeasured_model_claim"}
                row["audit"] = audit_candidate(parsed["text"], case, references, seen)
                tape, unsupported = normalized_letters(parsed["text"])
                if tape and not unsupported:
                    seen.setdefault(tape, attempt)
            payload["stats"] = summarize(payload["rows"])
            atomic_checkpoint(out, payload)
            if attempt % 2 == 0:
                initial_row = row
        payload["status"] = "completed"
    except BudgetExpired:
        payload["status"] = "time_limit"
    except (Exception, KeyboardInterrupt) as error:
        payload["status"] = "failed"
        payload["runtime_error"] = {"type": type(error).__name__, "message": str(error)[:1000]}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        payload["stats"] = summarize(payload["rows"])
        payload["elapsed_seconds"] = time.monotonic() - started
        atomic_checkpoint(out, payload)
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--model", default=DEFAULT_MODEL)
    result.add_argument("--revision", default=DEFAULT_REVISION)
    result.add_argument("--attempts", type=int, default=24)
    result.add_argument("--max-new-tokens", type=int, default=512)
    result.add_argument("--seconds", type=float, default=600.0)
    result.add_argument("--cache-dir", type=Path, default=None,
                        help="Existing task-owned HF cache root to reuse; otherwise create a fresh task cache.")
    return result


def main() -> None:
    argument_parser = parser()
    args = argument_parser.parse_args()
    if args.model != DEFAULT_MODEL or args.revision != DEFAULT_REVISION:
        argument_parser.error("this frozen experiment requires the pinned public Qwen3-8B model and revision")
    if args.attempts <= 0 or args.max_new_tokens <= 0 or args.seconds <= 0:
        argument_parser.error("attempts, max-new-tokens and seconds must be positive")
    try:
        payload = run(args)
    except (OSError, ValueError, KeyError, StopIteration) as error:
        argument_parser.exit(1, f"public-input/output preparation failed: {type(error).__name__}: {error}\n")
    print(json.dumps({"status": payload["status"], "stats": payload["stats"]}, sort_keys=True))
    if payload["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
