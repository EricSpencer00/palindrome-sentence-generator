# Offline first-person review packet: v4 palindrome outputs

## What this review is

This is one author's method-label-masked first-pass pilot of 52 unique texts that the v4 search had already selected. It is exploratory feedback, not an independent or confirmatory human evaluation. The sample is conditioned on search success and the system's existing selection rule. It cannot estimate generation coverage, show superiority over an external method, or support population-level readability claims.

The texts were taken from the frozen v4 static-verifier input. Repeated identical outputs were shown once. All 52 retained the source study's exact normalized-letter reversal check: the complete letter sequence reads the same backward after case and punctuation are ignored. None reverses the word order. Rate each text as written; do not give credit for the letter pattern. The page hides method and resource labels. Keep `sealed_key.json`, `shuffle_controls_key.json`, and the full `paper/v4-reproducibility.zip` archive closed until this first pass is complete; they can reveal item provenance.

## How to use the review page

Open `reviewer.html` directly in a current desktop browser. It is one standalone file and needs no server or network connection. Press **Start / Resume** to begin. Your work autosaves in that browser on this device after every answer. **Pause** preserves the current position. Reopen the same file in the same browser and press **Start / Resume** to continue. If local autosave is unavailable, the page warns before closing unsaved work; export a checkpoint and click **I saved the checkpoint** only after confirming the JSON file appears in your downloads. Use **Export answers** regularly to keep a portable checkpoint. **Import checkpoint** restores a previously exported file for this packet.

The export contains your ratings and anonymized item IDs. Keep a copy separate from the sealed key. Ratings remain in this browser unless you explicitly export them. To validate an exported file and print descriptive means, run `python3 paper/human_eval_v4_packet/validate_answers.py /path/to/palindrome-v4-first-person-pilot-answers.json` from the repository root, or `python3 validate_answers.py /path/to/palindrome-v4-first-person-pilot-answers.json` from the extracted review packet folder. The summary does not unblind methods or run significance tests.

## Rate each text on its own

Judge only what the text says as written. Do not try to repair it, infer intended wording, or reward it for satisfying a palindrome constraint. Do not compare it with other items. If you cannot recover a coherent reading, say so in the understandability score and leave the paraphrase blank.

**Grammaticality**

1. Not a well-formed sentence.
2. Major grammatical problems; only a fragment or strained reading is possible.
3. Some problems, but a sentence-level reading is possible.
4. Mostly well formed, with at most a minor problem.
5. Well formed and natural as written.

**Understandability**

1. No coherent meaning is recoverable.
2. Meaning is mostly unclear; a reading requires substantial guessing.
3. A plausible gist is recoverable, but important relations or meaning are uncertain.
4. The meaning is mostly clear.
5. The sentence expresses a clear, coherent meaning.

The optional paraphrase is one short line in your own words. It is a check on the reading you actually took from the sentence, not a rewrite or correction.

## Frozen packet provenance

- To regenerate the packet from the saved source: `python3 paper/human_eval_v4_packet/generate_packet.py`.
- Source: `runs/directional-search-resource-sensitivity-20260925/static-verifier-input-aws.json`.
- That saved verifier input contains 66 selected v4 candidates and 32 deterministic same-word shuffles per candidate.
- This review shows all 52 unique candidate texts once, in one fixed randomized order. Same-word shuffles are not shown in the review page; a deterministic set of up to 12 is archived separately for optional later calibration.
- Selection and display order were fixed without reading method labels: include every unique candidate text, derive opaque item IDs from text hashes, then order by the hash of the preregistered constant seed `20260926` plus each text hash. No labels, ratings, or wall-clock timestamps are used to select or order the items.
- `packet_manifest.json` records the source, generator, and output hashes. Regenerating from the same input and generator should reproduce the same packet files byte for byte.

## Interpretation limit

This is one author's first-person pilot. Do not describe it as three-rater agreement, a blinded independent study, or confirmatory evidence. Use it to inspect the items and refine a later evaluation. Preserve the raw export and the frozen packet if you report this pilot.
