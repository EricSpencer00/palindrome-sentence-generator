# Corrected beam debug check

Polaris debug job `7591781`, 4 September 2026, ran the corrected structural
beam across 32 ranks with the frozen 30k vocabulary. This is a closure and
candidate-menu verification, not a quality claim.

- 256 / 256 runs closed as valid 80--95-letter palindromes.
- The limited root menu contained 112 words of five or more letters (mean
  length 5.305; maximum 13). The former breadth-first menu contained none.
- Per-rank opening diversity summed to 116 (five unique opening words across
  the entire run), so closure is fixed but proposal diversity remains a
  separate score-calibration problem.
- Wall time was 26 seconds on one debug node; artifacts are in
  `runs/polaris/search_debug_20260904_152738/`.

The workload used a dependency-free rank scorer plus the repayable-overhang
term, specifically to validate the corrected search substrate on Polaris. It
does not evaluate prose quality or validate the directional model.
