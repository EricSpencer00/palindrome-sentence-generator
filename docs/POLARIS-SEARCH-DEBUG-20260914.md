# Polaris exact-search debug record — 2026-09-14

Job `7618101.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov` ran in the Polaris
`debug` queue under project `EVITA` and exited successfully (`Exit_status=0`).
The retained remote aggregate is:

`/home/eric-spencer/palindrome/runs/search_debug_20260914_142556/aggregate.json`

Configuration: 32 ranks, 30,000-word frozen vocabulary, candidate limit 200,
beam 48, eight children per parent, 80-letter floor, 240 search steps, and
eight seeds per rank. The job ran for 25 seconds of wall time, used 64 CPUs,
and no GPUs.

Results: 256/256 searches closed; every closure passed the independent exact
palindrome assertion; outputs were 80–81 letters; the candidate menu contained
112 words of at least five letters (mean 5.305 letters). The no-charge login
smoke test also passed and closed 1/1 search.

This is evidence for the exact-search path and its candidate-menu regression
only. It is not evidence of readable long output, reader acceptance, novelty,
or paper eligibility. No result was promoted to product output or manuscript
material; generation routes remain fail-closed with HTTP 503.
