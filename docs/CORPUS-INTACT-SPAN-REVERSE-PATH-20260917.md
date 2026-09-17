# Corpus intact-span reverse path (2026-09-17)

This experiment introduces a corpus-scale representation not present in the
novelty registry: each attested sentence is an immutable vertex, and a
directed edge records how many characters of a candidate span match the
reversed-character residual of another span. Search grows distinct-span paths
from both ends, retaining source sentence IDs for provenance rather than
rewriting or mirroring words.

The run used 160 authored sentences, 2,029 residual edges, and paths of at
most three sentences per side. It found no exact closure. The longest frontier
matched 14 characters before exhausting the bounded path search. Every
potential result is independently checked by a two-pointer character audit
and SHA-256 of the normalized tape; therefore no rendered candidate is
reported without an exact audit.

The next repair is a provenance-preserving bridge-span index keyed by residual
prefix/suffix, followed by a four-span path run. The corpus remains immutable;
this is not a duplicate CFG, language-model, scene, morphology, or semantic
residual sweep.

Run artifact: `runs/corpus-intact-span-reverse-path-20260917.json`.
