#!/usr/bin/env bash
# Build an anonymised copy of this repository for double-blind review.
#
# Copies every git-tracked file to $DEST, rewrites the identifying strings,
# replaces CITATION.cff, and then greps the result. A surviving identifier is
# a non-zero exit, so this is safe to run in CI before a submission.
#
#   ./tools/make_anon_mirror.sh [dest]
#
# Default dest is ../palindrome-anon. The mirror carries no .git directory:
# commit messages and author records deanonymise as thoroughly as the files do.

set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$(dirname "$SRC")/palindrome-anon}"

rm -rf "$DEST"
mkdir -p "$DEST"

cd "$SRC"
git ls-files -z | while IFS= read -r -d '' f; do
  mkdir -p "$DEST/$(dirname "$f")"
  cp "$f" "$DEST/$f"
done

cd "$DEST"

# Text substitutions. Order matters: longest match first.
find . -type f \( -name '*.md' -o -name '*.py' -o -name '*.js' -o -name '*.ts' \
     -o -name '*.tsx' -o -name '*.json' -o -name '*.cff' -o -name '*.txt' \
     -o -name '*.html' -o -name '*.yml' -o -name '*.yaml' -o -name '*.ini' \) \
     -print0 | xargs -0 perl -pi -e '
  s{https://github\.com/EricSpencer00/palindrome-sentence-generator}{https://anonymous.4open.science/r/palindrome-anon}g;
  s{https://palindrome-api\.ericspencer\.us}{https://api.example.invalid}g;
  s{palindrome\.ericspencer\.us}{palindrome.example.invalid}g;
  s{https://ericspencer\.us}{https://example.invalid}g;
  s{ericspencer\.us}{example.invalid}g;
  s{EricSpencer00}{anonymous}g;
  s{Eric Spencer}{Anonymous Author}g;
  s{\bSpencer\b}{Anonymous}g;
  s{Loyola University Chicago}{Anonymous Institution}g;
  s{Developed within the \*\*AI4FM group\*\*\.}{Group affiliation withheld for review.}g;
  s{\bAI4FM\b}{Anonymous Group}g;
'

# CITATION.cff cannot be patched into anonymity; replace it wholesale.
cat > CITATION.cff <<'CFF'
cff-version: 1.2.0
message: "Anonymised copy for double-blind review. Authorship withheld."
title: "Palindrome Sentence Generator (anonymised)"
abstract: "Generates long, multi-sentence character-level palindromes using an
  LLM-scored overhang search over a bigram-coherence model. Author, affiliation
  and repository URL are withheld for review."
type: software
authors:
  - name: "Anonymous Author"
license: MIT
version: "1.0.0"
CFF

cat > ANONYMITY.md <<'ANON'
# Anonymised artifact

Built by `tools/make_anon_mirror.sh` from the working repository. What changed:

| removed | replaced with |
|---|---|
| author name and affiliation in `CITATION.cff` | a stub naming no author |
| the project website and API host | `example.invalid` |
| the public repository URL | an anonymous.4open.science placeholder |
| the group credit in `README.md` | a withheld-for-review line |
| the `.git` directory | omitted: commit authorship and messages deanonymise |

Nothing else was altered. Code, data, experiments, documentation and test suite
are byte-identical to the working copy apart from the substitutions above.

Two things a reviewer should know. The `data/` directory contains catalogued
palindromes collected from published sources; the provenance split between
catalogued and generated material is recorded in `README.md` and enforced by
`llm_palindrome/safe_vocab.py` and the novelty check. And the deployed service
referenced in `web/` and `server/` is not reachable from this copy, by design.
ANON

# Verification.
echo "verifying..."
PAT='ericspencer|Eric Spencer|EricSpencer00|AI4FM|Loyola'
if grep -rIlnE "$PAT" . 2>/dev/null | grep -v '^./ANONYMITY.md$'; then
  echo "FAIL: identifying strings survive in the files listed above" >&2
  exit 1
fi
if [ -d .git ]; then echo "FAIL: .git present" >&2; exit 1; fi

echo "OK  $DEST"
echo "    $(find . -type f | wc -l | tr -d ' ') files, $(du -sh . | cut -f1) total"
