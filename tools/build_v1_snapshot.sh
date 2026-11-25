#!/usr/bin/env bash
# Build the site as it was live before the redesign (commit 9970db8b) under /v1/,
# into web/dist/v1, so palindrome.ericspencer.us/v1/ keeps showing it.
# Run after `npm run build` in web/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REF="${1:-9970db8b}"
TMP="$(mktemp -d)"
trap 'git -C "$ROOT" worktree remove --force "$TMP/src" >/dev/null 2>&1 || true; rm -rf "$TMP"' EXIT
git -C "$ROOT" worktree add -q --detach "$TMP/src" "$REF"
ln -s "$ROOT/web/node_modules" "$TMP/src/web/node_modules"
(cd "$TMP/src/web" && npx vite build --base /v1/ --outDir "$TMP/out" --emptyOutDir >/dev/null)
rm -rf "$ROOT/web/dist/v1"
cp -R "$TMP/out" "$ROOT/web/dist/v1"
echo "web/dist/v1 <- $REF"
