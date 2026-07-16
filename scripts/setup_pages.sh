#!/usr/bin/env bash
# Compatibility wrapper that initializes and seeds the gh-pages branch via
# `coinfosim publish pages --init-branch`.
#
# Python (coinfosim.publish.publisher) is the source of truth; this script
# only forwards environment-variable configuration to the CLI command.
set -euo pipefail
REMOTE="${REMOTE:-origin}"
PAGES="${PAGES:-gh-pages}"
OUTPUT_DIR="${OUTPUT_DIR:-output}"

echo "scripts/setup_pages.sh delegates to 'coinfosim publish pages --init-branch'." >&2
exec "${PYTHON:-python3}" -m coinfosim publish pages \
  --output-dir "$OUTPUT_DIR" \
  --branch "$PAGES" \
  --remote "$REMOTE" \
  --init-branch \
  --push
