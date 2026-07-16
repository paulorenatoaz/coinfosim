#!/usr/bin/env bash
# Compatibility wrapper around `coinfosim publish pages`.
#
# Python (coinfosim.publish.publisher) is the source of truth; this script
# only forwards environment-variable configuration to the CLI command.
set -euo pipefail
REMOTE="${REMOTE:-origin}"
PAGES="${PAGES:-gh-pages}"
OUTPUT_DIR="${OUTPUT_DIR:-output}"

echo "scripts/publish_output_to_pages.sh delegates to 'coinfosim publish pages'." >&2
exec "${PYTHON:-python3}" -m coinfosim publish pages \
  --output-dir "$OUTPUT_DIR" \
  --branch "$PAGES" \
  --remote "$REMOTE" \
  --push
