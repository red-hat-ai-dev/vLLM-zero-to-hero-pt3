#!/bin/sh
set -eu

# shellcheck source=scripts/tools.sh
. "$(CDPATH='' cd "$(dirname "$0")" && pwd)/scripts/tools.sh"

if [ "$#" -ne 0 ] && [ "$#" -ne 2 ]; then
  echo "Usage: ./compare.sh [first-label second-label]" >&2
  exit 1
fi

first="${1:-baseline}"
second="${2:-optimized}"
check_label "$first"
check_label "$second"
ensure_tools

"$tools_venv/bin/python" "$project_dir/scripts/compare.py" "$results_dir" "$first" "$second"
