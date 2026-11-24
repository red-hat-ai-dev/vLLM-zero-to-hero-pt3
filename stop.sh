#!/bin/sh
set -eu

script_dir="$(CDPATH='' cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/engine.sh
. "$script_dir/scripts/engine.sh"

state_dir="${TMPDIR:-/tmp}/vllm-zero-to-hero-pt3"


if [ "$(uname -s)" = Darwin ]; then
  exec "$script_dir/scripts/metal.sh" stop "$state_dir"
fi

resolve_managed_engine "$name" "$managed_label"
stop_lesson_container
