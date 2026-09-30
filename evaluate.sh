#!/bin/sh
set -eu

. "$(CDPATH='' cd "$(dirname "$0")" && pwd)/scripts/tools.sh"

tasks="${TASKS:-gsm8k}"
limit="${LIMIT:-}"
num_concurrent="${NUM_CONCURRENT:-8}"

usage() {
  echo "Usage: ./evaluate.sh baseline|optimized" >&2
}

if [ "$#" -ne 1 ]; then
  usage
  exit 1
fi

label="$1"
check_label "$label"
ensure_tools
require_server

output_dir="$results_dir/eval-$label"
log_file="$results_dir/eval-$label.log"
mkdir -p "$results_dir"
if [ -d "$output_dir" ]; then
  rm -rf "$output_dir"
fi

set -- \
  --model local-completions \
  --model_args "model=$served_model,base_url=http://127.0.0.1:$port/v1/completions,num_concurrent=$num_concurrent,max_retries=3,tokenized_requests=False,tokenizer=$tokenizer" \
  --tasks "$tasks" \
  --output_path "$output_dir" \
  --log_samples

if [ -n "$limit" ]; then
  set -- "$@" --limit "$limit"
  echo "Evaluating $tasks on the first $limit examples."
else
  echo "Evaluating $tasks on the full test set."
fi

status_file="$results_dir/.eval-$label.status"
{
  "$tools_venv/bin/lm_eval" "$@" 2>&1 &&
    echo 0 > "$status_file" || echo 1 > "$status_file"
} | tee "$log_file"

status="$(cat "$status_file" 2>/dev/null || echo 1)"
rm -f "$status_file"
if [ "$status" != "0" ]; then
  error "lm-eval did not finish. The full output is in $log_file"
  exit 1
fi

echo
echo "Saved $output_dir"
echo "Saved $log_file"
