#!/bin/sh
set -eu

# shellcheck source=scripts/tools.sh
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
serving_at_start="$(require_server)"

mkdir -p "$results_dir/runs"
run_dir="$(mktemp -d "$results_dir/runs/eval-$label.XXXXXX")"
output_dir="$run_dir/output"
log_file="$run_dir/eval.log"
export TOKENIZER="$tokenizer" TASKS="$tasks" LIMIT="$limit" NUM_CONCURRENT="$num_concurrent"

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

status_file="$run_dir/status"
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

publish_results eval "$label" "$run_dir" "$results_dir" "$serving_at_start"
echo
echo "Latest successful run: $results_dir/eval-$label.latest"
echo "Saved $output_dir"
echo "Saved $log_file"
