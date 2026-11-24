#!/bin/sh
set -eu

# shellcheck source=scripts/tools.sh
. "$(CDPATH='' cd "$(dirname "$0")" && pwd)/scripts/tools.sh"

max_seconds="${MAX_SECONDS:-30}"
max_concurrency="${MAX_CONCURRENCY:-512}"
prompt_tokens="${PROMPT_TOKENS:-256}"
output_tokens="${OUTPUT_TOKENS:-128}"

usage() {
  echo "Usage: ./benchmark.sh baseline|optimized" >&2
}

if [ "$#" -ne 1 ]; then
  usage
  exit 1
fi

label="$1"
check_label "$label"
case "$max_concurrency" in
  ''|*[!0-9]*) valid_concurrency=false ;;
  *[1-9]*) valid_concurrency=true ;;
  *) valid_concurrency=false ;;
esac
if [ "$valid_concurrency" != true ]; then
  error "MAX_CONCURRENCY must be a positive integer."
  exit 1
fi
ensure_tools
serving_at_start="$(require_server)"

mkdir -p "$results_dir/runs"
run_dir="$(mktemp -d "$results_dir/runs/benchmark-$label.XXXXXX")"
json_file="$run_dir/benchmark.json"
html_file="$run_dir/benchmark.html"
log_file="$run_dir/benchmark.log"
export TOKENIZER="$tokenizer" MAX_SECONDS="$max_seconds" PROMPT_TOKENS="$prompt_tokens" OUTPUT_TOKENS="$output_tokens"
export MAX_CONCURRENCY="$max_concurrency"

echo "Running a GuideLLM sweep against http://127.0.0.1:$port"
echo "The adaptive sweep targets up to 10 strategies, each measured for $max_seconds seconds, with $prompt_tokens prompt tokens and $output_tokens output tokens."
echo "Maximum concurrency for the throughput and constant-rate strategies: $max_concurrency."

# The console output is also saved so the tables can be reviewed later.
status_file="$run_dir/status"
{
  "$tools_venv/bin/guidellm" run \
    --backend "kind=openai_http,target=http://127.0.0.1:$port,model=$served_model" \
    --profile "kind=sweep,max_concurrency=$max_concurrency" \
    --constraint "kind=max_duration,seconds=$max_seconds" \
    --data "kind=synthetic_text,prompt_tokens=$prompt_tokens,output_tokens=$output_tokens" \
    --tokenizer "kind=huggingface_auto,model=$tokenizer" \
    --output "kind=json,path=$json_file" \
    --output "kind=html,path=$html_file" \
    --disable-console-interactive 2>&1 &&
    echo 0 > "$status_file" || echo 1 > "$status_file"
} | tee "$log_file"

status="$(cat "$status_file" 2>/dev/null || echo 1)"
rm -f "$status_file"
if [ "$status" != "0" ]; then
  error "GuideLLM did not finish. The full output is in $log_file"
  exit 1
fi

publish_results benchmark "$label" "$run_dir" "$results_dir" "$serving_at_start"
echo
echo "Latest successful run: $results_dir/benchmark-$label.latest"
echo "Saved $json_file"
echo "Saved $html_file"
echo "Saved $log_file"
