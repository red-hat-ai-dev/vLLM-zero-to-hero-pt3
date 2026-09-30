#!/bin/sh
set -eu

. "$(CDPATH='' cd "$(dirname "$0")" && pwd)/scripts/tools.sh"

max_seconds="${MAX_SECONDS:-30}"
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
ensure_tools
require_server

mkdir -p "$results_dir"
json_file="$results_dir/benchmark-$label.json"
html_file="$results_dir/benchmark-$label.html"
log_file="$results_dir/benchmark-$label.log"

echo "Running a GuideLLM sweep against http://127.0.0.1:$port"
echo "Each of the 10 strategies runs for $max_seconds seconds with $prompt_tokens prompt tokens and $output_tokens output tokens."

# The console output is also saved so the tables can be reviewed later.
status_file="$results_dir/.benchmark-$label.status"
{
  "$tools_venv/bin/guidellm" run \
    --backend "kind=openai_http,target=http://127.0.0.1:$port,model=$served_model" \
    --profile kind=sweep \
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

echo
echo "Saved $json_file"
echo "Saved $html_file"
echo "Saved $log_file"
