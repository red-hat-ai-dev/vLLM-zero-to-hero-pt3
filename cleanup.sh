#!/bin/sh
set -eu

script_dir="$(CDPATH='' cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/engine.sh
. "$script_dir/scripts/engine.sh"
managed_label="io.github.red-hat-ai-dev.vllm-zero-to-hero.managed"

name="vllm-zero-to-hero"
volume="vllm-models"
script_dir="$(CDPATH='' cd "$(dirname "$0")" && pwd)"
tools_venv="$script_dir/.venv"
temp_root="${TMPDIR:-/tmp}"
state_dir="${temp_root%/}/vllm-zero-to-hero"
user_home="${HOME:-}"
hf_home="${HF_HOME:-$user_home/.cache/huggingface}"
metal_models="models--mlx-community--Qwen3-4B-bf16 models--mlx-community--Qwen3-4B-4bit"
assume_yes="false"
engine=""

usage() {
  echo "Usage: ./cleanup.sh [--yes]"
  echo
  echo "Stop vLLM and remove the files downloaded by this part."
  echo "  --yes  Skip the confirmation prompt."
}

error() {
  echo "Error: $*" >&2
}

if [ -z "$user_home" ] || [ "$user_home" = "/" ]; then
  error "HOME does not point to a safe user directory. Nothing was removed."
  exit 1
fi

if [ -z "$hf_home" ] || [ "$hf_home" = "/" ]; then
  error "HF_HOME does not point to a safe cache directory. Nothing was removed."
  exit 1
fi

describe_path() {
  label="$1"
  path="$2"
  if [ -e "$path" ] || [ -L "$path" ]; then
    echo "  - $label: $path"
  else
    echo "  - $label: already removed"
  fi
}

remove_path() {
  label="$1"
  path="$2"
  case "$path" in
    "$tools_venv"|"$state_dir"|"$hf_home"/hub/models--mlx-community--Qwen3-4B-*|"$hf_home"/hub/.locks/models--mlx-community--Qwen3-4B-*) ;;
    *)
      error "Refusing to remove an unexpected path: $path"
      return 1
      ;;
  esac

  if [ ! -e "$path" ] && [ ! -L "$path" ]; then
    echo "$label is already removed."
    return
  fi

  echo "Removing $label..."
  rm -rf "$path"
  echo "Removed $label."
}

confirm_cleanup() {
  if [ "$assume_yes" = "true" ]; then
    return
  fi

  printf "Continue with cleanup? [y/N] "
  answer=""
  read -r answer || true
  case "$answer" in
    y|Y|[yY][eE][sS]) ;;
    *)
      echo "Cleanup cancelled. Nothing was removed."
      exit 0
      ;;
  esac
}

select_engine() {
  resolve_managed_engine "$name" "$managed_label" "${volume:-}"
}

case "${1:-}" in
  "") ;;
  --yes) assume_yes="true" ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    error "Unknown option: $1"
    usage >&2
    exit 1
    ;;
esac

os="$(uname -s)"
echo "This will stop vLLM and permanently remove:"
case "$os" in
  Darwin)
    for model in $metal_models; do
      describe_path "example model" "$hf_home/hub/$model"
    done
    ;;
  Linux)
    select_engine
    echo "  - container: $name"
    echo "  - downloaded model volume: $volume"
    ;;
  *)
    error "This operating system is not supported: $os"
    exit 1
    ;;
esac
describe_path "GuideLLM and lm-eval environment" "$tools_venv"
describe_path "temporary logs and state" "$state_dir"

echo
echo "Your results/ folder, the vLLM Metal environment, and container images will not be removed."
echo
echo "The shared vLLM Metal environment is kept."
confirm_cleanup

if [ -n "$engine" ]; then
  ENGINE="$engine" "$script_dir/stop.sh"
else
  "$script_dir/stop.sh"
fi

if [ "$os" = "Darwin" ]; then
  for model in $metal_models; do
    remove_path "example model" "$hf_home/hub/$model"
    remove_path "example model locks" "$hf_home/hub/.locks/$model"
  done
elif [ -n "$engine" ]; then
  if "$engine" volume inspect "$volume" >/dev/null 2>&1; then
    echo "Removing container model volume '$volume' with $engine..."
    "$engine" volume rm "$volume" >/dev/null
    echo "Removed container model volume."
  else
    echo "Container model volume is already removed."
  fi
fi
remove_path "GuideLLM and lm-eval environment" "$tools_venv"
remove_path "temporary state" "$state_dir"

echo "Cleanup complete. Run ./run.sh baseline whenever you want to start again."
