#!/bin/sh
set -eu

script_dir="$(CDPATH='' cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/engine.sh
. "$script_dir/scripts/engine.sh"

tools_venv="$script_dir/.venv"
temp_root="${TMPDIR:-/tmp}"
state_dir="${temp_root%/}/vllm-zero-to-hero-pt3"
user_home="${HOME:-}"
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

describe_path() {
  label="$1"
  path="$2"
  if [ -e "$path" ] || [ -L "$path" ]; then
    echo "  - $label: $path"
  else
    echo "  - $label: already removed"
  fi
}

owns_tools_venv() {
  [ ! -L "$tools_venv" ] && [ ! -L "$tools_venv/.vllm-zero-to-hero-pt3" ] &&
    [ -f "$tools_venv/.vllm-zero-to-hero-pt3" ]
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
    describe_path "temporary logs and state" "$state_dir"
    echo "  The shared Hugging Face cache is kept."
    ;;
  Linux)
    resolve_managed_engine "$name" "$managed_label" "$volume"
    echo "  - container: $name"
    echo "  - downloaded model volume: $volume"
    ;;
  *)
    error "This operating system is not supported: $os"
    exit 1
    ;;
esac
remove_tools=false
if owns_tools_venv; then
  remove_tools=true
  describe_path "GuideLLM and lm-eval environment" "$tools_venv"
elif [ -e "$tools_venv" ] || [ -L "$tools_venv" ]; then
  echo "  Existing unmarked or linked tools environment will be kept: $tools_venv"
fi

echo
echo "Your results/ folder, the shared vLLM Metal environment, and container images are kept."
confirm_cleanup

if [ "$os" = Darwin ]; then
  "$script_dir/scripts/metal.sh" cleanup "$state_dir"
else
  stop_lesson_container
  if [ -n "$engine" ]; then remove_model_volume; fi
fi
if [ "$remove_tools" = true ] && owns_tools_venv; then
  echo "Removing the lesson's GuideLLM and lm-eval environment..."
  rm -rf "$tools_venv"
fi

echo "Cleanup complete. Run ./run.sh baseline whenever you want to start again."
