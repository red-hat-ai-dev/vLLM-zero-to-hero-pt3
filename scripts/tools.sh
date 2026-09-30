# Shared helpers for benchmark.sh, evaluate.sh, and compare.sh.
# Installs GuideLLM and lm-eval into .venv the first time they are needed.

project_dir="$(CDPATH='' cd "$(dirname "$0")" && pwd)"
tools_venv="${TOOLS_VENV:-$project_dir/.venv}"
results_dir="${RESULTS_DIR:-$project_dir/results}"
port="${PORT:-8000}"
served_model="qwen3-4b"
tokenizer="${TOKENIZER:-Qwen/Qwen3-4B}"

error() {
  echo "Error: $*" >&2
}

find_python() {
  for candidate in ${PYTHON:-} python3.13 python3.12 python3.11 python3.10 python3; do
    [ -n "$candidate" ] || continue
    command -v "$candidate" >/dev/null 2>&1 || continue
    if "$candidate" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)' 2>/dev/null; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

ensure_tools() {
  if [ -x "$tools_venv/bin/guidellm" ] && [ -x "$tools_venv/bin/lm_eval" ]; then
    return
  fi

  if ! python="$(find_python)"; then
    error "GuideLLM and lm-eval need Python 3.10 through 3.13."
    echo "Install a supported Python, or point PYTHON at one, then try again." >&2
    exit 1
  fi

  echo "GuideLLM and lm-eval are not installed yet."
  echo "Installing them in $tools_venv with $python."
  echo "This is a one-time setup and may take several minutes."

  if ! "$python" -m venv "$tools_venv"; then
    error "Could not create a Python virtual environment at $tools_venv."
    exit 1
  fi

  "$tools_venv/bin/python" -m pip install --quiet --upgrade pip

  # The tools only send requests and count tokens, so a CPU-only PyTorch is enough.
  if [ "$(uname -s)" = "Linux" ]; then
    if ! "$tools_venv/bin/python" -m pip install --quiet torch \
      --index-url https://download.pytorch.org/whl/cpu; then
      error "Could not install PyTorch."
      rm -rf "$tools_venv"
      exit 1
    fi
  fi

  if ! "$tools_venv/bin/python" -m pip install --quiet -r "$project_dir/requirements.txt"; then
    error "Could not install GuideLLM and lm-eval."
    echo "Review the pip output above, then try again." >&2
    rm -rf "$tools_venv"
    exit 1
  fi
  echo "GuideLLM and lm-eval are installed."
}

require_server() {
  if ! curl --fail --silent "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
    error "No vLLM server is answering at http://127.0.0.1:$port/v1"
    echo "Start one first, for example: ./run.sh baseline" >&2
    exit 1
  fi
}

check_label() {
  case "$1" in
    ""|*[!A-Za-z0-9_-]*)
      error "Use a simple label such as baseline or optimized."
      exit 1
      ;;
  esac
}
