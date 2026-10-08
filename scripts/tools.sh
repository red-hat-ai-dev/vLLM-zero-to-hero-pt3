# shellcheck shell=sh
# shellcheck disable=SC2034 # Variables are consumed by the sourcing scripts.
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
  if [ -n "${PYTHON:-}" ]; then
    set -- "$PYTHON"
  else
    set -- python3.13 python3.12 python3.11 python3.10 python3
  fi
  for candidate in "$@"; do
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
  if [ -x "$tools_venv/bin/guidellm" ] && [ -x "$tools_venv/bin/lm_eval" ] &&
    "$tools_venv/bin/python" -c 'import guidellm, lm_eval; import importlib.metadata as m; assert m.version("guidellm") == "0.7.4"; assert m.version("lm-eval") == "0.4.13"' >/dev/null 2>&1; then
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
      exit 1
    fi
  fi

  if ! "$tools_venv/bin/python" -m pip install --quiet -r "$project_dir/scripts/requirements.txt"; then
    error "Could not install GuideLLM and lm-eval."
    echo "Review the pip output above, then try again." >&2
    exit 1
  fi
  echo "GuideLLM and lm-eval are installed."
}

require_server() {
  "$tools_venv/bin/python" - "$port" "$served_model" <<'PYCODE'
import json
import sys
import urllib.request
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/v1/models", timeout=5) as response:
        models = json.load(response)
    if sys.argv[2] not in {model["id"] for model in models["data"]}:
        raise ValueError(f"Expected model alias {sys.argv[2]} is not being served")
except Exception as exc:
    raise SystemExit(f"Server check failed: {exc}. Start ./run.sh first.")
PYCODE
}

check_label() {
  case "$1" in
    ""|*[!A-Za-z0-9_-]*)
      error "Use a simple label such as baseline or optimized."
      exit 1
      ;;
  esac
}
