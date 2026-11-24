#!/bin/sh
set -eu

# An unused lesson has nothing to stop or clean up, even without Python.
case "$1" in
  stop|cleanup)
    if [ ! -e "$2" ] && [ ! -L "$2" ]; then
      echo "vLLM Metal is already stopped."
      exit 0
    fi
    ;;
esac
python="${VLLM_METAL_VENV:-$HOME/.venv-vllm-metal}/bin/python"
if [ ! -x "$python" ]; then python=python3; fi
exec "$python" "$(dirname "$0")/metal.py" "$@"
