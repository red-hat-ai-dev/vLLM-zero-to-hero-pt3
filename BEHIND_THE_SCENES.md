# Behind the scenes

Part 3 has two halves. `run.sh` starts a vLLM server the same way as Part 1.
`benchmark.sh`, `evaluate.sh`, and `compare.sh` measure that server from the
outside through its OpenAI-compatible API, so they work the same on every
platform.

## What `run.sh` does

`run.sh` takes the model to serve, `baseline` or `optimized`, and optionally an
accelerator override:

```bash
./run.sh baseline
./run.sh optimized nvidia
```

Both models are served with the same settings. Only the checkpoint changes:

```bash
vllm serve MODEL \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name qwen3-4b \
  --max-model-len 8192
```

Serving both models as `qwen3-4b` means the benchmark and evaluation commands
never change between runs.

| Platform | `baseline` | `optimized` |
| --- | --- | --- |
| Linux | `Qwen/Qwen3-4B` | `RedHatAI/Qwen3-4B-FP8-dynamic` |
| Apple Silicon | `mlx-community/Qwen3-4B-bf16` | `mlx-community/Qwen3-4B-4bit` |

Each model can be replaced with an environment variable:

```bash
BASELINE_MODEL=Qwen/Qwen3-4B ./run.sh baseline
OPTIMIZED_MODEL=RedHatAI/Qwen3-4B-FP8-dynamic ./run.sh optimized
METAL_BASELINE_MODEL=mlx-community/Qwen3-4B-bf16 ./run.sh baseline
METAL_OPTIMIZED_MODEL=mlx-community/Qwen3-4B-4bit ./run.sh optimized
```

When `OPTIMIZED_MODEL` or `BASELINE_MODEL` points to a local folder, the
launcher mounts it read-only into the container at `/models/optimized` or
`/models/baseline` and serves it from there. This is how a checkpoint created
in Part 2 is benchmarked.

### Linux containers

The Linux path uses the same `cuda`, `rocm`, and `xpu` images as Part 1:

```text
ghcr.io/red-hat-ai-dev/vllm-zero-to-hero:cuda
ghcr.io/red-hat-ai-dev/vllm-zero-to-hero:rocm
ghcr.io/red-hat-ai-dev/vllm-zero-to-hero:xpu
```

The images do not contain model weights. The launcher passes the model and its
options to the image, which replaces the Part 1 defaults. Models are stored in
the same `vllm-models` volume as Part 1, so a model that is already downloaded
is reused. Set `IMAGE_REPO` to use images built elsewhere, and `HF_TOKEN` to
pass a Hugging Face token into the container.

Accelerator detection, `ENGINE`, NVIDIA Container Toolkit checks, and
automatic cleanup of failed starts all work as described in
[Part 1](https://github.com/red-hat-ai-dev/vLLM-zero-to-hero-pt1/blob/main/BEHIND_THE_SCENES.md).

### Apple Silicon

The Mac path uses the vLLM Metal environment from Part 1 at
`~/.venv-vllm-metal`, and installs it if it is missing. The FP8 checkpoint is
built for GPU kernels that vLLM Metal does not provide, so the Mac comparison
uses a full-precision MLX checkpoint and a four-bit MLX checkpoint instead.

The process ID and log are stored in the same place as Part 1:

```text
$TMPDIR/vllm-zero-to-hero/metal.pid
$TMPDIR/vllm-zero-to-hero/metal.log
```

### One server at a time

Both models listen on port 8000. `run.sh` refuses to start while anything else
answers on that port, so a benchmark always measures the model you just
started. `./stop.sh` stops either model.

## The benchmarking tools

The first `./benchmark.sh`, `./evaluate.sh`, or `./compare.sh` creates a Python
virtual environment in `.venv` and installs the versions pinned in
`requirements.txt`:

```text
guidellm[recommended]==0.7.4
lm-eval[api]==0.4.13
```

GuideLLM depends on PyTorch for tokenization. On Linux, the script installs the
CPU-only PyTorch build first, because the tools only send requests and count
tokens. This keeps the download much smaller than the default GPU build.

The script looks for `python3.13` through `python3.10`, then `python3`. Set
`PYTHON` to choose one explicitly, or `TOOLS_VENV` to use another environment
that already has both tools installed.

All results are written to `results/`. Set `RESULTS_DIR` to write them
somewhere else.

```text
results/
  benchmark-baseline.json
  benchmark-baseline.html
  benchmark-baseline.log
  eval-baseline/qwen3-4b/results_<timestamp>.json
  eval-baseline/qwen3-4b/samples_gsm8k_<timestamp>.jsonl
  eval-baseline.log
```

## What `benchmark.sh` does

`benchmark.sh` checks that a server answers at `/v1/models`, then runs one
GuideLLM sweep:

```bash
guidellm run \
  --backend kind=openai_http,target=http://127.0.0.1:8000,model=qwen3-4b \
  --profile kind=sweep \
  --constraint kind=max_duration,seconds=30 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128 \
  --tokenizer kind=huggingface_auto,model=Qwen/Qwen3-4B \
  --output kind=json,path=results/benchmark-baseline.json \
  --output kind=html,path=results/benchmark-baseline.html \
  --disable-console-interactive
```

- `--backend` sends requests to the server's `/v1/chat/completions` endpoint
  with streaming turned on, which is how GuideLLM records the time of every
  token.
- `--tokenizer` is needed because the served name, `qwen3-4b`, is not a Hugging
  Face model ID. GuideLLM uses the tokenizer to build synthetic prompts of an
  exact length. Set `TOKENIZER` to change it.
- `--disable-console-interactive` replaces the live progress display with
  plain log lines, so the saved log file stays readable.

The HTML report is built from a page template that GuideLLM downloads when the
benchmark finishes, so the report needs internet access. The JSON file does
not.

## What `evaluate.sh` does

`evaluate.sh` runs lm-eval against the same server:

```bash
lm_eval \
  --model local-completions \
  --model_args model=qwen3-4b,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=8,max_retries=3,tokenized_requests=False,tokenizer=Qwen/Qwen3-4B \
  --tasks gsm8k \
  --output_path results/eval-baseline \
  --log_samples
```

- `local-completions` uses the `/v1/completions` endpoint. GSM8K is written as
  plain text, with five solved examples followed by the new question, so it
  does not need the chat template.
- `tokenized_requests=False` sends the prompts as text and lets vLLM tokenize
  them. lm-eval still loads a tokenizer, and `tokenizer` tells it which one,
  because the served name is not a Hugging Face model ID. Set `TOKENIZER` to
  change it.
- `num_concurrent=8` sends eight problems at a time so the run finishes
  sooner. GSM8K uses greedy decoding (`temperature=0`).
- Each run replaces the previous results for the same label.

## What `compare.sh` does

`compare.sh` runs `scripts/compare.py` on two labels, `baseline` and
`optimized` by default. Any other labels work too:

```bash
./compare.sh baseline my-experiment
```

From each GuideLLM JSON file it reads:

- The median TTFT, ITL, and request latency of the `synchronous` benchmark.
- The requests per second, output tokens per second, and p95 TTFT of the
  `throughput` benchmark.
- The highest `constant` rate where the completed requests per second reached
  at least 90 percent of the target rate.

From each lm-eval results file it reads every `exact_match` score and shows the
difference in percentage points.

## Complete local cleanup

`./cleanup.sh` lists what it will remove and asks for confirmation. It stops the
server, then removes:

- On Linux, the `vllm-zero-to-hero` container and the `vllm-models` volume. This
  volume is shared with Part 1.
- On Apple Silicon, the two MLX models from the Hugging Face cache. The vLLM
  Metal environment is kept, because Part 1 manages it.
- The `.venv` tools environment and temporary state.

The `results/` folder is never removed.
