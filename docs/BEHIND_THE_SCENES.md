# Behind the scenes

Part 3 has two halves. `run.sh` starts a vLLM server the same way as Part 1.
`benchmark.sh`, `evaluate.sh`, and `compare.sh` measure that server from the
outside through its OpenAI-compatible API, so they work the same on every
platform.

The [README](../README.md) is the walkthrough, including how to read the
results. This guide explains how the scripts fit together.

## What is in this repository?

| File or folder | What it does |
| --- | --- |
| [`run.sh`](../run.sh), [`stop.sh`](../stop.sh), [`cleanup.sh`](../cleanup.sh) | Manage the model server and local setup. |
| [`benchmark.sh`](../benchmark.sh) | Measures latency and throughput with GuideLLM. |
| [`evaluate.sh`](../evaluate.sh) | Measures answer accuracy with lm-eval. |
| [`compare.sh`](../compare.sh) | Compares two sets of saved results. |
| [`scripts/`](../scripts) | Holds shared shell helpers, result handling, comparison code, and tool requirements. |
| [`docs/assets/`](assets) | Holds the diagrams used in the walkthrough. |

The tools create `.venv/` and `results/` locally when you use them. You do not
need to edit the supporting files to follow the lesson.

## What `run.sh` does

`run.sh` takes the model to serve, `baseline` or `optimized`, and optionally an
accelerator override:

```bash
./run.sh baseline
./run.sh optimized nvidia
```

On GPU, both models are served with the same settings. Only the checkpoint changes:

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
| CUDA | `Qwen/Qwen3-4B` | `RedHatAI/Qwen3-4B-FP8-dynamic` |
| ROCm (hardware validation pending) | `Qwen/Qwen3-4B` | `RedHatAI/Qwen3-4B-FP8-dynamic` |
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

The Linux path uses the same `cuda` and `rocm` images as Part 1:

```text
ghcr.io/red-hat-ai-dev/vllm-zero-to-hero:cuda
ghcr.io/red-hat-ai-dev/vllm-zero-to-hero:rocm
```

The images do not contain model weights. The launcher passes the model and its
options to the image, which replaces the Part 1 defaults. Part 3 keeps its
models in the labeled `vllm-zero-to-hero-pt3-models` volume. The first run
downloads the models even if an earlier release used the shared `vllm-models`
volume. Set `IMAGE_REPO` to use images built elsewhere, and `HF_TOKEN` to
pass a Hugging Face token into the container.

Accelerator detection, `ENGINE`, NVIDIA Container Toolkit checks, and
automatic cleanup of failed starts all work as described in
[Part 1](https://github.com/red-hat-ai-dev/vLLM-zero-to-hero-pt1).

[`scripts/engine.sh`](../scripts/engine.sh) is the helper used by `stop.sh` and
`cleanup.sh` to locate the engine that owns the managed container or volume.
It verifies resource ownership labels and asks for an explicit `ENGINE`
if separate project resources exist in both Podman and Docker.

### CPU

`./run.sh baseline cpu` uses `vllm/vllm-openai-cpu:v0.28.0-x86_64` in an
x86-64 Linux runtime, without GPU devices. It keeps the same model alias and
cache volume, with BF16, eager execution, a 4,096-token context, a 1 GiB KV cache,
and at most four concurrent requests. One CPU core is reserved for the API.
Only baseline serving is enabled; benchmarking and evaluation use the same API.

### Apple Silicon

The Mac path uses the vLLM Metal environment from Part 1 at
`~/.venv-vllm-metal`, and installs it if it is missing. The FP8 checkpoint is
built for GPU kernels that vLLM Metal does not provide, so the Mac comparison
uses a full-precision MLX checkpoint and a four-bit MLX checkpoint instead.

Part 3 records the server's process identity and log in
`${TMPDIR:-/tmp}/vllm-zero-to-hero-pt3/`. Stop checks the recorded identity
before signaling a process. If you upgrade while an older server is running,
stop it from the checkout that launched it.

### One server at a time

Both models listen on port 8000. `run.sh` refuses to start while anything else
answers on that port, so a benchmark always measures the model you just
started. `./stop.sh` stops either model.

## The benchmarking tools

The first `./benchmark.sh`, `./evaluate.sh`, or `./compare.sh` creates a Python
virtual environment in `.venv` and installs the versions pinned in
[`scripts/requirements.txt`](../scripts/requirements.txt):

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
  benchmark-baseline.latest -> runs/benchmark-baseline.<run-id>/
  eval-baseline.latest -> runs/eval-baseline.<run-id>/
  runs/
    benchmark-baseline.<run-id>/
      benchmark.json
      benchmark.html
      benchmark.log
      run.json
    eval-baseline.<run-id>/
      output/  # lm-eval scores and per-question samples
      eval.log
      run.json
```

Each run has its own directory. After a successful run, `scripts/results.py`
checks the expected report and writes a `run.json` manifest with a unique run
ID, tool versions, and workload settings. The client reads `/v1/models` at the
start and end of the workload. It records the checkpoint root and context
length when available. For a server it can match to the lesson launcher, it
also records the mode, backend, and server instance. For a local Linux
checkpoint it records the source of the bind mount. The `.latest` link moves
only after these checks pass. If the observed server changes or the workload
fails, the previous successful result stays available.

## What `benchmark.sh` does

`benchmark.sh` checks that a server answers at `/v1/models`, then runs one
GuideLLM sweep. These are the options, with simple standalone output paths:

```bash
guidellm run \
  --backend kind=openai_http,target=http://127.0.0.1:8000,model=qwen3-4b \
  --profile "kind=sweep,max_concurrency=${MAX_CONCURRENCY:-512}" \
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

The default concurrency cap is 512. The walkthrough recommends
`MAX_CONCURRENCY=8` on Apple Silicon and a smaller workload on CPU. Keep the
same settings for both sides of a comparison.

GuideLLM downloads the HTML template when the benchmark finishes. Opening
the report also needs internet access because the browser loads the chart
code. The saved JSON contains the measurements, and `guidellm export` can
convert it to CSV offline.

## What `evaluate.sh` does

`evaluate.sh` runs lm-eval against the same server. With a standalone output
path, the command looks like this:

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
- The output goes into a new run directory; only the `.latest` link for that
  label changes after success.

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

Before comparing, it checks the saved workload settings and evaluation
metadata, including the task version and sample count. Mismatched or missing
metadata produces an error so you can rerun both sides with matching settings.
It also validates the GuideLLM report structure and the reported workload.

The serving endpoint recorded in each task's `metadata.base_url` is excluded
from the comparison, allowing matching runs served on different ports. All
other task configuration and metadata checks still apply, including tokenizer,
generation settings, samples, and seeds. Each label's benchmark and evaluation
must agree on the serving settings they both recorded. Independent runs of one
checkpoint can still be compared, including runs with different context
lengths. Older manifests with matching workloads and tools can be compared
with a notice that the checkpoint identity is unavailable.

## Complete local cleanup

`./cleanup.sh` lists what it will remove and asks for confirmation. It stops the
server, then removes:

- On Linux, the owned `vllm-zero-to-hero-pt3` container and its labeled
  `vllm-zero-to-hero-pt3-models` cache volume.
- The local tools environment created by this version of the lesson and its temporary state.
  Pre-existing environments and directories supplied through `TOOLS_VENV` are kept.

Cleanup preserves `results/` and the older Part 1/3 shared volume. On Apple
Silicon, both MLX checkpoints and the shared Metal environment remain available
to other lessons.

## Tested support

The [Part 3 RFC](https://github.com/red-hat-ai-dev/vLLM-zero-to-hero-pt3/issues/3)
holds validation results, exact environments, and remaining platform checks.
Performance and accuracy depend on the model, workload, and hardware recorded
with each result.

[Back to the walkthrough](../README.md)
