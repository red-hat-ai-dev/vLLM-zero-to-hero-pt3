# Tested support: Part 3

This is a tutorial validation record, not a vendor support guarantee. Status is
specific to the versions and models below. Last reviewed: 2026-10-07.

## Platform record

| Path | Configuration | Evidence |
| --- | --- | --- |
| Linux / NVIDIA | RTX 3090 24 GiB, Podman 5.4.2, vLLM 0.28.0 | Baseline/FP8: ten-strategy sweeps, full 1,319-example GSM8K evaluations, comparison, stop, and cleanup passed |
| Apple Silicon | Apple M4 Pro 48 GiB, vLLM Metal 0.29.0; MLX Qwen3-4B BF16 versus 4-bit | Both models passed API checks, ten-strategy sweeps at concurrency eight, 20-example GSM8K evaluations, comparison, and stop |
| AMD / Intel | ROCm / XPU launch paths | Exact benchmark workflow and optimized model unverified |

Client tools are pinned to GuideLLM 0.7.4 and lm-eval 0.4.13. Part 3 consumes
Part 1's public legacy GHCR image and overrides its model argument explicitly.
Parts 1 and 3 share the managed container name/volume and Metal PID directory;
stop the active lesson before starting another.

The [2026-10-07 hardware record](validation/2026-10-07.json) includes the image
digest, source model revision, FP8 shard hashes, commands, and workload settings.
The full sweeps used the default 30 seconds per strategy, 256 prompt tokens, and
128 output tokens. GSM8K strict-match accuracy was 85.06% for baseline and 84.38%
for FP8; flexible-extract accuracy was 84.46% and 83.47%, respectively.
This is one paired run, not a repeated performance study. The sweeps recorded
zero errored requests but 721 baseline and 673 FP8 requests incomplete at the
measurement cutoff. Cleanup preserved all smoke/full result JSON hashes and
both Part 2 checkpoints. The separate 20-example smoke evaluations are retained
in the record and must not be confused with the full evaluation.

The [Metal hardware record](validation/2026-10-07-metal.json) includes the
successful run and previous failed attempts. With `MAX_CONCURRENCY=8`, both
models completed ten strategies at 30 seconds, using 256 prompt and 128 output
tokens. The sweeps recorded 132/187 successful, zero errored, and 75/46 incomplete
requests for BF16/4-bit. Both 20-example GSM8K evaluations and the comparison
passed; exact-match accuracy was 80%/85%. These small evaluations validate the
pipeline and do not establish a quality difference. Both servers stopped.

The earlier 512-concurrency BF16 throughput attempts completed no requests in
either 10 or 30 seconds (541/534 incomplete requests). They were rejected without
publishing latest-success pointers and are retained as failed attempts.
`MAX_CONCURRENCY` defaults to 512; the tested Metal workflow explicitly uses
eight for both models. Its throughput is bounded by that cap. Metal compares
BF16 with MLX 4-bit; Linux compares BF16 with FP8. Model-cache cleanup on the Mac
was not exercised against the shared cache.

## Result integrity

Every wrapper run writes to a separate `results/runs/` directory. A validated
successful run atomically updates `benchmark-LABEL.latest` or
`eval-LABEL.latest`. Failed runs keep logs without replacing the last success.
Previous outputs remain available. Concurrent writers use separate run directories;
the last successfully published run becomes latest. Run measurements sequentially
against one server to avoid distorting their performance.

Comparison checks workload (including the concurrency cap), tokenizer/client
versions and lm-eval task
configuration, task versions, few-shot settings, effective sample counts,
generation settings, and seeds. It rejects missing provenance and mismatches.
Old benchmark output without wrapper provenance needs a new run. Missing metric
distributions display `--`; Poisson strategies are not labeled constant rate.
Only bounded accuracy metrics are converted to percent correct.

Short sweeps and limited evaluations are smoke tests. Use the full tutorial
workload and full task dataset for performance/quality conclusions, and keep
hardware, tokenizer, prompt/output lengths, tasks, and sample counts matched.

## Reproduce the checks

Run `python3 -m unittest discover -s tests -v` for the committed regression and
documentation checks. The pull-request workflow runs these on Linux and macOS
with Python 3.10 and 3.13. These tests use fake engines; they do not establish GPU
support. Successful CI, a successful image build, and successful inference are
separate kinds of evidence.

For a hardware result, record the commit, date, OS/architecture, GPU and driver,
engine/version, image digest or Python packages, exact model revision, command,
request payload, response, and logs. For speculative decoding include accepted
and drafted token counters. Record failed steps too. Do not include credentials
or model weights in an issue or PR.

## Remaining validation

AMD ROCm, Intel XPU, and WSL2 have not been exercised by this change. Do not infer
support for the tutorial's exact model and quantization from an upstream backend
support table. Maintainers should attach equivalent hardware evidence before
marking those paths tested. No image publication or GPU CI is performed by the
pull-request checks.
