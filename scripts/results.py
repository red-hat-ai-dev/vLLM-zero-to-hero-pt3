"""Keep versioned run output and atomically publish a pointer after success."""

from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import uuid


def validate_benchmark_report(data):
    if data.get("metadata", {}).get("version") != 2:
        raise ValueError("Unsupported GuideLLM report schema; expected version 2")
    benchmarks = data.get("benchmarks")
    if not isinstance(benchmarks, list) or not benchmarks:
        raise ValueError("GuideLLM returned no benchmarks")
    for benchmark in benchmarks:
        counts = benchmark.get("scheduler_metrics", {}).get("requests_made", {})
        if (
            not isinstance(counts.get("successful"), (int, float))
            or counts["successful"] <= 0
        ):
            raise ValueError(
                "A benchmark has no successful requests or is missing request counts"
            )
        if not benchmark.get("config", {}).get("strategy", {}).get("type_"):
            raise ValueError("Benchmark is missing its strategy configuration")
        if not isinstance(benchmark.get("metrics"), dict):
            raise ValueError("Benchmark is missing metrics")
    return benchmarks


def publish_run(
    kind, label, directory, results, serving_at_start=None, serving_at_finish=None
):
    directory, results = Path(directory).resolve(), Path(results).resolve()
    if directory.parent != results / "runs":
        raise ValueError("Run directory must be inside results/runs")
    if kind == "benchmark":
        data = json.loads((directory / "benchmark.json").read_text())
        validate_benchmark_report(data)
    elif kind == "eval":
        files = list(directory.glob("**/results_*.json"))
        if len(files) != 1 or not json.loads(files[0].read_text()).get("results"):
            raise ValueError("Expected one completed lm-eval result file")
    else:
        raise ValueError("Unknown result kind")
    metadata = {
        "run_id": uuid.uuid4().hex,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "label": label,
        "platform": platform.platform(),
        "tools": {
            name: importlib.metadata.version(name) for name in ("guidellm", "lm-eval")
        },
        "workload": {
            key: os.environ.get(key)
            for key in (
                (
                    "TOKENIZER",
                    "PROMPT_TOKENS",
                    "OUTPUT_TOKENS",
                    "MAX_SECONDS",
                    "MAX_CONCURRENCY",
                )
                if kind == "benchmark"
                else ("TOKENIZER", "TASKS", "LIMIT", "NUM_CONCURRENT")
            )
        },
    }
    if serving_at_start is not None:
        initial = json.loads(serving_at_start)
        final = json.loads(serving_at_finish)
        if initial != final:
            raise ValueError(
                "Serving identity changed or could not be verified during this run; rerun it"
            )
        metadata["serving"] = initial
    (directory / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    link = results / f".{kind}-{label}.{uuid.uuid4().hex}"
    link.symlink_to(Path("runs") / directory.name, target_is_directory=True)
    try:
        os.replace(link, results / f"{kind}-{label}.latest")
    finally:
        link.unlink(missing_ok=True)


if __name__ == "__main__":
    publish_run(*sys.argv[1:])
