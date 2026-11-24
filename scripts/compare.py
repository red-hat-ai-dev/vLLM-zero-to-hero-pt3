"""Print the GuideLLM and lm-eval results of two runs side by side."""

from copy import deepcopy
import json
import math
from pathlib import Path
import os
import sys
from results import validate_benchmark_report

# A constant-rate strategy counts as sustained when the server completed at
# least this fraction of the requests per second that GuideLLM asked for.
SUSTAINED_FRACTION = 0.9


def load_benchmarks(results_dir, label, run_dir=None):
    path = os.path.join(results_dir, f"benchmark-{label}.json")
    latest = (
        run_dir or Path(results_dir) / f"benchmark-{label}.latest"
    ) / "benchmark.json"
    if latest.exists():
        path = latest
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return validate_benchmark_report(json.load(handle))


def stat(benchmark, metric, key="median", status="successful"):
    values = benchmark["metrics"].get(metric, {}).get(status)
    if not values:
        return None
    value = (
        values.get("percentiles", {}).get(key)
        if key.startswith("p")
        else values.get(key)
    )
    return value if isinstance(value, (int, float)) and math.isfinite(value) else None


def summarize(benchmarks):
    by_type = {}
    for benchmark in benchmarks:
        by_type.setdefault(benchmark["config"]["strategy"]["type_"], []).append(
            benchmark
        )

    summary = {}
    for status in ("successful", "errored", "incomplete"):
        summary[f"Sweep requests: {status}"] = sum(
            benchmark.get("scheduler_metrics", {})
            .get("requests_made", {})
            .get(status, 0)
            for benchmark in benchmarks
        )
    sync = by_type.get("synchronous", [None])[0]
    if sync:
        summary["Single request: TTFT median (ms)"] = stat(
            sync, "time_to_first_token_ms"
        )
        summary["Single request: ITL median (ms)"] = stat(
            sync, "inter_token_latency_ms"
        )
        summary["Single request: latency median (s)"] = stat(sync, "request_latency")

    peak = by_type.get("throughput", [None])[0]
    if peak:
        summary["Max throughput: requests/s"] = stat(
            peak, "requests_per_second", "mean"
        )
        summary["Max throughput: output tokens/s"] = stat(
            peak, "output_tokens_per_second", "mean", "total"
        )
        summary["Max throughput: TTFT p95 (ms)"] = stat(
            peak, "time_to_first_token_ms", "p95"
        )

    sustained = None
    for benchmark in by_type.get("constant", []):
        rate = benchmark["config"]["strategy"].get("rate")
        achieved = stat(benchmark, "requests_per_second", "mean")
        if rate and achieved is not None and achieved >= SUSTAINED_FRACTION * rate:
            sustained = max(sustained or 0, rate)
    if sustained is not None:
        summary["Highest sustained constant rate (req/s)"] = sustained
    return summary


def load_eval(results_dir, label, run_dir=None):
    latest = run_dir or Path(results_dir) / f"eval-{label}.latest"
    root = latest if latest.exists() else Path(results_dir) / f"eval-{label}"
    files = list(root.glob("**/results_*.json"))
    if not files:
        return None
    if len(files) != 1:
        raise ValueError(
            f"Ambiguous evaluation results for {label}: expected one result file"
        )
    return json.loads(files[0].read_text())


def evaluation_scores(data):
    scores = {}
    for task, metrics in data["results"].items():
        for name, value in metrics.items():
            if (
                "," not in name
                or "stderr" in name
                or not isinstance(value, (int, float))
                or isinstance(value, bool)
            ):
                continue
            metric, _, filter_name = name.partition(",")
            # Only bounded accuracy metrics have the meaning '% correct'.
            if metric not in ("acc", "acc_norm", "exact_match") or not math.isfinite(
                value
            ):
                continue
            if not 0 <= value <= 1:
                raise ValueError(f"Accuracy metric {task}/{name} is outside [0, 1]")
            scores[f"{task} {metric} ({filter_name})"] = value * 100
    if not scores:
        raise ValueError("No supported bounded accuracy metrics were found")
    return scores


def comparison_task_configs(configs):
    """Exclude serving endpoints without changing the recorded task definitions."""
    configs = deepcopy(configs)
    for task in configs.values():
        metadata = task.get("metadata")
        if isinstance(metadata, dict):
            metadata.pop("base_url", None)
    return configs


def validate_compatible_runs(first, second):
    """Reject unequal task definitions or effective samples before printing deltas."""
    for key in ("configs", "versions", "n-shot", "n-samples"):
        if key not in first or key not in second:
            raise ValueError(
                f"Evaluation is missing {key}; compatibility cannot be established"
            )
        a, b = first[key], second[key]
        if key == "configs":
            a, b = comparison_task_configs(a), comparison_task_configs(b)
        if a != b:
            raise ValueError(
                f"Evaluation {key} differ; rerun with matching tasks and sample counts"
            )
    for key in (
        "limit",
        "batch_size",
        "gen_kwargs",
        "random_seed",
        "numpy_seed",
        "torch_seed",
        "fewshot_seed",
    ):
        if first.get("config", {}).get(key) != second.get("config", {}).get(key):
            raise ValueError(f"Evaluation {key} differ")
    for key in (
        "lm_eval_version",
        "task_hashes",
        "system_instruction_sha",
        "chat_template_sha",
        "fewshot_as_multiturn",
    ):
        if first.get(key) != second.get(key):
            raise ValueError(f"Evaluation {key} differ")
    if set(first["results"]) != set(second["results"]):
        raise ValueError("Evaluation tasks differ")


def validate_benchmark_workloads(
    results_dir, first, second, kind="benchmark", directories=None
):
    """Check wrapper provenance, including tokenizer, workload, and client versions."""
    runs = []
    if directories is None:
        directories = [
            (Path(results_dir) / f"{kind}-{label}.latest").resolve()
            for label in (first, second)
        ]
    for directory in directories:
        path = directory / "run.json"
        if not path.exists():
            raise ValueError(
                f"{kind} provenance is missing; rerun both runs with this version"
            )
        runs.append(json.loads(path.read_text()))
    if directories[0] == directories[1] or (
        runs[0].get("run_id") and runs[0]["run_id"] == runs[1].get("run_id")
    ):
        raise ValueError(
            f"Both labels point to the same saved {kind} run; collect two independent runs"
        )
    if (
        not runs[0].get("run_id")
        and runs[0].get("created_at_utc")
        and runs[0] == runs[1]
    ):
        raise ValueError(
            f"Both labels carry the same dated {kind} manifest; check for a copied run"
        )
    if (
        runs[0]["workload"] != runs[1]["workload"]
        or runs[0]["tools"] != runs[1]["tools"]
    ):
        raise ValueError(
            f"{kind} workloads or tool versions differ; rerun with matching settings"
        )
    return [run.get("serving") or {} for run in runs]


def report_serving(labels, records):
    """Check what each column represents before printing either metric table."""
    settings = ("checkpoint", "backend", "image_id", "max_model_len")
    messages = []
    for index, label in enumerate(labels):
        observations = [
            pair[index] for pair in records.values() if pair[index] is not None
        ]
        for key in (*settings, "mode"):
            values = {item[key] for item in observations if item.get(key) is not None}
            if len(values) > 1:
                raise ValueError(
                    f"Benchmark and evaluation for {label} used different {key} values; rerun the mismatched measurement"
                )
        for item in observations:
            mode = item.get("mode")
            if label in ("baseline", "optimized") and mode and mode != label:
                messages.append(
                    f"Warning: {label} was collected with the launcher in {mode} mode."
                )
        known = [item for item in observations if item.get("checkpoint")]
        if known:
            checkpoint = known[0]["checkpoint"]
            mode = next((item["mode"] for item in known if item.get("mode")), "unknown")
            messages.append(f"{label} served {checkpoint} (launcher mode: {mode}).")
    for first, second in records.values():
        if first is None or second is None:
            continue
        if not first.get("checkpoint") or not second.get("checkpoint"):
            messages.append(
                "Checkpoint identity is unavailable for one or more runs; verify which models were served."
            )
        elif first["checkpoint"] == second["checkpoint"]:
            changed = [
                key
                for key in settings
                if first.get(key) is not None
                and second.get(key) is not None
                and first[key] != second[key]
            ]
            if changed:
                messages.append(
                    f"Both runs served {first['checkpoint']} with different settings ({', '.join(changed)})."
                )
            else:
                messages.append(
                    f"Warning: both runs served {first['checkpoint']}. Confirm that the same checkpoint was intended before interpreting changes."
                )
    for message in dict.fromkeys(messages):
        print(message, file=sys.stderr)


def fmt(value):
    if value is None:
        return "--"
    if abs(value) >= 100:
        return f"{value:,.0f}"
    return f"{value:.2f}"


def change(before, after):
    if before is None or after is None or before == 0:
        return "--"
    return f"{(after - before) / before * 100:+.1f}%"


def points(before, after):
    return "--" if before is None or after is None else f"{after - before:+.1f} pts"


def print_table(title, rows, first, second, delta):
    print(title)
    width = max([len(row[0]) for row in rows] + [10])
    print(f"  {'':<{width}}  {first:>12}  {second:>12}  {'Change':>10}")
    for name, before, after in rows:
        print(
            f"  {name:<{width}}  {fmt(before):>12}  {fmt(after):>12}  {delta(before, after):>10}"
        )
    print()


def main():
    results_dir, first, second = sys.argv[1], sys.argv[2], sys.argv[3]
    found = False

    labels = first, second
    # A concurrently completed run may move .latest. Keep each column bound
    # to one archived directory while reading reports and their manifests.
    directories = {
        kind: [
            (Path(results_dir) / f"{kind}-{label}.latest").resolve() for label in labels
        ]
        for kind in ("benchmark", "eval")
    }
    first_bench, second_bench = (
        load_benchmarks(results_dir, label, directory)
        for label, directory in zip(labels, directories["benchmark"])
    )
    first_eval, second_eval = (
        load_eval(results_dir, label, directory)
        for label, directory in zip(labels, directories["eval"])
    )
    records = {}
    if first_eval and second_eval:
        validate_compatible_runs(first_eval, second_eval)
    for kind, reports in (
        ("benchmark", (first_bench, second_bench)),
        ("eval", (first_eval, second_eval)),
    ):
        if all(reports):
            records[kind] = validate_benchmark_workloads(
                results_dir, first, second, kind, directories[kind]
            )
        elif any(reports):
            # A single available result can still contradict the other kind
            # for this label, even when its counterpart is missing.
            records[kind] = []
            for report, directory in zip(reports, directories[kind]):
                manifest = directory / "run.json"
                serving = None
                if report and manifest.is_file():
                    serving = json.loads(manifest.read_text()).get("serving")
                records[kind].append(serving)
    report_serving(labels, records)
    if first_bench and second_bench:
        found = True
        a, b = summarize(first_bench), summarize(second_bench)
        rows = [
            (name, a.get(name), b.get(name))
            for name in dict.fromkeys(list(a) + list(b))
        ]
        print_table("Performance (GuideLLM sweep)", rows, first, second, change)
        print(
            "  Lower is better for TTFT, ITL, and latency. Higher is better for throughput and rate."
        )
        if any(
            summary.get(f"Sweep requests: {status}", 0)
            for summary in (a, b)
            for status in ("errored", "incomplete")
        ):
            print(
                "  This sweep includes errored or incomplete requests; inspect the reports before drawing performance conclusions."
            )
        print()
    else:
        missing = first if not first_bench else second
        print(f"No GuideLLM results for {missing}. Run ./benchmark.sh {missing} first.")
        print()

    if first_eval and second_eval:
        found = True
        first_eval, second_eval = (
            evaluation_scores(first_eval),
            evaluation_scores(second_eval),
        )
        if set(first_eval) != set(second_eval):
            raise ValueError("Evaluation metrics differ or one result is incomplete")
        rows = [
            (name, first_eval.get(name), second_eval.get(name))
            for name in dict.fromkeys(list(first_eval) + list(second_eval))
        ]
        print_table("Accuracy (lm-eval, % correct)", rows, first, second, points)
    else:
        missing = first if not first_eval else second
        print(f"No lm-eval results for {missing}. Run ./evaluate.sh {missing} first.")
        print()

    return 0 if found else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"Cannot compare these runs: {exc}", file=sys.stderr)
        sys.exit(1)
