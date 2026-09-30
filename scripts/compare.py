"""Print the GuideLLM and lm-eval results of two runs side by side."""

import glob
import json
import os
import sys

# A constant-rate strategy counts as sustained when the server completed at
# least this fraction of the requests per second that GuideLLM asked for.
SUSTAINED_FRACTION = 0.9


def load_benchmarks(results_dir, label):
    path = os.path.join(results_dir, f"benchmark-{label}.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)["benchmarks"]


def stat(benchmark, metric, key="median", status="successful"):
    values = benchmark["metrics"][metric][status]
    if key.startswith("p"):
        return values["percentiles"][key]
    return values[key]


def summarize(benchmarks):
    by_type = {}
    for benchmark in benchmarks:
        by_type.setdefault(benchmark["config"]["strategy"]["type_"], []).append(benchmark)

    summary = {}
    sync = by_type.get("synchronous", [None])[0]
    if sync:
        summary["Single request: TTFT median (ms)"] = stat(sync, "time_to_first_token_ms")
        summary["Single request: ITL median (ms)"] = stat(sync, "inter_token_latency_ms")
        summary["Single request: latency median (s)"] = stat(sync, "request_latency")

    peak = by_type.get("throughput", [None])[0]
    if peak:
        summary["Max throughput: requests/s"] = stat(peak, "requests_per_second", "mean")
        summary["Max throughput: output tokens/s"] = stat(peak, "output_tokens_per_second", "mean", "total")
        summary["Max throughput: TTFT p95 (ms)"] = stat(peak, "time_to_first_token_ms", "p95")

    sustained = None
    for benchmark in by_type.get("constant", []) + by_type.get("poisson", []):
        rate = benchmark["config"]["strategy"].get("rate")
        achieved = stat(benchmark, "requests_per_second", "mean")
        if rate and achieved >= SUSTAINED_FRACTION * rate:
            sustained = max(sustained or 0, rate)
    if sustained is not None:
        summary["Highest sustained constant rate (req/s)"] = sustained
    return summary


def load_eval(results_dir, label):
    files = glob.glob(os.path.join(results_dir, f"eval-{label}", "**", "results_*.json"), recursive=True)
    if not files:
        return None
    with open(max(files, key=os.path.getmtime)) as handle:
        results = json.load(handle)["results"]
    scores = {}
    for task, metrics in results.items():
        for name, value in metrics.items():
            if "," not in name or "stderr" in name or not isinstance(value, (int, float)):
                continue
            metric, _, filter_name = name.partition(",")
            scores[f"{task} {metric} ({filter_name})"] = value * 100
    return scores


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


def print_table(title, rows, first, second, delta):
    print(title)
    width = max([len(row[0]) for row in rows] + [10])
    print(f"  {'':<{width}}  {first:>12}  {second:>12}  {'Change':>10}")
    for name, before, after in rows:
        print(f"  {name:<{width}}  {fmt(before):>12}  {fmt(after):>12}  {delta(before, after):>10}")
    print()


def main():
    results_dir, first, second = sys.argv[1], sys.argv[2], sys.argv[3]
    found = False

    first_bench = load_benchmarks(results_dir, first)
    second_bench = load_benchmarks(results_dir, second)
    if first_bench and second_bench:
        found = True
        a, b = summarize(first_bench), summarize(second_bench)
        rows = [(name, a.get(name), b.get(name)) for name in dict.fromkeys(list(a) + list(b))]
        print_table("Performance (GuideLLM sweep)", rows, first, second, change)
        print("  Lower is better for TTFT, ITL, and latency. Higher is better for throughput and rate.")
        print()
    else:
        missing = first if not first_bench else second
        print(f"No GuideLLM results for {missing}. Run ./benchmark.sh {missing} first.")
        print()

    first_eval = load_eval(results_dir, first)
    second_eval = load_eval(results_dir, second)
    if first_eval and second_eval:
        found = True
        rows = [(name, first_eval.get(name), second_eval.get(name)) for name in dict.fromkeys(list(first_eval) + list(second_eval))]
        points = lambda before, after: "--" if before is None or after is None else f"{after - before:+.1f} pts"
        print_table("Accuracy (lm-eval, % correct)", rows, first, second, points)
    else:
        missing = first if not first_eval else second
        print(f"No lm-eval results for {missing}. Run ./evaluate.sh {missing} first.")
        print()

    return 0 if found else 1


if __name__ == "__main__":
    sys.exit(main())
