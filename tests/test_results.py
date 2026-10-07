import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare, results = load("compare"), load("results")


def evaluation(samples=200):
    return {
        "configs": {
            "gsm8k": {"dataset_path": "gsm8k", "generation_kwargs": {"temperature": 0}}
        },
        "versions": {"gsm8k": 3},
        "n-shot": {"gsm8k": 5},
        "n-samples": {"gsm8k": {"original": 1319, "effective": samples}},
        "config": {"limit": samples},
        "results": {"gsm8k": {"exact_match,strict-match": 0.5}},
    }


class ComparisonTests(unittest.TestCase):
    def test_matching_evaluations(self):
        compare.validate_compatible_runs(evaluation(), evaluation())

    def test_unequal_sample_counts_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "n-samples"):
            compare.validate_compatible_runs(evaluation(), evaluation(1319))

    def test_changed_task_definition_is_rejected(self):
        other = copy.deepcopy(evaluation())
        other["configs"]["gsm8k"]["generation_kwargs"]["temperature"] = 1
        with self.assertRaisesRegex(ValueError, "configs"):
            compare.validate_compatible_runs(evaluation(), other)

    def test_missing_metadata_is_not_treated_as_comparable(self):
        other = evaluation()
        del other["versions"]
        with self.assertRaisesRegex(ValueError, "missing versions"):
            compare.validate_compatible_runs(evaluation(), other)

    def test_only_accuracy_metrics_are_percentages(self):
        data = evaluation()
        data["results"]["gsm8k"]["perplexity,none"] = 10
        self.assertEqual(list(compare.evaluation_scores(data).values()), [50])

    def test_empty_metric_does_not_crash(self):
        self.assertIsNone(
            compare.stat(
                {"metrics": {"request_latency": {"successful": None}}},
                "request_latency",
            )
        )

    def test_poisson_is_not_reported_as_constant(self):
        self.assertNotIn(
            "Highest sustained constant rate (req/s)",
            compare.summarize(
                [
                    {
                        "config": {"strategy": {"type_": "poisson", "rate": 1}},
                        "metrics": {},
                    }
                ]
            ),
        )


class PublicationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        (self.root / "runs").mkdir()

    def run_dir(self, name, data):
        p = self.root / "runs" / name
        p.mkdir()
        data.setdefault("metadata", {"version": 2})
        (p / "benchmark.json").write_text(json.dumps(data))
        return p

    def publish(self, p):
        with patch.object(results.importlib.metadata, "version", return_value="test"):
            results.publish_run("benchmark", "baseline", p, self.root)

    def test_successful_rerun_keeps_previous_output(self):
        a = self.run_dir(
            "a",
            {
                "benchmarks": [
                    {
                        "config": {"strategy": {"type_": "synchronous"}},
                        "scheduler_metrics": {"requests_made": {"successful": 1}},
                        "metrics": {},
                    }
                ]
            },
        )
        self.publish(a)
        b = self.run_dir(
            "b",
            {
                "benchmarks": [
                    {
                        "config": {"strategy": {"type_": "synchronous"}},
                        "scheduler_metrics": {"requests_made": {"successful": 2}},
                        "metrics": {},
                    }
                ]
            },
        )
        self.publish(b)
        self.assertEqual((self.root / "benchmark-baseline.latest").resolve(), b)
        self.assertTrue((a / "benchmark.json").exists())

    def test_empty_or_invalid_rerun_keeps_latest(self):
        a = self.run_dir(
            "a",
            {
                "benchmarks": [
                    {
                        "config": {"strategy": {"type_": "synchronous"}},
                        "scheduler_metrics": {"requests_made": {"successful": 1}},
                        "metrics": {},
                    }
                ]
            },
        )
        self.publish(a)
        b = self.run_dir("b", {"benchmarks": []})
        with self.assertRaises(ValueError):
            self.publish(b)
        self.assertEqual((self.root / "benchmark-baseline.latest").resolve(), a)

    def test_workload_mismatch_is_rejected(self):
        for field, first, second in (
            ("PROMPT_TOKENS", "256", "512"),
            ("MAX_CONCURRENCY", "8", "512"),
            ("MAX_CONCURRENCY", "8", None),
        ):
            with self.subTest(field=field, second=second):
                for label, value in (("baseline", first), ("optimized", second)):
                    p = self.root / f"benchmark-{label}.latest"
                    p.mkdir(exist_ok=True)
                    workload = {field: value} if value is not None else {}
                    (p / "run.json").write_text(
                        json.dumps({"workload": workload, "tools": {}})
                    )
                with self.assertRaisesRegex(ValueError, "workloads"):
                    compare.validate_benchmark_workloads(
                        self.root, "baseline", "optimized"
                    )

    def test_published_run_records_concurrency(self):
        data = json.loads(
            (SCRIPTS.parent / "tests/fixtures/guidellm-0.7.4.json").read_text()
        )
        run = self.run_dir("concurrency", data)
        with patch.dict("os.environ", {"MAX_CONCURRENCY": "8"}):
            self.publish(run)
        metadata = json.loads((run / "run.json").read_text())
        self.assertEqual(metadata["workload"]["MAX_CONCURRENCY"], "8")


class FailedWrapperTests(unittest.TestCase):
    def test_failed_tool_preserves_previous_results(self):
        import os
        import subprocess
        import sys

        for script, command, kind in [
            ("benchmark.sh", "guidellm", "benchmark"),
            ("evaluate.sh", "lm_eval", "eval"),
        ]:
            with self.subTest(script=script), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                bin_dir = root / "tools/bin"
                bin_dir.mkdir(parents=True)
                for tool in ("guidellm", "lm_eval"):
                    p = bin_dir / tool
                    p.write_text("#!/bin/sh\necho intentional tool failure\nexit 7\n")
                    p.chmod(0o755)
                p = bin_dir / "python"
                p.write_text("#!/bin/sh\nexit 0\n")
                p.chmod(0o755)
                data = root / "results"
                data.mkdir()
                old = data / "previous"
                old.mkdir()
                (old / "sentinel").write_text("keep")
                (data / f"{kind}-baseline.latest").symlink_to(
                    "previous", target_is_directory=True
                )
                # Legacy files must survive too.
                (data / f"{kind}-baseline.json").write_text("legacy")
                env = dict(
                    os.environ, TOOLS_VENV=str(root / "tools"), RESULTS_DIR=str(data)
                )
                result = subprocess.run(
                    ["/bin/sh", str(SCRIPTS.parent / script), "baseline"],
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=10,
                )
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(
                    (data / f"{kind}-baseline.latest" / "sentinel").read_text(), "keep"
                )
                self.assertEqual((data / f"{kind}-baseline.json").read_text(), "legacy")
                self.assertIn("intentional tool failure", result.stdout)


class IncompleteResultTests(unittest.TestCase):
    def test_zero_successful_requests_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "no successful"):
            results.validate_benchmark_report(
                {
                    "metadata": {"version": 2},
                    "benchmarks": [
                        {
                            "scheduler_metrics": {
                                "requests_made": {"successful": 0, "errored": 5}
                            }
                        }
                    ],
                }
            )

    def test_unknown_result_schema_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "schema"):
            results.validate_benchmark_report(
                {"metadata": {"version": 3}, "benchmarks": []}
            )


class RetainedFixtureTests(unittest.TestCase):
    def test_guidellm_074_units_and_percentiles(self):
        data = json.loads(
            (SCRIPTS.parent / "tests/fixtures/guidellm-0.7.4.json").read_text()
        )
        benchmarks = results.validate_benchmark_report(data)
        summary = compare.summarize(benchmarks)
        self.assertEqual(
            summary["Single request: TTFT median (ms)"],
            benchmarks[0]["metrics"]["time_to_first_token_ms"]["successful"]["median"],
        )
        self.assertEqual(
            summary["Max throughput: TTFT p95 (ms)"],
            benchmarks[1]["metrics"]["time_to_first_token_ms"]["successful"][
                "percentiles"
            ]["p95"],
        )
        self.assertGreater(summary["Sweep requests: incomplete"], 0)


class MalformedAccuracyTests(unittest.TestCase):
    def test_out_of_range_accuracy_is_rejected(self):
        data = evaluation()
        data["results"]["gsm8k"]["exact_match,strict-match"] = 2
        with self.assertRaisesRegex(ValueError, "outside"):
            compare.evaluation_scores(data)

    def test_unbounded_only_metrics_are_not_called_accuracy(self):
        data = evaluation()
        data["results"]["gsm8k"] = {"perplexity,none": 10}
        with self.assertRaisesRegex(ValueError, "No supported"):
            compare.evaluation_scores(data)


class ToolSetupTests(unittest.TestCase):
    def test_failed_install_preserves_existing_environment(self):
        import os
        import subprocess

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            venv = root / "tools"
            bin_dir = venv / "bin"
            bin_dir.mkdir(parents=True)
            (venv / "sentinel").write_text("keep")
            p = bin_dir / "python"
            p.write_text("#!/bin/sh\nexit 7\n")
            p.chmod(0o755)
            python = root / "python"
            python.write_text("#!/bin/sh\nexit 0\n")
            python.chmod(0o755)
            env = dict(os.environ, TOOLS_VENV=str(venv), PYTHON=str(python))
            result = subprocess.run(
                ["/bin/sh", "-c", ". ./scripts/tools.sh; ensure_tools"],
                cwd=SCRIPTS.parent,
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((venv / "sentinel").read_text(), "keep")


class EvaluationFixtureTests(unittest.TestCase):
    def fixture(self):
        return json.loads(
            (SCRIPTS.parent / "tests/fixtures/lm-eval-0.4.13.json").read_text()
        )

    def test_pinned_evaluation_parses_scores(self):
        data = self.fixture()
        compare.validate_compatible_runs(data, data)
        self.assertEqual(
            compare.evaluation_scores(data)["gsm8k exact_match (strict-match)"], 80
        )

    def test_actual_seed_fields_are_compared(self):
        for key in ("numpy_seed", "torch_seed", "fewshot_seed"):
            with self.subTest(key=key):
                a = self.fixture()
                b = copy.deepcopy(a)
                b["config"][key] = 99
                with self.assertRaisesRegex(ValueError, key):
                    compare.validate_compatible_runs(a, b)


if __name__ == "__main__":
    unittest.main()
