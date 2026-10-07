"""Check benchmark configuration before launching a real load generator."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class BenchmarkConfigurationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        tools = self.root / "tools/bin"
        tools.mkdir(parents=True)
        for name in ("python", "lm_eval"):
            p = tools / name
            p.write_text("#!/bin/sh\nexit 0\n")
            p.chmod(0o755)
        self.capture = self.root / "invocation.json"
        p = tools / "guidellm"
        p.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "Path(os.environ['BENCHMARK_TEST_CAPTURE']).write_text(json.dumps({"
            "'args': sys.argv[1:], 'concurrency': os.environ.get('MAX_CONCURRENCY')}))\n"
            "sys.exit(7)  # Capture configuration without generating load.\n"
        )
        p.chmod(0o755)

    def run_benchmark(self, concurrency=None):
        env = dict(
            os.environ,
            TOOLS_VENV=str(self.root / "tools"),
            RESULTS_DIR=str(self.root / "results"),
            BENCHMARK_TEST_CAPTURE=str(self.capture),
        )
        env.pop("MAX_CONCURRENCY", None)
        if concurrency is not None:
            env["MAX_CONCURRENCY"] = concurrency
        return subprocess.run(
            ["/bin/sh", str(ROOT / "benchmark.sh"), "baseline"],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )

    def test_default_and_override_reach_guidellm_and_publication_environment(self):
        for value, expected in ((None, "512"), ("8", "8")):
            with self.subTest(concurrency=value):
                self.capture.unlink(missing_ok=True)
                result = self.run_benchmark(value)
                self.assertNotEqual(result.returncode, 0)  # Stub exits after capture.
                invocation = json.loads(self.capture.read_text())
                args = invocation["args"]
                self.assertEqual(
                    args[args.index("--profile") + 1],
                    f"kind=sweep,max_concurrency={expected}",
                )
                self.assertEqual(invocation["concurrency"], expected)

    def test_invalid_concurrency_is_rejected_before_a_run_is_created(self):
        for value in ("0", "00", "-1", "1.5", "abc", "8,sweep_size=2"):
            with self.subTest(concurrency=value):
                result = self.run_benchmark(value)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    "MAX_CONCURRENCY must be a positive integer", result.stderr
                )
                self.assertFalse(self.capture.exists())
                self.assertFalse((self.root / "results").exists())
