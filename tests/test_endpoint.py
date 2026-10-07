"""Exercise endpoint validation against a real loopback HTTP server."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]


class EndpointTests(unittest.TestCase):
    def check_response(self, status, body):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(status)
                self.end_headers()
                self.wfile.write(json.dumps(body).encode())

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as td:
                tools = Path(td)
                (tools / "bin").mkdir()
                (tools / "bin/python").symlink_to(sys.executable)
                env = dict(
                    os.environ, PORT=str(server.server_port), TOOLS_VENV=str(tools)
                )
                return subprocess.run(
                    ["/bin/sh", "-c", ". ./scripts/tools.sh; require_server"],
                    cwd=ROOT,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_expected_model_is_accepted(self):
        self.assertEqual(
            self.check_response(200, {"data": [{"id": "qwen3-4b"}]}).returncode, 0
        )

    def test_unrelated_model_is_rejected(self):
        result = self.check_response(200, {"data": [{"id": "other"}]})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Expected model alias", result.stderr)

    def test_server_error_is_rejected(self):
        self.assertNotEqual(
            self.check_response(503, {"error": "not ready"}).returncode, 0
        )
