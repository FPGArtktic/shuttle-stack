# SPDX-License-Identifier: GPL-3.0-only
"""The HTTP client, against a server that answers from this process."""

from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from shuttle_delegate.backend import Backend, BackendError
from shuttle_delegate.config import Endpoint

ANSWERS = {
    "/completion": {
        "content": "  an answer  ",
        "timings": {"prompt_n": 12, "predicted_n": 3},
    },
    "/tokenize": {"tokens": [1, 2, 3, 4]},
    "/health": {"status": "ok"},
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: object) -> None:
        pass

    def _reply(self) -> None:
        if self.path not in ANSWERS:
            self.send_error(404, "no such endpoint")
            return
        body = json.dumps(ANSWERS[self.path]).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        self._reply()

    def do_POST(self) -> None:  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self._reply()


class BackendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = HTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(
            target=cls.server.serve_forever, daemon=True
        )
        cls.thread.start()
        host, port = cls.server.server_address[:2]
        cls.backend = Backend(Endpoint("fast", f"http://{host}:{port}"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.thread.join(timeout=5)

    def test_completion_is_stripped_and_counted(self) -> None:
        answer = self.backend.complete("hello", n_predict=8)
        self.assertEqual(answer.content, "an answer")
        self.assertEqual(answer.tokens_in, 12)
        self.assertEqual(answer.tokens_out, 3)
        self.assertEqual(answer.tokens, 15)

    def test_tokens_are_counted(self) -> None:
        self.assertEqual(self.backend.count_tokens("whatever"), 4)

    def test_health_is_a_question_not_an_error(self) -> None:
        self.assertTrue(self.backend.healthy())

    def test_an_http_error_names_the_server_and_the_path(self) -> None:
        with self.assertRaises(BackendError) as caught:
            self.backend.props()
        message = str(caught.exception)
        self.assertIn("shuttle-fast", message)
        self.assertIn("/props", message)

    def test_a_closed_port_suggests_where_to_look(self) -> None:
        down = Backend(Endpoint("long", "http://127.0.0.1:1"))
        self.assertFalse(down.healthy())
        with self.assertRaises(BackendError) as caught:
            down.complete("hello", n_predict=1)
        self.assertIn("shuttle-long.service", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
