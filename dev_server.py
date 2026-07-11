"""Minimal local dev server.

Serves the static frontend from the repo root and routes ``/api/*`` requests
to the Vercel-style serverless handlers under ``api/``. Lets the game run
locally (solo + agent modes) without the Vercel CLI.

Run: ``.venv/bin/python dev_server.py`` then open http://localhost:3000
"""

import importlib.util
import os
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Use the locally running Ollama server instead of the dead free API.
os.environ.setdefault("LLM_BACKEND", "local")
os.environ.setdefault("LLM_MODEL", "deepseek-r1:7b")

ROOT = Path(__file__).resolve().parent
PORT = 3000

# Map URL path -> serverless handler module file.
ROUTES = {
    "/api/game/sample": ROOT / "api" / "game" / "sample.py",
    "/api/llm/models": ROOT / "api" / "llm" / "models.py",
    "/api/llm/guess": ROOT / "api" / "llm" / "guess.py",
}


def _load_handler(module_path):
    """Import a serverless module by file path and return its ``handler`` class."""
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.handler


class DevHandler(SimpleHTTPRequestHandler):
    """Static file server that delegates ``/api/*`` to serverless handlers."""

    def _delegate(self, method):
        route = self.path.split("?", 1)[0]
        module_path = ROUTES.get(route)
        if module_path is None:
            self.send_error(404, "Unknown API route")
            return
        print(f"[api] -> {self.command} {route}", flush=True)
        started = time.monotonic()
        target_cls = _load_handler(module_path)
        # Build the handler instance without re-running BaseHTTPRequestHandler's
        # request loop, then borrow this request's I/O streams and metadata.
        target = target_cls.__new__(target_cls)
        target.rfile = self.rfile
        target.wfile = self.wfile
        target.headers = self.headers
        target.path = self.path
        target.command = self.command
        target.client_address = self.client_address
        target.server = self.server
        target.request_version = self.request_version
        target.requestline = self.requestline
        target.close_connection = True
        target._headers_buffer = []
        target.log_request = lambda *args, **kwargs: None
        try:
            getattr(target, method)()
        finally:
            print(
                f"[api] <- {route} done in {time.monotonic() - started:.1f}s",
                flush=True,
            )

    def do_POST(self):
        self._delegate("do_POST")

    def do_OPTIONS(self):
        if self.path.split("?", 1)[0] in ROUTES:
            self._delegate("do_OPTIONS")
        else:
            self.send_error(405)

    def do_GET(self):
        if self.path.split("?", 1)[0] in ROUTES:
            self._delegate("do_GET")
        else:
            super().do_GET()


def main():
    """Start the threaded dev server on ``PORT`` rooted at the repo."""
    handler = lambda *args, **kwargs: DevHandler(*args, directory=str(ROOT), **kwargs)
    with ThreadingHTTPServer(("127.0.0.1", PORT), handler) as httpd:
        print(f"Dev server on http://localhost:{PORT}  (Ctrl+C to stop)")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
