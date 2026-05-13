from http.server import BaseHTTPRequestHandler
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connections_sampler import (  # noqa: E402
    GameMode,
    SamplingParameters,
    sample_basic_game,
    sample_game,
)


DEFAULT_CONFIG = ROOT / "configs" / "category-templates-new.json"


def _write_json(handler, status, payload):
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_json(handler):
    length = int(handler.headers.get("content-length", "0") or "0")
    if length <= 0:
        return {}
    return json.loads(handler.rfile.read(length).decode("utf-8") or "{}")


def _load_default_config():
    with DEFAULT_CONFIG.open("r", encoding="utf-8") as file:
        return json.load(file)


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        try:
            body = _read_json(self)
        except Exception:
            _write_json(self, 400, {"error": "Invalid JSON request body."})
            return

        config = body.get("config")
        if config is None:
            config = _load_default_config()
        if not isinstance(config, list) or not config:
            _write_json(self, 400, {"error": "A non-empty config array is required."})
            return

        try:
            mode = GameMode.parse(body.get("mode") or GameMode.BASIC.value)
        except ValueError:
            _write_json(
                self,
                400,
                {"error": "Mode must be easy, basic, or advanced."},
            )
            return

        parameters = SamplingParameters(
            seed=body.get("seed"),
            max_attempts=40,
            per_solve_timeout_seconds=0.75,
            total_timeout_seconds=8.0,
        )

        try:
            num_categories = body.get("numCategories")
            items_per_category = body.get("itemsPerCategory")
            if num_categories is not None or items_per_category is not None:
                if mode is not GameMode.BASIC:
                    raise ValueError(
                        "Custom P x Q dimensions are available only in Basic mode."
                    )
                p = int(num_categories or 4)
                q = int(items_per_category or 4)
                if p < 1 or q < 1 or p * q > 64:
                    raise ValueError(
                        "P and Q must be positive and produce at most 64 board items."
                    )
                result = sample_basic_game(
                    config,
                    num_categories=p,
                    items_per_category=q,
                    parameters=parameters,
                )
            else:
                result = sample_game(config, mode=mode, parameters=parameters)
        except ValueError as error:
            _write_json(self, 400, {"error": str(error)})
            return
        except Exception as error:
            _write_json(
                self,
                500,
                {
                    "error": "The sampler failed unexpectedly.",
                    "details": str(error),
                },
            )
            return

        status = 200 if result.ok else (503 if result.status == "timeout" else 422)
        _write_json(self, status, result.to_dict())
