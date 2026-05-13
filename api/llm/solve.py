from http.server import BaseHTTPRequestHandler
import json
import os
import sys

sys.path.append(os.path.dirname(__file__))
from _ollamafree import (
    default_model,
    extract_json,
    get_client,
    normalize_guesses,
    read_json,
    write_json,
)


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            body = read_json(self)
        except Exception:
            write_json(self, 400, {"error": "Invalid JSON request body."})
            return

        board = [str(item) for item in body.get("board", [])] if isinstance(body.get("board"), list) else []
        if len(board) != 16:
            write_json(self, 400, {"error": "A 16-item board is required."})
            return

        try:
            num_categories = max(1, min(4, int(body.get("numCategories") or 4)))
        except Exception:
            num_categories = 4
        try:
            requested_max = int(body.get("maxGuesses") or 8)
        except Exception:
            requested_max = 8
        max_guesses = max(num_categories, min(10, requested_max))
        model = str(body.get("model") or default_model()).strip() or default_model()

        prompt = "\n".join(
            [
                "You are playing a Connections-style puzzle.",
                f"Find {num_categories} groups of exactly 4 terms from the board.",
                "Use only exact terms from the board. Do not reuse a term across guesses.",
                "Return strict JSON only with this schema:",
                '{"guesses":[{"label":"short category label","items":["term","term","term","term"]}],"notes":"one short sentence"}',
                "",
                f"Board: {json.dumps(board)}",
            ]
        )

        try:
            client = get_client()
            response = client.chat(
                prompt=prompt,
                model=model,
                temperature=float(body.get("temperature") or 0.2),
                num_predict=700,
            )
            parsed = extract_json(response)
            if not parsed:
                write_json(
                    self,
                    502,
                    {
                        "error": "Model did not return parseable JSON.",
                        "raw": str(response)[:2000],
                    },
                )
                return

            write_json(
                self,
                200,
                {
                    "model": model,
                    "guesses": normalize_guesses(parsed, board, max_guesses),
                    "notes": str(parsed.get("notes", ""))[:300] if isinstance(parsed, dict) else "",
                    "raw": str(response),
                },
            )
        except Exception as error:
            write_json(
                self,
                500,
                {
                    "error": "Could not run OllamaFreeAPI model solver.",
                    "details": str(error),
                },
            )
