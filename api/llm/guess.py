from http.server import BaseHTTPRequestHandler
import json
import os
import sys

sys.path.append(os.path.dirname(__file__))
from _ollamafree import (
    default_model,
    extract_json,
    get_client,
    normalize_one_guess,
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

        remaining = (
            [str(item) for item in body.get("remaining", [])]
            if isinstance(body.get("remaining"), list)
            else []
        )
        if len(remaining) < 4:
            write_json(self, 400, {"error": "At least 4 remaining terms are required."})
            return

        try:
            num_categories = max(1, min(4, int(body.get("numCategories") or 4)))
        except Exception:
            num_categories = 4
        model = str(body.get("model") or default_model()).strip() or default_model()

        history = body.get("history")
        if not isinstance(history, list):
            history = []
        tried = []
        for entry in history:
            if isinstance(entry, dict) and isinstance(entry.get("items"), list):
                items = [str(item) for item in entry["items"]]
                if items:
                    tried.append(items)

        prompt_lines = [
            "You are playing a Connections-style puzzle, one group at a time.",
            f"This puzzle has {num_categories} groups total.",
            "Pick the single group of exactly 4 terms you are most confident about.",
            "Use only exact terms from the remaining list below.",
        ]
        if tried:
            prompt_lines.append(
                "Do not repeat any of these already-tried wrong groups: "
                + json.dumps(tried)
            )
        prompt_lines.extend(
            [
                "Return strict JSON only with this schema:",
                '{"guess":{"label":"short label","items":["term","term","term","term"]},"notes":"one short sentence"}',
                "",
                f"Remaining terms: {json.dumps(remaining)}",
            ]
        )
        prompt = "\n".join(prompt_lines)

        try:
            client = get_client()
            response = client.chat(
                prompt=prompt,
                model=model,
                temperature=float(body.get("temperature") or 0.2),
                num_predict=3000,
                think=True,
            )
            parsed = extract_json(response)
            write_json(
                self,
                200,
                {
                    "model": model,
                    "guess": normalize_one_guess(parsed, remaining),
                    "notes": str(parsed.get("notes", ""))[:300]
                    if isinstance(parsed, dict)
                    else "",
                    "thinking": str(getattr(client, "last_thinking", "") or "")[:4000],
                    "raw": str(response),
                },
            )
        except Exception as error:
            write_json(
                self,
                500,
                {
                    "error": "Could not run OllamaFreeAPI model guesser.",
                    "details": str(error),
                },
            )
