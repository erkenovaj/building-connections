import json
import os
import re
import urllib.request


def write_json(handler, status, payload):
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def read_json(handler):
    length = int(handler.headers.get("content-length", "0") or "0")
    if length <= 0:
        return {}
    raw = handler.rfile.read(length).decode("utf-8")
    return json.loads(raw or "{}")


class LocalOllama:
    """Minimal client for a locally running Ollama server (``/api`` endpoints)."""

    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/")

    def chat(self, prompt, model, temperature=0.2, num_predict=400, think=False):
        """Run a single non-streaming completion and return the response text.

        When ``think`` is true the model reasons in a separate channel: its
        chain-of-thought is captured on ``self.last_thinking`` and the returned
        text holds only the final answer. Give such calls a larger
        ``num_predict`` so the reasoning does not exhaust the answer budget.
        """
        body = json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "think": think,
                "options": {"temperature": temperature, "num_predict": num_predict},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = json.loads(response.read().decode("utf-8"))
        self.last_thinking = payload.get("thinking", "") or ""
        return payload.get("response", "")

    def list_models(self):
        """Return the names of models installed on the local Ollama server."""
        with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return [model["name"] for model in payload.get("models", [])]


def get_client():
    if os.getenv("LLM_BACKEND", "free").lower() == "local":
        return LocalOllama(os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    from ollamafreeapi import OllamaFreeAPI

    return OllamaFreeAPI()


def default_model():
    return os.getenv("LLM_MODEL", "llama3.2:3b")


def extract_json(text):
    raw = str(text or "").strip()
    if not raw:
        return None
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if not raw:
        return None
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) >= 3:
            raw = parts[1].removeprefix("json").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def normalize_one_guess(parsed, remaining):
    remaining_set = set(remaining)
    guess = None
    if isinstance(parsed, dict):
        candidate = parsed.get("guess")
        if isinstance(candidate, dict):
            guess = candidate
        elif isinstance(parsed.get("items"), list):
            guess = parsed
    if not isinstance(guess, dict):
        return None
    raw_items = guess.get("items")
    if not isinstance(raw_items, list):
        raw_items = guess.get("words")
    if not isinstance(raw_items, list):
        return None
    items = []
    for item in raw_items:
        term = str(item or "").strip()
        if term and term in remaining_set and term not in items:
            items.append(term)
        if len(items) == 4:
            break
    if len(items) != 4:
        return None
    label = str(guess.get("label") or guess.get("category") or "Guess")[:80]
    return {"label": label, "items": items}


def normalize_guesses(parsed, board, max_guesses):
    board_set = set(board)
    source = []
    if isinstance(parsed, dict):
        if isinstance(parsed.get("guesses"), list):
            source = parsed["guesses"]
        elif isinstance(parsed.get("groups"), list):
            source = parsed["groups"]

    guesses = []
    for index, guess in enumerate(source[:max_guesses]):
        if not isinstance(guess, dict):
            continue
        raw_items = guess.get("items")
        if not isinstance(raw_items, list):
            raw_items = guess.get("words")
        if not isinstance(raw_items, list):
            continue

        items = []
        for item in raw_items:
            term = str(item or "").strip()
            if term and term in board_set and term not in items:
                items.append(term)
            if len(items) == 4:
                break
        if len(items) != 4:
            continue

        label = str(guess.get("label") or guess.get("category") or f"Guess {index + 1}")[:80]
        guesses.append({"label": label, "items": items})
    return guesses
