import json
import os


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


def get_client():
    from ollamafreeapi import OllamaFreeAPI

    return OllamaFreeAPI()


def default_model():
    return os.getenv("LLM_MODEL", "llama3.2:3b")


def extract_json(text):
    raw = str(text or "").strip()
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
            return json.loads(raw[start : end + 1])
    return None


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
