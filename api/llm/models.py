from http.server import BaseHTTPRequestHandler
import os
import sys

sys.path.append(os.path.dirname(__file__))
from _ollamafree import default_model, get_client, write_json


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            client = get_client()
            models = client.list_models()
            write_json(
                self,
                200,
                {
                    "models": models,
                    "defaultModel": default_model(),
                    "totalModels": len(models),
                },
            )
        except Exception as error:
            write_json(
                self,
                500,
                {
                    "error": "Could not list OllamaFreeAPI models.",
                    "details": str(error),
                },
            )
