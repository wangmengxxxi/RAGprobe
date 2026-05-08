from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from python_retriever import retrieve


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length).decode("utf-8"))

        if "queries" in payload:
            body = {
                "results": [
                    {
                        "test_case_id": item["id"],
                        "query": item["query"],
                        "retrieved": retrieve(item["query"], payload.get("top_k", 10)),
                    }
                    for item in payload["queries"]
                ]
            }
        else:
            body = retrieve(payload["query"], payload.get("top_k", 10))

        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args) -> None:
        return


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8008), Handler)
    print("Listening on http://127.0.0.1:8008/search")
    server.serve_forever()
