"""Serves the current directory over HTTP with wide-open CORS headers, so
Label Studio (running on a different origin, e.g. localhost:8080) can fetch
images from it in the browser.

Usage:
    python cors_static_server.py [port]
"""

import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler


class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8898
    HTTPServer(("0.0.0.0", port), CORSRequestHandler).serve_forever()
