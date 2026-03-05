import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):

    def do_GET(self):

        if self.path == "/health":

            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")

        else:

            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"BTC Predictor Running")


def run_server():

    port = int(os.environ.get("PORT", 8080))

    server = HTTPServer(("0.0.0.0", port), Handler)

    print(f"KeepAlive server running on port {port}")

    server.serve_forever()


def start_keep_alive():

    thread = threading.Thread(target=run_server)

    thread.daemon = True

    thread.start()
