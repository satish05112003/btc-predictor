import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"BTC Predictor Running")


def run_server():

    server = HTTPServer(("0.0.0.0", 8080), Handler)

    print("KeepAlive server running on port 8080")

    server.serve_forever()


def start_keep_alive():

    thread = threading.Thread(target=run_server)

    thread.daemon = True

    thread.start()
