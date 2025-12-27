#!/usr/bin/env python3
"""
Simple HTTP server for the web frontend.
Run this to serve the autocomplete app locally.
"""

import http.server
import socketserver
import os
import webbrowser
from pathlib import Path
from urllib.parse import urlparse

PORT = 8000


class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers for local development
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

    def log_message(self, format, *args):
        # Better logging
        print(f"[{self.log_date_time_string()}] {format % args}")


def main():
    # Change to web-frontend directory
    web_dir = Path(__file__).parent
    os.chdir(web_dir)

    print("=" * 60)
    print("🚀 Autocomplete Web Server")
    print("=" * 60)
    print(f"📁 Serving from: {web_dir}")
    print(f"🌐 Server URL: http://localhost:{PORT}")
    print(f"\n💡 Press Ctrl+C to stop the server")
    print("=" * 60 + "\n")

    try:
        with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
            print(f"✅ Server running on http://localhost:{PORT}")
            print("   Opening browser...")
            webbrowser.open(f"http://localhost:{PORT}")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n❌ Server stopped")
    except OSError as e:
        print(f"\n❌ Error: {e}")
        if e.errno == 48:
            print(f"   Port {PORT} is already in use. Try another port:")
            print(f"   python3 server.py 8001")
        exit(1)


if __name__ == "__main__":
    main()
