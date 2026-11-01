"""Local server that behaves like the real thing: gzip, and TCP slow start.
Each response leaves in flights of 14KB, 28KB, 56KB with one round trip between them.
"""
import gzip, http.server, mimetypes, os, sys, time
RTT = float(sys.argv[2]) if len(sys.argv) > 2 else .15
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        p = self.path.split('?')[0]
        p = '.' + (p + 'index.html' if p.endswith('/') else p)
        if not os.path.isfile(p):
            self.send_error(404); return
        body = open(p, 'rb').read()
        kind = mimetypes.guess_type(p)[0] or 'application/octet-stream'
        self.send_response(200)
        self.send_header('Content-Type', kind)
        if kind.startswith('text/'):
            body = gzip.compress(body, 6)
            self.send_header('Content-Encoding', 'gzip')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        time.sleep(RTT)  # the request's own round trip, here so flights stay separate
        at, flight = 0, 14000
        while at < len(body):
            self.wfile.write(body[at:at + flight]); self.wfile.flush()
            at += flight; flight *= 2
            if at < len(body): time.sleep(RTT)
http.server.ThreadingHTTPServer(('127.0.0.1', int(sys.argv[1])), H).serve_forever()
