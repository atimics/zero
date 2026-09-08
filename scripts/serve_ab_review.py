"""Serve the local A/B review and append durable, idempotent preference events."""
import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from ab_review import validate_vote


class VoteStore:
    def __init__(self, path, packet):
        self.path, self.packet = Path(path), packet
        self.lock = threading.Lock()
        self.events = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                e = validate_vote(json.loads(line), packet)
                if e['event_id'] in self.events and self.events[e['event_id']] != e:
                    raise ValueError('Conflicting stored event ID')
                self.events[e['event_id']] = e

    def append(self, raw):
        e = validate_vote(raw, self.packet)
        with self.lock:
            if e['event_id'] in self.events:
                if self.events[e['event_id']] != e:
                    raise ValueError('Event ID already used for another vote')
                return False
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('a') as f:
                f.write(json.dumps(e)+'\n'); f.flush(); os.fsync(f.fileno())
            self.events[e['event_id']] = e
            return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--responses', type=Path, required=True)
    p.add_argument('--port', type=int, default=8795)
    a = p.parse_args()
    packet = json.loads((a.directory/'ab-packet.json').read_text())
    page = (a.directory/'ab-review.html').read_bytes()
    store = VoteStore(a.responses, packet)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def send(self, code, data, content_type='application/json'):
            body = data if isinstance(data, bytes) else json.dumps(data).encode()
            self.send_response(code); self.send_header('Content-Type',content_type)
            self.send_header('Cache-Control','no-store'); self.send_header('Content-Length',str(len(body)))
            self.end_headers(); self.wfile.write(body)
        def do_GET(self):
            if self.path in ['/', '/ab-review.html']:
                return self.send(200,page,'text/html; charset=utf-8')
            if self.path == '/api/review/status':
                return self.send(200,{'packet_id':packet['packet_id']})
            return self.send(404,{'error':'Unknown page'})
        def do_POST(self):
            if self.path != '/api/review/vote': return self.send(404,{'error':'Unknown route'})
            if self.headers.get('Origin') not in [None,f'http://127.0.0.1:{a.port}',f'http://localhost:{a.port}']:
                return self.send(403,{'error':'Open the local review page to submit picks.'})
            try:
                size = int(self.headers.get('Content-Length','0'))
                if not 0 < size <= 4096: raise ValueError('Invalid vote size')
                inserted = store.append(json.loads(self.rfile.read(size)))
                return self.send(200,{'saved':True,'new_event':inserted})
            except (ValueError,TypeError,KeyError):
                return self.send(400,{'error':'Invalid vote. Keep the browser copy and check the packet.'})
            except OSError:
                return self.send(503,{'error':'Saving failed. Keep the browser copy and retry.'})
    print(f'Open http://127.0.0.1:{a.port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()

if __name__ == '__main__': main()
