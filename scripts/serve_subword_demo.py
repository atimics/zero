"""Serve the subword comparison demo and local inference on loopback."""
import argparse
import json
import threading
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import torch
from tokenizers import Tokenizer
from subword_model import create,sample
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8765);args=parser.parse_args()
    torch.set_num_threads(2)
    tokenizer=Tokenizer.from_file(str(ROOT/'experiments/subword-scaling/tokenizer.json'))
    models={}
    for name in ['5m-256','5m-1024']:
        state=torch.load(ROOT/f'models/subword/{name}.pt',map_location='cpu',weights_only=True)
        model=create(state['config']);model.load_state_dict(state['model']);models[name]=model.eval()
    lock=threading.Lock()
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self,*a,**kw):super().__init__(*a,directory=str(ROOT/'docs/subword'),**kw)
        def log_message(self,*args):pass
        def respond(self,status,data):
            raw=json.dumps(data).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
        def do_GET(self):
            if self.path=='/api/status':return self.respond(200,{'ready':True})
            return super().do_GET()
        def do_POST(self):
            if self.path!='/api/generate':return self.respond(404,{'error':'Unknown route'})
            if self.headers.get('Origin') not in [None,f'http://127.0.0.1:{args.port}',f'http://localhost:{args.port}']:
                return self.respond(403,{'error':'Open the local demo to generate text.'})
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=16384:raise ValueError('Request is too large.')
                data=json.loads(self.rfile.read(size))
                if not isinstance(data,dict):raise ValueError('Expected a prompt object')
                prompt=data.get('prompt')
                if not isinstance(prompt,str) or not prompt.strip() or len(prompt)>2000:raise ValueError('Enter a prompt of 1 to 2,000 characters.')
            except (ValueError,TypeError):return self.respond(400,{'error':'Enter a prompt of 1 to 2,000 characters.'})
            if not lock.acquire(blocking=False):return self.respond(409,{'error':'Generation is in progress. Try again shortly.'})
            try:
                outputs={name:sample(model,tokenizer,prompt,128,7) for name,model in models.items()}
                self.respond(200,{'outputs':outputs,'mode':'Live · CPU FP32 · seed 7'})
            except Exception:
                self.respond(500,{'error':'Generation failed. Check the model files and try again.'})
            finally:lock.release()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'Open http://127.0.0.1:{args.port}',flush=True)
    server.serve_forever()
if __name__=='__main__':main()
