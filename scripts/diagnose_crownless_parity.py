"""Small CPU diagnostic for the published Crownless sentence across kernels."""
import json
import hashlib
from pathlib import Path
import torch
from tokenizers import Tokenizer
from torch.nn.attention import sdpa_kernel, SDPBackend
from crownless_v2 import encode_row, generate
from crownless_v2_export import load_export


def run():
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1] / 'models/crownless-core-v2'
    tokenizer = Tokenizer.from_file(str(root / 'tokenizer.json'))
    prefix = '- Éva posts a notice at Newhaven: Flood relief.\n'
    fields = []
    for slot, (text, role) in enumerate([('Éva', 1), ('Newhaven', 3), ('Flood relief', 4)]):
        start = len(prefix[:prefix.index(text)].encode())
        fields.append(dict(field=slot, text=text, start=start, end=start+len(text.encode()),
                           role=role, spoken=True, knowledge=0, provenance=3, event=1))
    for name in ['native', 'math', 'double', 'epsilon']:
        model, meta = load_export(root / 'core.ccv2', root / 'tokenizer.json')
        row = dict(id='diagnostic', kind_id=meta['meaning_ids']['notice_posted_0'], prefix=prefix,
                   output='', fields=fields, copies=[])
        record = encode_row(tokenizer, row, slots=True, packet=True)
        if name == 'native':
            fingerprint = {key: hashlib.sha256(value.detach().numpy().tobytes()).hexdigest()
                           for key, value in model.state_dict().items()}
            print(json.dumps({'model_sha': hashlib.sha256((root / 'core.ccv2').read_bytes()).hexdigest(),
                              'record': record, 'weights': fingerprint,
                              'cosine': model.cosine[:4].tolist(), 'sine': model.sine[:4].tolist()}, sort_keys=True), flush=True)
        if name == 'double': model.double()
        if name == 'epsilon':
            for module in model.modules():
                if isinstance(module, torch.nn.RMSNorm): module.eps = 1e-6
        trace = []; original = model.heads
        def heads(*args, **kwargs):
            logits, gate, scores = original(*args, **kwargs)
            values, ids = logits[0, -1].topk(3)
            trace.append({'gate':gate.item(), 'top':[(tokenizer.id_to_token(i),v) for i,v in zip(ids.tolist(),values.tolist())]})
            return logits, gate, scores
        model.heads = heads
        if name == 'math':
            with sdpa_kernel(SDPBackend.MATH): result = generate(model, tokenizer, record)
        else: result = generate(model, tokenizer, record)
        print(json.dumps({'mode':name,'text':result['text'],'trace':trace[:6]},ensure_ascii=False),flush=True)


if __name__ == '__main__': run()
