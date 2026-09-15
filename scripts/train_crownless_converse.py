"""Train the Crownless 5M on a governed Braid Converse next-turn mix."""
import argparse, json, random, hashlib
from pathlib import Path
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from crownless_v2 import batch, encode_row
from crownless_v2_export import export, load_export
from crownless_conversation import build_rows
from train_crownless_conversation import read


def converse_row(record):
    history = [{'speaker': 'other' if i % 2 else 'self', 'text': turn['evidence']['text']}
               for i, turn in enumerate(record['turns'][:-1])]
    return {'id': record['id'], 'rule': 'converse', 'kind': 'CONVERSE', 'kind_id': 0,
            'prefix': '', 'output': record['turns'][-1]['evidence']['text'].strip().strip('"').strip(),
            'fields': [], 'copies': [], 'confidence': 80, 'retold': False,
            'voice': 'narrative', 'mind': {'goal': 'secure_livelihood', 'stress': 'medium',
            'courage': 'medium', 'memories': [], 'thoughts': []},
            'history': history, 'control': 'say', 'act': 'converse',
            'source_scene': record['sceneRecordId']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--converse', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--converse-ratio', type=float, default=0.02)
    p.add_argument('--seed', type=int, default=109)
    p.add_argument('--device', default='mps')
    args = p.parse_args()
    if args.output.exists(): p.error('Use a fresh output directory')
    torch.manual_seed(args.seed); random.seed(args.seed); torch.set_num_threads(4)
    model, metadata = load_export(args.base, args.tokenizer, args.device); model.mode = 'conversation'
    teacher, _ = load_export(args.base, args.tokenizer, args.device); teacher.eval()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    rules = {r['id']: r for r in json.loads((args.data / 'rules.json').read_text())['rules']}
    bases = {split: read(args.data / f'{split}.jsonl') for split in ('train', 'validation')}
    for rows in bases.values():
        for row in rows: row['kind_id'] = metadata['meaning_ids'][row['rule']]
    crownless = build_rows(bases['train'], rules, f'{args.seed}:train', repeats=2)
    converse = [converse_row(json.loads(line)) for line in args.converse.read_text().splitlines()]
    training = [encode_row(tokenizer, row, slots=True, conversation=True) for row in crownless]
    converse_records = []
    for row in converse:
        try:
            converse_records.append(encode_row(tokenizer, row, slots=True, conversation=True))
        except ValueError:
            continue
    validation = [encode_row(tokenizer, row, slots=True, conversation=True)
                  for row in build_rows(bases['validation'], rules, f'{args.seed}:validation')[:244]]
    args.output.mkdir(parents=True)
    manifest = {'base_sha256': hashlib.sha256(args.base.read_bytes()).hexdigest(),
                'tokenizer_sha256': hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
                'converse_sha256': hashlib.sha256(args.converse.read_bytes()).hexdigest(),
                'crownless_rows': len(training), 'converse_rows': len(converse_records),
                'converse_ratio': args.converse_ratio, 'steps': args.steps, 'seed': args.seed}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=.01)
    rng = random.Random(args.seed); best = float('inf'); history = []
    batch_size = 20; converse_count = max(1, round(batch_size * args.converse_ratio))
    for step in range(1, args.steps + 1):
        model.train()
        crown_records = rng.choices(training, k=batch_size - converse_count)
        converse_batch = rng.choices(converse_records, k=converse_count)
        inputs = batch(crown_records + converse_batch, args.device)
        crown_inputs = batch(crown_records, args.device)
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(inputs)
        student_hidden, _ = model.hidden(crown_inputs['tokens'], crown_inputs['meta'])
        with torch.no_grad(): teacher_hidden, _ = teacher.hidden(crown_inputs['tokens'], crown_inputs['meta'])
        student_logits, _, _ = model.heads(student_hidden, crown_inputs['candidates'], crown_inputs['candidate_mask'])
        with torch.no_grad(): teacher_logits, _, _ = teacher.heads(teacher_hidden, crown_inputs['candidates'], crown_inputs['candidate_mask'])
        distill = F.kl_div(F.log_softmax(student_logits, -1), F.softmax(teacher_logits, -1), reduction='batchmean')
        loss = loss + 0.1 * distill
        if not torch.isfinite(loss): raise ValueError('Non-finite loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step == 1 or step % 250 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad(): val = sum(model.loss(batch(validation[i:i+16], args.device)).item() for i in range(0, len(validation), 16)) / ((len(validation)+15)//16)
            item = {'step':step,'loss':loss.item(),'crownless_validation':val,'distill':distill.item()}; history.append(item)
            (args.output/'history.json').write_text(json.dumps(history,indent=2)+'\n'); print(json.dumps(item),flush=True)
            if val < best:
                best = val; torch.save({'state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'step':step},args.output/'best.pt')
    saved=torch.load(args.output/'best.pt',map_location='cpu',weights_only=True); model.to('cpu').load_state_dict(saved['state'])
    export(model,args.tokenizer,args.output/'core.ccv2',metadata|{'step':saved['step'],'seed':args.seed})
    print(json.dumps({'model_sha256':hashlib.sha256((args.output/'core.ccv2').read_bytes()).hexdigest(),'best_step':saved['step'],'converse_rows':len(converse_records)}))

if __name__ == '__main__': main()
