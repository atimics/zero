"""Build paired fact changes from the Crownless v2 gossip corpus."""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path

# Named groups are facts copied from the simulation's account and spoken line.
PATTERNS = [
 ('NOTICE', r'(?P<person>.+) posts a notice at (?P<place>.+): (?P<subject>.+)\.'),
 ('DEATH', r'(?P<person>.+) died at age \d+ after a life in (?P<place>.+)\.'),
 ('DRAGON FIRE', r'(?P<dragon>.+) burns (?P<place>.+) because \d+ stolen crowns remain missing\.'),
 ('DRAGON OMEN', r"Smoke falls into (?P<place>.+)'s chimneys; old readers count \d+ nights until (?P<dragon>.+) comes\."),
 ('GOBLIN RAID', r'(?P<faction>.+) raids (?P<place>.+): .+\.'),
 ('SHORTAGE', r'(?P<place>.+) has \d+ (?:food in store\..+|weeks of food;.+)'),
 ('HARVEST', r"(?P<place>.+)'s drought harvest cannot supply the (?P<direction>\w+) settlements\."),
 ('ROUTE', r'(?P<place>.+) closes the treaty bridge and delays the relief convoy\.'),
 ('PEACE', r"(?P<faction>.+)'s courier reaches (?P<other>.+): peace now binds the .+ courts\."),
 ('TREASURE', r'(?P<place>.+) finishes (?P<object>.+) from \d+ Raw Gold, \d+ Gems, and \d+ weeks of work\.'),
 ('BANDITS', r'(?P<faction>.+) recruits from hungry debtors and unpaid households; road control reaches \d+%\.'),
 ('BANDITS', r'Displaced workers reinforce (?P<faction>.+) on the old road\.'),
 ('CULT RALLIES', r"(?P<faction>.+) gathers \d+ new tithe-bearers; (?P<dragon>.+)'s court returns to \d+\."),
]
# Two additional input forms for training; the third is reserved for the wording test.
WORDING = {
 'NOTICE': ['{person} posted a {subject} in {place}.', 'In {place}, {person} put up a {subject}.', 'A {subject} was posted in {place} by {person}.'],
 'DEATH': ['{person} died after living in {place}.', '{person} lived in {place} and has died.', 'News from {place}: {person} has died.'],
 'DRAGON FIRE': ['{dragon} burned {place} over stolen hoard money.', 'Stolen hoard money led {dragon} to burn {place}.', '{place} was burned by {dragon} over money taken from the hoard.'],
 'DRAGON OMEN': ["Smoke in {place}'s chimneys warns of {dragon}.", '{place} saw smoke in its chimneys; readers feared {dragon}.', 'Readers feared {dragon} after smoke entered chimneys in {place}.'],
 'GOBLIN RAID': ['{faction} raided {place} and took supplies.', '{place} lost supplies in a raid by {faction}.', 'Supplies were taken from {place} when {faction} raided it.'],
 'SHORTAGE': ['{place} was running short of food.', 'Food reserves were low in {place}.', 'The food stored in {place} fell below what was needed.'],
 'HARVEST': ["{place}'s drought harvest fell short of the {direction} settlements' needs.", 'Drought in {place} left too little harvest for the {direction} settlements.', 'The {direction} settlements needed more than the drought harvest from {place} could supply.'],
 'ROUTE': ['{place} shut the treaty bridge and delayed the relief convoy.', 'The relief convoy was delayed after {place} shut the treaty bridge.', 'Closing the treaty bridge at {place} held up the relief convoy.'],
 'PEACE': ['{faction} and {other} made peace.', '{faction} agreed to peace with {other}.', 'Peace was agreed between {faction} and {other}.'],
 'TREASURE': ['{place} finished making {object}.', 'Craftspeople in {place} completed {object}.', '{object} was completed by craftspeople in {place}.'],
 'BANDITS': ['{faction} recruited hungry debtors and unpaid households.', 'Hungry debtors and unpaid households joined {faction}.', '{faction} gained recruits from hungry debtors and unpaid households.'],
 'CULT RALLIES': ["{faction} gathered new tithe-bearers for {dragon}'s court.", "New tithe-bearers joined {faction} at {dragon}'s court.", "At {dragon}'s court, {faction} gained new tithe-bearers."],
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def templates(root):
    rows = [json.loads(l) for l in (root / 'train.audit.jsonl').read_text().splitlines()]
    found = {}
    for row in rows:
        kind = row['input']['kind']
        for i, (k, pattern) in enumerate(PATTERNS):
            match = re.fullmatch(pattern, row['input']['account']) if k == kind else None
            if not match:
                continue
            facts = match.groupdict()
            # Longest first: a treasure may contain the settlement's name.
            def replace(text):
                for key, val in sorted(facts.items(), key=lambda item: -len(item[1])):
                    text = re.sub(re.escape(val), lambda m: '{' + key + '}', text, flags=re.I)
                return text
            source, target = replace(row['input']['account']), replace(row['output'])
            required = re.findall(r'\{(\w+)\}', target)
            if not required or any(facts[key] not in row['input']['account'] for key in required):
                continue
            # Keep the source rule and uncertainty state together.
            key = (i, source, target, row['input']['confidence'] < 40, row['input']['retellings'] >= 4)
            found.setdefault(key, dict(kind=kind, pattern=i, source=source, target=target,
                                      fields=list(facts), required=sorted(set(required)),
                                      source_id=row['id'], confidence=row['input']['confidence'],
                                      retellings=row['input']['retellings'],
                                      # This subtype describes reinforcement, rather than recruitment.
                                      paraphrase=not (kind == 'BANDITS' and 'reinforce' in source)))
            break
    if {t['kind'] for t in found.values()} != set(WORDING):
        raise ValueError('Missing gossip event kinds')
    return list(found.values())


def pools():
    first = 'Alder Amber Ash Birch Bracken Briar Cedar Cinder Copper Dusk Elm Fern Flint Frost Hazel Heather Holly Iron Ivy Juniper Lark Laurel Linden Maple Mist Moss Oak Pine Reed Rose Rowan Sable Sage Silver Stone Thorn Willow Winter'.split()
    last = 'bank barrow beck borough bridge brook burg burn cliff coast cove crest dale den fell fen field ford gate glen grove hall haven heath hill hollow holt keep lake land marsh meadow mere mill moor mound point port reach ridge rock shore spire spring stead stone vale view wall watch water well wick wood'.split()
    people = 'Ada Alden Arlen Bera Bram Bren Cora Dain Della Doran Edda Elin Eren Fara Fenn Fira Galen Halen Iona Ira Jora Kellan Leda Lira Loren Mera Mira Nera Nola Orin Orla Perrin Rena Rian Rilla Sera Seren Talen Tara Torin Vela Wren'.split()
    jobs = 'Baker Bell Brook Carter Cooper Dyer Field Fisher Fletcher Forester Gardener Hunter Keeper Mason Miller Potter Reed Roper Shepherd Smith Stone Tanner Thatcher Weaver Wood Wright'.split()
    values = {
      'place': [a+b for a in first for b in last],
      'person': [a+' '+b for a in people for b in jobs],
      'faction': [a+' '+b for a in first for b in ['Court','Banner','Company','League','Knives','Throne','Council','Oath','Order','Covenant','Republic','Compact']],
      'dragon': [a+b for a in first for b in ['wing','fang','claw','scale','flame','tooth','tail','maw']],
      'object': [a+' '+b for a in first for b in ['Cup','Crown','Lark','Ring','Bell','Icon','Lantern','Chalice']],
    }
    result = {s: {} for s in ['train','validation','test']}
    for role, names in values.items():
        for s, index in [('train',0),('validation',1),('test',2)]:
            result[s][role] = [n for n in names if (0 if int(sha(n.encode())[:8],16)%10 < 8 else 1 if int(sha(n.encode())[:8],16)%10 == 8 else 2) == index]
        assert set(result['train'][role]).isdisjoint(result['test'][role])
    for s in result:
        result[s]['other'] = result[s]['faction']
        result[s]['subject'] = ['relief charter','road compact','quiet commission','repair contract','supply request','guard commission']
        result[s]['direction'] = ['northern','southern','eastern','western']
    return result


def render(t, facts, form):
    pattern = t['source'] if form == 0 or not t['paraphrase'] else WORDING[t['kind']][form-1]
    cue = ('? ' if t['confidence'] < 40 else '') + ('~ ' if t['retellings'] >= 4 else '')
    return '- ' + cue + pattern.format(**facts) + '\n', t['target'].format(**facts)


def build(source, output, pairs=12000, seed=29):
    ts, ps = templates(source), pools()
    output.mkdir(parents=True, exist_ok=False)
    by_kind = {k: [t for t in ts if t['kind']==k] for k in WORDING}
    manifest = dict(version=3, seed=seed, source_sha256=sha((source/'train.audit.jsonl').read_bytes()),
                    source_manifest_sha256=sha((source/'manifest.json').read_bytes()),
                    templates=len(ts), source='Crownless v2 training split', splits={},
                    note='Synthetic fact substitutions preserve source event rules; earlier events are distractors.')
    for split, count in [('train',pairs),('validation',120),('test',120),('wording_test',120)]:
        rng = random.Random(seed + ['train','validation','test','wording_test'].index(split))
        pool = ps['test' if split == 'wording_test' else split]
        text, audit = bytearray(), []
        for i in range(count):
            kind = list(WORDING)[i % len(WORDING)]
            t = rng.choice([t for t in by_kind[kind] if split != 'wording_test' or t['paraphrase']])
            facts = {k:rng.choice(pool[k]) for k in t['fields']}
            while 'other' in facts and facts['other']==facts['faction']:
                facts['other']=rng.choice(pool['other'])
            # Only score named facts, with exact names reserved by split.
            named = [k for k in t['required'] if k not in ['subject','direction']]
            field = rng.choice(named)
            changed = dict(facts)
            changed[field] = rng.choice([v for v in pool[field] if v not in facts.values()])
            form = 3 if split == 'wording_test' else rng.randrange(3)
            context = ''
            for _ in range(rng.choices([0,1,2], [6,3,1])[0]):
                other = rng.choice(ts)
                otherfacts = {k:rng.choice(pool[k]) for k in other['fields']}
                context += render(other, otherfacts, 0)[0]
            longest = max(len(render(t, f, form)[0]) + len(render(t, f, form)[1]) + 1 for f in [facts,changed])
            while len(context) + longest > 513 and context:
                context = context.split('\n',1)[1]
            for variant, f in enumerate([facts,changed]):
                event, speech = render(t,f,form)
                prefix = context + event
                # Use one context for both members; trim only whole earlier events.
                while len(prefix.encode()) + max(len(t['target'].format(**v).encode()) for v in [facts,changed]) + 1 > 513 and prefix.count('\n')>1:
                    prefix=prefix.split('\n',1)[1]
                if len(prefix.encode())+len(speech.encode())+1>513:
                    raise ValueError('Example exceeds base context')
                start=len(text); text.extend((prefix+speech+'\n\n').encode('ascii'))
                audit.append(dict(version=3,id=f'{split}-{i}-{variant}',pair_id=f'{split}-{i}',
                    changed_field=field,changed_from=facts[field],changed_to=changed[field],
                    source_id=t['source_id'], pattern=t['pattern'], form=form,
                    input=dict(kind=kind,confidence=t['confidence'],retellings=t['retellings']),
                    facts=f,required={k:f[k] for k in named},output=speech,
                    text_start=start,output_start=start+len(prefix.encode()),text_bytes=len(text)-start))
        (output/(split+'.txt')).write_bytes(text)
        (output/(split+'.audit.jsonl')).write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in audit))
        manifest['splits'][split]=dict(rows=len(audit),bytes=len(text),sha256=sha(text))
    # Preserve the original separately authored challenge unchanged.
    for suffix in ['txt','audit.jsonl']:
        (output/('editorial_test.'+suffix)).write_bytes((source/('editorial_test.'+suffix)).read_bytes())
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pairs',type=int,default=12000)
    p.add_argument('--seed',type=int,default=29)
    a=p.parse_args()
    if a.pairs < 12: p.error('Use at least twelve pairs')
    print(json.dumps(build(a.source,a.output,a.pairs,a.seed),indent=2))
