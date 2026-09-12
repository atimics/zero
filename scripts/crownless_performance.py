"""Matched creature and emotion examples for the Crownless conversation core."""
import copy
import hashlib
import random
from crownless_conversation import ACTS, response, approved_response

CREATURES = ('human', 'goblin', 'pony')
EMOTIONS = ('calm', 'afraid', 'relieved')
VERSION = 'crownless.performance.v1'
# These are authored performance conventions for the first controlled trial.
# The middle of the sentence retains the independently scored held claim.
OPENERS = {
    'human': ('Here is what I heard. ', 'Let me tell you. '),
    'goblin': ('Listen close, you. ', 'Got a tale for you. '),
    'pony': ('Oh, listen, dear. ', 'Gather close, dear. '),
}
ENDINGS = {
    'calm': (' Take your time.', ' We can talk quietly.'),
    'afraid': (' I am frightened.', ' My voice is shaking.'),
    'relieved': (' I can breathe again.', ' What a relief.'),
}


def control_text(performance):
    if not isinstance(performance, dict) or set(performance) != {'creature', 'emotion'}:
        raise ValueError('Performance needs creature and emotion')
    creature, emotion = performance['creature'], performance['emotion']
    if creature not in CREATURES or emotion not in EMOTIONS:
        raise ValueError('Unknown performance control')
    return f'voice: {creature}; feeling: {emotion}\n'


def styled(row, creature, emotion, variant=0):
    performance = {'creature': creature, 'emotion': emotion}
    control_text(performance)
    out = copy.deepcopy(row)
    opening = OPENERS[creature][variant % 2]
    out['plain_output'] = row['output']
    out['output'] = opening + row['output'] + ENDINGS[emotion][variant % 2]
    for span in out['copies']:
        span['start'] += len(opening.encode())
        span['end'] += len(opening.encode())
    out['performance'] = performance
    out['id'] += f':{creature}:{emotion}:{variant % 2}'
    return out


def plain(row):
    out = copy.deepcopy(row)
    if 'performance' in out:
        out['output'] = out['plain_output']
        del out['performance']
    return out


def score(row, rule, result):
    text = result['text']
    if 'performance' not in row:
        valid = result['stopped'] and text in approved_response(row, rule)
        return {'meaning': valid, 'identity': True, 'emotion': True, 'joint': valid}
    p = row['performance']
    # Strip every recognised style, so a correct claim with the wrong style
    # still receives a meaning pass and a separate style failure.
    opening = next((x for values in OPENERS.values() for x in values if text.startswith(x)), '')
    ending = next((x for values in ENDINGS.values() for x in values if text.endswith(x)), '')
    core = text[len(opening):len(text)-len(ending) if ending else len(text)]
    meaning = result['stopped'] and core in approved_response(plain(row), rule)
    identity = bool(opening) and opening in OPENERS[p['creature']]
    emotion = bool(ending) and ending in ENDINGS[p['emotion']]
    return {'meaning': meaning, 'identity': identity, 'emotion': emotion,
            'joint': meaning and identity and emotion}


def selected(bases, limit):
    """Round-robin across meanings while retaining each source-value pair."""
    groups = {}
    for row in bases:
        groups.setdefault(row['rule'], []).append(row)
    output, offset = [], 0
    while len(output) < limit:
        before = len(output)
        for rule in sorted(groups):
            output.extend(groups[rule][offset:offset+2])
        if len(output) == before: break
        offset += 2
    return output[:limit]


def corpus(bases, rules, seed, limit, held_rules=(), neutral=False, paraphrase=False):
    eligible = [r for r in bases if r['rule'] not in held_rules]
    rows = []
    for i, base in enumerate(selected(eligible, limit)):
        # Act assignment is independent of rule order and source pair position.
        number = int(hashlib.sha256(f'{seed}:{base["pair"]}'.encode()).hexdigest()[:8], 16)
        act = ACTS[number % len(ACTS)]
        item = response(base, rules[base['rule']], act, random.Random(f'{seed}:{base["id"]}'), paraphrase=paraphrase)
        if neutral:
            rows.append(item)
        else:
            for creature in CREATURES:
                for emotion in EMOTIONS:
                    rows.append(styled(item, creature, emotion, number // len(ACTS)))
    return rows
