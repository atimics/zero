"""Build response examples from held accounts and spoken events."""
import copy
import random
from score_crownless_v2 import accepted_forms

QUESTIONS = ['What happened?', 'What have you heard?', 'What is the news?']
CERTAINTY = ['How sure are you?', 'Are you sure?', 'Can we trust that account?']
CHECK = 'We should ask someone who was there.'
ASK = "Let's ask around before we pass it on."
CLOSE = "Agreed. Let's leave it there for now."
ACTS = ['start', 'question', 'agree', 'disagree', 'certainty', 'source', 'check', 'close', 'end']


def response(base, rule, act, rng, replacement=None, paraphrase=False):
    row = copy.deepcopy(base)
    own = row['output']
    heard = rng.choice(sorted(accepted_forms(row, rule)))
    changed = None
    candidates = [f for f in row['fields'] if f.get('spoken') and f['knowledge'] != 3 and
                  f['text'] in heard and f['role'] not in (0, 8)]
    if act == 'disagree' and candidates:
        field = rng.choice(candidates)
        alternative = replacement(field) if replacement else 'Farhaven'
        if alternative == field['text']: alternative = 'Another place'
        heard = heard.replace(field['text'], alternative)
        changed = {'field': field['field'], 'own': field['text'], 'heard': alternative}
    elif act == 'disagree': act = 'agree'
    history = []
    prefix, suffix = '', ''
    use_claim = False
    if act == 'start': target, use_claim = own, True
    elif act == 'question':
        history = [{'speaker': 'other', 'text': 'Tell me the news.' if paraphrase else rng.choice(QUESTIONS)}]
        target, use_claim = own, True
    elif act == 'agree':
        history = [{'speaker': 'other', 'text': heard}]
        target = "That's what I heard too. How sure are you?"
    elif act == 'disagree':
        history = [{'speaker': 'other', 'text': heard}]
        prefix, suffix = 'I heard a different account. ', ' How sure are you?'
        target, use_claim = prefix + own + suffix, True
    elif act == 'certainty':
        history = [{'speaker': 'self', 'text': own}, {'speaker': 'other', 'text':
            'How certain is that?' if paraphrase else rng.choice(CERTAINTY)}]
        if not paraphrase and rng.randrange(2):
            history[-1]['text'] = 'I heard a different account. ' + heard + ' ' + history[-1]['text']
        target = ('I am unsure. ' if row['confidence'] < 40 else
                  'The account has passed through several people. ' if row['retold'] else
                  'That is the account I hold. ') + CHECK
    elif act == 'source':
        history = [{'speaker': 'self', 'text': own}, {'speaker': 'other', 'text':
                   'Where did that story come from?' if paraphrase else 'Who told you?'}]
        target = 'That is the account I hold. ' + CHECK
    elif act == 'check':
        history = [{'speaker': 'self', 'text': rng.choice(CERTAINTY)},
                   {'speaker': 'other', 'text': rng.choice(['I am unsure. ', 'That is the account I hold. ',
                       'The account has passed through several people. ']) + CHECK}]
        target = ASK
    elif act == 'close':
        history = [{'speaker': 'self', 'text': CHECK}, {'speaker': 'other', 'text': ASK}]
        target = CLOSE
    elif act == 'end':
        history = [{'speaker': 'self', 'text': ASK}, {'speaker': 'other', 'text': CLOSE}]
        target = 'Very well.'
    else: raise ValueError(act)
    if history and rng.randrange(2):
        first = history[0]['speaker']
        history = [{'speaker': first, 'text': rng.choice(QUESTIONS)},
                   {'speaker': 'self' if first == 'other' else 'other', 'text': own}] + history
    row.update(output=target, history=history, act=act, changed=changed)
    if use_claim:
        offset = len(prefix.encode())
        for span in row['copies']:
            span['start'] += offset
            span['end'] += offset
    else: row['copies'] = []
    return row


def build_rows(bases, rules, seed, repeats=1, paraphrase=False):
    rng = random.Random(seed)
    pools = {}
    for row in bases:
        for f in row['fields']:
            if f.get('spoken') and f['knowledge'] != 3: pools.setdefault(f['role'], set()).add(f['text'])
    pools = {role: sorted(values) for role, values in pools.items()}
    def replacement(field):
        options = [x for x in pools[field['role']] if x != field['text']]
        return rng.choice(options) if options else 'A different account'
    result = []
    for repetition in range(repeats):
        for i, base in enumerate(bases):
            # Adjacent source-value pairs receive the same response task.
            act = ACTS[(i // 2 + repetition * 4) % len(ACTS)]
            row = response(base, rules[base['rule']], act, rng, replacement, paraphrase)
            row['id'] += f':conversation:{repetition}'
            result.append(row)
    return result


def approved_response(row, rule):
    if row['act'] in ('start', 'question'): return accepted_forms(row, rule)
    if row['act'] == 'disagree':
        return {'I heard a different account. ' + form + ' How sure are you?' for form in accepted_forms(row, rule)}
    return {row['output']}
