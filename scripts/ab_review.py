"""Stable A/B packets, preference validation, and descriptive rankings."""
import hashlib
import json
from collections import Counter


def build_packet(samples, key):
    lookup = {(r['prompt'], r['seed'], r['repetition_penalty']): r for r in samples
              if r['temperature'] == .7 and r['top_k'] == 40}
    cases = []
    for row in key:
        cases.append({'case_id': row['case'], 'prompt': row['prompt'],
                      'A': lookup[row['prompt'], row['seed'], row['A_penalty']]['output'],
                      'B': lookup[row['prompt'], row['seed'], row['B_penalty']]['output']})
    digest = hashlib.sha256(json.dumps(cases, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return {'schema_version': 1, 'packet_id': 'zero-ab-'+digest[:16], 'cases': cases}


def validate_vote(vote, packet):
    if not isinstance(vote, dict):
        raise ValueError('Expected a vote object')
    required = ['packet_id', 'reviewer_id', 'event_id', 'case_id', 'choice']
    if any(k not in vote for k in required):
        raise ValueError('Missing vote fields')
    if vote['packet_id'] != packet['packet_id']:
        raise ValueError('This vote belongs to another review packet')
    for field in ['reviewer_id', 'event_id']:
        value = vote[field]
        if not isinstance(value, str) or not 1 <= len(value) <= 80 or not all(c.isascii() and (c.isalnum() or c in '-_') for c in value):
            raise ValueError('Invalid '+field)
    if type(vote['case_id']) is not int or vote['case_id'] not in {r['case_id'] for r in packet['cases']}:
        raise ValueError('Unknown case')
    if vote['choice'] not in ['A', 'B', 'skip', 'withdraw']:
        raise ValueError('Choose A, B, or skip')
    return {k: vote[k] for k in required}


def effective_votes(events):
    """Arrival order is authoritative; retries of the same event count once."""
    seen = {}; latest = {}
    for event in events:
        identity = event['event_id']
        if identity in seen:
            if seen[identity] != event:
                raise ValueError('An event ID was reused with different content')
            continue
        seen[identity] = event
        key = (event['packet_id'], event['reviewer_id'], event['case_id'])
        if event['choice'] == 'withdraw':
            latest.pop(key, None)
        else:
            latest[key] = event
    return list(latest.values())


def rankings(events, packet, key):
    events = [validate_vote(e, packet) for e in events]
    votes = effective_votes(events)
    mapping = {r['case']: r for r in key}
    wins = Counter({'1.0': 0, '1.1': 0}); letters = Counter({'A': 0, 'B': 0})
    skipped = 0
    for vote in votes:
        if vote['choice'] == 'skip':
            skipped += 1
        else:
            letters[vote['choice']] += 1
            wins[str(mapping[vote['case_id']][vote['choice']+'_penalty'])] += 1
    return {'packet_id': packet['packet_id'], 'reviewers': len({v['reviewer_id'] for v in votes}),
            'choices': sum(wins.values()), 'skips': skipped, 'letter_picks': dict(letters),
            'wins_by_repetition_penalty': dict(wins),
            'scope': 'Descriptive preference counts. Skips are excluded from wins. Repeated events and revised answers count once per reviewer/case. Partial voluntary reviews can be selective; these counts alone do not establish a population preference.'}
