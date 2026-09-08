"""Decode saved preferences through the frozen packet key."""
import argparse
import json
from pathlib import Path
from ab_review import rankings


def load_events(path):
    text = path.read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(data, dict) and isinstance(data.get('events'), list):
        return data['events']
    if isinstance(data, dict) and 'event_id' in data:
        return [data]
    raise ValueError('Expected exported picks JSON or collector JSONL')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--events',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    packet=json.loads((a.directory/'ab-packet.json').read_text())
    key=json.loads((a.directory/'blind-review-key.json').read_text())
    events=[event for path in a.events for event in load_events(path)]
    result=rankings(events,packet,key)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__ == '__main__':main()
