"""Build a one-pair-at-a-time review while preserving the original A/B map."""
import argparse
import json
from pathlib import Path
from ab_review import build_packet


def write_review(directory, rows, key):
    packet = build_packet(rows, key)
    (directory/'ab-packet.json').write_text(json.dumps(packet, indent=2)+'\n')
    template = Path(__file__).with_name('ab_review.html').read_text()
    payload = json.dumps(packet).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    page = template.replace('/*PACKET*/null', payload)
    (directory/'ab-review.html').write_text(page)
    (directory/'blind-review.html').write_text(page)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--directory', type=Path, required=True); a = p.parse_args()
    rows = [json.loads(line) for line in (a.directory/'samples.jsonl').read_text().splitlines()]
    key = json.loads((a.directory/'blind-review-key.json').read_text())
    write_review(a.directory, rows, key)

if __name__ == '__main__': main()
