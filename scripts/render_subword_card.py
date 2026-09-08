"""Render the model card for the static comparison demo (Markdown 3.8.2)."""
from pathlib import Path
import markdown
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'docs/subword'
body=markdown.markdown((p/'MODEL_CARD.md').read_text(),extensions=['tables','fenced_code'])
(p/'model-card.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ZERO · Model card</title><link rel="stylesheet" href="style.css"><header><a href="index.html">← Model comparison</a><a href="MODEL_CARD.md">Download Markdown</a></header><main class="card">'+body+'</main></html>')
