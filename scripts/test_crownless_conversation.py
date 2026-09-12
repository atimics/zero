import copy
import random
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from tokenizers import Tokenizer
from crownless_conversation import response, ACTS
from crownless_v2 import encode_row
from chat_crownless import chat


class ConversationTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'models/crownless-core-v2/tokenizer.json'
        self.tokenizer = Tokenizer.from_file(str(path))
        source = '- Éva posts a notice at Newhaven: Flood relief.\n'
        target = 'Éva posted a notice about Flood relief in Newhaven.'
        fields, copies = [], []
        for slot, (text, role) in enumerate([('Éva', 1), ('Newhaven', 3), ('Flood relief', 4)]):
            def span(s):
                start = len(s[:s.index(text)].encode())
                return {'start': start, 'end': start + len(text.encode()), 'text': text, 'field': slot, 'role': role}
            fields.append(span(source) | {'knowledge': 0, 'spoken': True, 'provenance': 3, 'event': 1})
            copies.append(span(target) | {'spoken': True})
        self.row = {'id': 'test', 'prefix': source, 'output': target, 'fields': fields,
                    'copies': copies, 'confidence': 80, 'retold': False, 'kind_id': 1}
        self.rule = {'roles': ['actor','place','object'], 'outputs': ['{0} posted a notice about {2} in {1}.']}

    def test_history_reaches_model_and_labels_stay_out(self):
        a = response(self.row, self.rule, 'agree', random.Random(1))
        b = response(self.row, self.rule, 'disagree', random.Random(1))
        one = encode_row(self.tokenizer, a, slots=True, conversation=True)
        two = encode_row(self.tokenizer, b, slots=True, conversation=True)
        self.assertNotEqual(one['tokens'][:one['prefix_length']], two['tokens'][:two['prefix_length']])
        changed = copy.deepcopy(a); changed['act'] = 'disagree'; changed['changed'] = {'invented': True}
        self.assertEqual(one['tokens'], encode_row(self.tokenizer, changed, slots=True, conversation=True)['tokens'])
        self.assertEqual(encode_row(self.tokenizer, self.row, slots=True, packet=True)['tokens'],
                         encode_row(self.tokenizer, self.row, slots=True, conversation=True)['tokens'])

    def test_response_spans_and_bounded_history(self):
        for act in ACTS:
            row = response(self.row, self.rule, act, random.Random(2))
            for span in row['copies']:
                self.assertEqual(row['output'].encode()[span['start']:span['end']].decode(), span['text'])
            encode_row(self.tokenizer, row, slots=True, conversation=True)
        bad = copy.deepcopy(self.row)
        bad['history'] = [{'speaker': 'other', 'text': 'x ' * 1000}]
        with self.assertRaisesRegex(ValueError, 'budget'):
            encode_row(self.tokenizer, bad, slots=True, conversation=True)

    def test_generated_speech_is_the_next_event(self):
        fields = copy.deepcopy(self.row['fields'])
        for f in fields: f['start'] -= 2; f['end'] -= 2
        packet = {'text': self.row['prefix'][2:-1], 'kind': 133, 'confidence': 80,
                  'rule': 'notice', 'fields': fields}
        avatars = [{'name': 'Éva'}, {'name': 'Mara'}]
        outputs = ['What happened?', 'There was a notice.', 'Who told you?']
        def generated(model, tokenizer, record):
            text = outputs.pop(0)
            return {'text': text, 'stopped': True, 'actions': []}
        with patch('chat_crownless.generate', side_effect=generated):
            turns = chat(SimpleNamespace(mode='conversation'), {'meaning_ids': {'notice': 1}},
                         self.tokenizer, avatars, [packet, packet], turns=3)
        self.assertEqual(turns[1]['history'][-1], {'speaker': 'other', 'text': turns[0]['text']})
        self.assertEqual(turns[2]['history'][-1], {'speaker': 'other', 'text': turns[1]['text']})
        self.assertEqual(turns[2]['history'][0]['speaker'], 'self')


if __name__ == '__main__': unittest.main()
