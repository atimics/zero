import copy
import random
import unittest
from crownless_conversation import response
from crownless_performance import CREATURES, EMOTIONS, control_text, styled, score, corpus
from crownless_v2 import encode_row
import test_crownless_conversation as fixtures


class PerformanceTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.ConversationTests(); fixture.setUp()
        self.row, self.rule, self.tokenizer = fixture.row, fixture.rule, fixture.tokenizer
        self.row.update(pair='test:0', rule='notice', kind='NOTICE_POSTED')

    def test_all_nine_controls_preserve_copies_and_knowledge(self):
        ordinary = response(self.row, self.rule, 'disagree', random.Random(1))
        inputs = set()
        for creature in CREATURES:
            for emotion in EMOTIONS:
                row = styled(ordinary, creature, emotion)
                self.assertEqual(row['fields'], ordinary['fields'])
                self.assertEqual(row['confidence'], ordinary['confidence'])
                self.assertEqual(row['history'], ordinary['history'])
                for span in row['copies']:
                    self.assertEqual(row['output'].encode()[span['start']:span['end']].decode(), span['text'])
                record = encode_row(self.tokenizer, row, slots=True, conversation=True)
                inputs.add(tuple(record['tokens'][:record['prefix_length']]))
                self.assertTrue(score(row, self.rule, {'text':row['output'], 'stopped':True})['joint'])
        self.assertEqual(len(inputs), 9)

    def test_style_meaning_and_completion_score_independently(self):
        original = response(self.row, self.rule, 'start', random.Random(1))
        row = styled(original, 'goblin', 'afraid')
        wrong = styled(original, 'pony', 'calm')
        result = score(row, self.rule, {'text':wrong['output'], 'stopped':True})
        self.assertTrue(result['meaning']); self.assertFalse(result['identity']); self.assertFalse(result['emotion'])
        corrupt = row['output'].replace('Newhaven', 'Farhaven')
        result = score(row, self.rule, {'text':corrupt, 'stopped':True})
        self.assertFalse(result['meaning']); self.assertTrue(result['identity']); self.assertTrue(result['emotion'])
        self.assertFalse(score(row, self.rule, {'text':row['output'], 'stopped':False})['joint'])

    def test_controls_validate_and_score_labels_stay_out_of_input(self):
        for bad in ({'creature':'goblin'}, {'creature':'dragon','emotion':'calm'},
                    {'creature':'pony','emotion':'afraid','confidence':0}, 'goblin'):
            with self.assertRaises(ValueError): control_text(bad)
        row = styled(response(self.row, self.rule, 'start', random.Random(1)), 'pony', 'afraid')
        first = encode_row(self.tokenizer, row, slots=True, conversation=True)
        changed = copy.deepcopy(row); changed['act'] = 'certainty'; changed['plain_output'] = 'Label only'
        second = encode_row(self.tokenizer, changed, slots=True, conversation=True)
        self.assertEqual(first['tokens'], second['tokens'])

    def test_held_rules_and_reproducibility(self):
        rules = {'notice':self.rule}
        one = corpus([self.row], rules, 1, 2)
        self.assertEqual(one, corpus([self.row], rules, 1, 2))
        self.assertEqual(len(one), 9)
        self.assertEqual(corpus([self.row], rules, 1, 2, held_rules=['notice']), [])

    def test_fear_and_confidence_are_separate_inputs(self):
        ordinary = response(self.row, self.rule, 'start', random.Random(1))
        afraid = styled(ordinary, 'goblin', 'afraid')
        calm = styled(ordinary, 'goblin', 'calm')
        unsure = copy.deepcopy(afraid); unsure['confidence'] = 20
        def prefix(row):
            r = encode_row(self.tokenizer, row, slots=True, conversation=True)
            return self.tokenizer.decode(r['tokens'][:r['prefix_length']])
        self.assertIn('feeling: afraid', prefix(afraid))
        self.assertIn('feeling: calm', prefix(calm))
        self.assertIn('feeling: afraid', prefix(unsure))
        self.assertIn('? ', prefix(unsure))
        self.assertNotIn('? ', prefix(afraid))


if __name__ == '__main__': unittest.main()
