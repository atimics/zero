import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from canada_narrative import checked, encode_record, read_json, EXPERIMENT
from run_canada_narrative import windows, update, evaluate, learning_rate, state_digest, train
import run_canada_narrative as runner
from review_canada_narrative import tally, repeat_rate, validate_pair
from subword_model import create


class CanadaTests(unittest.TestCase):
    def test_exact_tail_and_wrap_targets(self):
        data = np.arange(11, dtype=np.uint16)
        batches = list(windows(data, 8, 13, 4))
        scored = np.concatenate([y[:n] for x, y, n in batches])
        np.testing.assert_array_equal(scored, np.arange(8, 21) % 11)
        self.assertEqual(sum(n for _, _, n in batches), 13)
        self.assertEqual(batches[-1][1].tolist()[1:], [-100] * 3)
        for x, y, n in batches:
            np.testing.assert_array_equal((x[:n] + 1) % 11, y[:n])

    def test_fixed_budgets(self):
        c = read_json(EXPERIMENT / 'contract.json')
        self.assertEqual(c['comparisons']['AB']['target_tokens_per_arm'], 27157191)
        self.assertEqual(c['comparisons']['BC']['target_tokens_per_arm'], 100499909)
        self.assertEqual(c['seeds'], [7, 17, 29])

    def test_pair_initialization(self):
        config = dict(vocab=32, context=4, dim=8, heads=2, layers=1, ff=16)
        self.assertEqual(state_digest(create(config, 7)), state_digest(create(config, 7)))
        self.assertNotEqual(state_digest(create(config, 7)), state_digest(create(config, 17)))

    def test_tail_backward_is_finite(self):
        torch.set_num_threads(1)
        model = create(dict(vocab=32, context=4, dim=8, heads=2, layers=1, ff=16), 7)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        before = state_digest(model)
        loss = update(model, optimizer, np.arange(19, dtype=np.uint16), 17, 5, .001, 'cpu')
        self.assertTrue(np.isfinite(loss))
        self.assertNotEqual(before, state_digest(model))

    def test_larger_microbatch_preserves_masked_gradient(self):
        config = dict(vocab=32, context=4, dim=8, heads=2, layers=1, ff=16)
        models = [create(config, 7), create(config, 7)]
        data = np.arange(19, dtype=np.uint16)
        losses = []
        for model, micro in zip(models, [1, 4]):
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.)
            losses.append(update(model, optimizer, data, 17, 13, 0., 'cpu', micro, dropout=0.))
        self.assertAlmostEqual(losses[0], losses[1], places=6)
        for first, second in zip(models[0].parameters(), models[1].parameters()):
            torch.testing.assert_close(first.grad, second.grad, atol=1e-6, rtol=1e-5)

    def test_shared_scoring_and_bounds(self):
        model = create(dict(vocab=32, context=1024, dim=8, heads=2, layers=1, ff=16), 7)
        data = np.arange(4096, dtype=np.uint16) % 32
        score = evaluate(model, data, [1024], np.ones(32), 'cpu')
        self.assertEqual(score['tokens'], 256)
        self.assertEqual(score['bytes'], 256)
        self.assertTrue(np.isfinite(score['bits_per_byte']))
        with self.assertRaises(ValueError):
            evaluate(model, data, [0], np.ones(32), 'cpu')

    def test_hash_and_path_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'x').write_bytes(b'accepted')
            sha = hashlib.sha256(b'accepted').hexdigest()
            self.assertEqual(checked(root, 'x', sha), (root / 'x').resolve())
            with self.assertRaises(ValueError):
                checked(root, '../x', sha)
            (root / 'x').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                checked(root, 'x', sha)

    def test_encoding_line_boundaries_and_counts(self):
        class Tokenizer:
            def encode(self, text):
                return type('Encoding', (), {'ids': list(text.encode())})()
            def decode(self, ids):
                return bytes(ids).decode()
        text = 'First.\n\nLast.'
        row = {'text': text, 'contentHash': hashlib.sha256(text.encode()).hexdigest(),
               'metadata': {'exactTokens': len(text)}}
        self.assertEqual(encode_record(Tokenizer(), row), list(text.encode()))
        row['metadata']['exactTokens'] += 1
        with self.assertRaises(ValueError):
            encode_record(Tokenizer(), row)

    def review_fixture(self):
        packet = {'packet_id': 'test', 'cases': [{'case_id': i} for i in range(1, 201)]}
        key = [{'case_id': i, 'candidate': 'B'} for i in range(1, 201)]
        events = [{'packet_id': 'test', 'case_id': i, 'reviewer_id': 'reader',
                   'event_id': str(i), 'choice': 'B' if i <= 120 else 'A'} for i in range(1, 201)]
        return packet, key, events

    def test_complete_review_and_revisions(self):
        packet, key, events = self.review_fixture()
        self.assertTrue(tally(events + events, packet, key, 'reader')['human_gate_pass'])
        events.append({**events[0], 'event_id': 'revision', 'choice': 'A'})
        self.assertFalse(tally(events, packet, key, 'reader')['human_gate_pass'])
        self.assertEqual(tally(events, packet, key, 'reader')['candidate_wins'], 119)

    def test_partial_review_and_skip_gate(self):
        packet, key, events = self.review_fixture()
        self.assertFalse(tally(events[:120], packet, key, 'reader')['human_gate_pass'])
        for event in events[120:]:
            event['choice'] = 'skip'
        self.assertFalse(tally(events, packet, key, 'reader')['human_gate_pass'])
        self.assertEqual(tally(events, packet, key, 'reader')['candidate_share_all_200'], .6)

    def test_repetition_and_schedule(self):
        self.assertEqual(repeat_rate('one two three four'), 0)
        self.assertGreater(repeat_rate('one two three four one two three four'), 0)
        self.assertGreater(learning_rate(1, 1000), 0)
        self.assertEqual(learning_rate(1000, 1000), 0)

    def test_invalid_pair_rejected(self):
        with self.assertRaises((ValueError, KeyError)):
            validate_pair({'status': 'incomplete'}, {}, 'AB', 'x')

    def test_paired_budget_and_seed_identity(self):
        from canada_narrative import digest
        control = {'status': 'trained', 'comparison': 'AB', 'arm': 'A', 'seed': 7,
                   'target_presentations': 27157191, 'manifest_sha256': 'manifest',
                   'contract_sha256': digest(EXPERIMENT / 'contract.json'),
                   'implementation_sha256': digest(EXPERIMENT / 'implementation.lock.json'),
                   'initial_weights_sha256': 'weights', 'config': 'same',
                   'parameters': 5049600, 'device': 'cpu', 'platform': 'same',
                   'cpu_threads': 8, 'microbatch_sequences': 4}
        candidate = {**control, 'arm': 'B'}
        validate_pair(control, candidate, 'AB', 'manifest')
        with self.assertRaises(ValueError):
            validate_pair(control, {**candidate, 'seed': 17}, 'AB', 'manifest')
        with self.assertRaises(ValueError):
            validate_pair(control, {**candidate, 'target_presentations': 27157192}, 'AB', 'manifest')

    def test_replication_requires_matching_passing_pilot(self):
        from canada_narrative import digest
        runner.validate_seed(7, 'AB', 'manifest')
        with self.assertRaises(ValueError):
            runner.validate_seed(17, 'AB', 'manifest')
        decision = {'pilot_pass': True, 'comparison': 'AB', 'seed': 7,
                    'manifest_sha256': 'manifest', 'contract_sha256': digest(EXPERIMENT / 'contract.json')}
        runner.validate_seed(17, 'AB', 'manifest', decision)
        with self.assertRaises(ValueError):
            runner.validate_seed(29, 'BC', 'manifest', decision)
        decision['pilot_pass'] = False
        with self.assertRaises(ValueError):
            runner.validate_seed(17, 'AB', 'manifest', decision)

    def test_frozen_implementation_and_versions(self):
        runner.verify_code()

    def test_training_completion_and_saved_checkpoint(self):
        from canada_narrative import write_json, digest
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); inputs = root / 'inputs'; inputs.mkdir()
            for name in ['A.bin', 'validation.bin']:
                (np.arange(4096, dtype=np.uint16) % 32).tofile(inputs / name)
            write_json(inputs / 'manifest.json', {})
            write_json(inputs / 'evaluation.json', {'selection': [1024]})
            write_json(inputs / 'token_bytes.json', [1] * 32)
            contract = read_json(EXPERIMENT / 'contract.json')
            contract['comparisons']['AB']['target_tokens_per_arm'] = 13
            write_json(root / 'contract.json', contract)
            write_json(root / 'implementation.lock.json', {})
            write_json(root / 'preparation.json', {'manifest_sha256': digest(inputs / 'manifest.json')})
            config = dict(vocab=32, context=1024, dim=8, heads=2, layers=1, ff=16)
            def tiny_setup(seed, device):
                model = create(config, seed)
                return model, torch.optim.AdamW(model.parameters(), lr=.0003)
            with patch.object(runner, 'EXPERIMENT', root), patch.object(runner, 'setup', tiny_setup), \
                    patch.object(runner, 'verify_prepared', return_value={}), \
                    patch.dict(runner.CONFIGS, {'5m-1024': config}):
                train(SimpleNamespace(data=inputs, output=root / 'run', comparison='AB', arm='A', device='cpu'))
            result = read_json(root / 'run/result.json')
            self.assertEqual(result['target_presentations'], 13)
            self.assertEqual(result['selected_step'], 1)
            state = torch.load(root / 'run/best.pt', weights_only=True)
            self.assertEqual(state['target_presentations'], 13)
            self.assertEqual(state['config'], config)
            self.assertEqual(result['best_sha256'], digest(root / 'run/best.pt'))


if __name__ == '__main__':
    unittest.main()
