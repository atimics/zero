import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from ab_review import build_packet, rankings, effective_votes
from serve_ab_review import VoteStore
from rank_ab_reviews import load_events
ROOT=Path(__file__).resolve().parents[1]

class ABReview(unittest.TestCase):
    def setUp(self):
        self.packet={'packet_id':'packet-1','cases':[{'case_id':1},{'case_id':2}]}
        self.first=dict(packet_id='packet-1',reviewer_id='r1',event_id='e1',case_id=1,choice='A')
    def test_retry_and_conflicting_event(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'votes.jsonl';store=VoteStore(path,self.packet)
            self.assertTrue(store.append(self.first));self.assertFalse(store.append(self.first))
            with self.assertRaises(ValueError):store.append({**self.first,'choice':'B'})
            self.assertEqual(len(path.read_text().splitlines()),1)
            self.assertFalse(VoteStore(path,self.packet).append(self.first))
    def test_undo_and_revision(self):
        undo={**self.first,'event_id':'e2','choice':'withdraw'}
        self.assertEqual(effective_votes([self.first,self.first,undo]),[])
        revised={**self.first,'event_id':'e3','choice':'B'}
        self.assertEqual(effective_votes([self.first,undo,revised]),[revised])
    def test_packet_validation(self):
        with tempfile.TemporaryDirectory() as d:
            store=VoteStore(Path(d)/'votes.jsonl',self.packet)
            for patch in [{'packet_id':'other'},{'case_id':3},{'case_id':True},{'choice':'tie'}]:
                with self.assertRaises(ValueError):store.append({**self.first,**patch})
    def test_collector_and_export_merge_without_duplicate_votes(self):
        with tempfile.TemporaryDirectory() as d:
            line=Path(d)/'events.jsonl'; export=Path(d)/'export.json'
            line.write_text(json.dumps(self.first)+'\n')
            export.write_text(json.dumps({'events':[self.first]}))
            merged=load_events(line)+load_events(export)
            self.assertEqual(effective_votes(merged),[self.first])

    def test_original_picks_keep_their_mapping(self):
        directory=ROOT/'experiments/inference-decoding/sweep'
        rows=[json.loads(line) for line in (directory/'samples.jsonl').read_text().splitlines()]
        key=json.loads((directory/'blind-review-key.json').read_text());packet=build_packet(rows,key)
        self.assertEqual(packet['packet_id'],'zero-ab-b6f83e8ed0c2b56c')
        events=[json.loads(line) for line in (ROOT/'experiments/inference-decoding/reviews/review-001-events.jsonl').read_text().splitlines()]
        result=rankings(events,packet,key)
        self.assertEqual(result['wins_by_repetition_penalty'],{'1.0':7,'1.1':7})
        self.assertEqual(result['choices'],14);self.assertEqual(result['skips'],1)

if __name__=='__main__':unittest.main()
