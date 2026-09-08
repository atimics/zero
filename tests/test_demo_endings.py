import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from serve_subword_demo import present_continuation, generation_options

class DemoEndings(unittest.TestCase):
    def test_trim_and_preserve_raw(self):
        prompt='She said '
        raw=prompt+'"Hello." Then she turn'
        self.assertEqual(present_continuation(prompt,raw,True),prompt+'"Hello."')
        self.assertEqual(present_continuation(prompt,raw,False),raw)
    def test_no_boundary_keeps_generated_text(self):
        self.assertEqual(present_continuation('Who? ', 'Who? A girl was', True), 'Who? A girl was')
    def test_bounded_options(self):
        self.assertEqual(generation_options({}), (128,False))
        for value in [True, 100000, '128']:
            with self.assertRaises(ValueError): generation_options({'count':value})

if __name__=='__main__':unittest.main()
