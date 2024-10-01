import unittest
from bpe_comparisons import compare_bpe_methods
from data import load_raw_text


class TestBPEComparisons(unittest.TestCase):

    def setUp(self):
        self.raw_text = load_raw_text('../01_main-chapter-code/the-verdict.txt')

    def test_compare_bpe_methods(self):
        results = compare_bpe_methods(self.raw_text)
        self.assertIn('tiktoken', results)
        self.assertIn('original_bpe', results)
        self.assertIn('hugging_face', results)
        # Further assertions can be added based on expected output


if __name__ == '__main__':
    unittest.main()
