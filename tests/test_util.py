import unittest
import torch
from bseg.util import Util


class TestBunsetsu(unittest.TestCase):

    def setUp(self):
        self.raw_training_data = [(["彼", "は", "京都", "に", "行っ", "た"],
                                   ["O", "O", "B", "O", "O", "O"])]

    def test_make_index_from(self):
        util = Util()
        word_to_index = util.make_index_from(self.raw_training_data)
        self.assertEqual(word_to_index["彼"], 0)

    def test_degitize(self):
        util = Util()
        words = self.raw_training_data[0][0]
        word_to_index = util.make_index_from(self.raw_training_data)
        tensor = util.degitize(words, word_to_index)
        expected_tensor = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
        self.assertTrue(torch.equal(tensor, expected_tensor))


if __name__ == "__main__":
    unittest.main()
