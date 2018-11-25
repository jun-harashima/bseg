import unittest
from bseg.util import Util


class TestBunsetsu(unittest.TestCase):

    def test_make_index_from(self):
        training_data = [(["彼", "は", "京都", "に", "行っ", "た"],
                          ["O", "O", "B", "O", "O", "O"])]
        util = Util()
        word_to_index = util.make_index_from(training_data)
        self.assertEqual(word_to_index["彼"], 0)


if __name__ == "__main__":
    unittest.main()
