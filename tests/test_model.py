import unittest
import torch
from bseg.util import Util
from bseg.model import Model


class TestModel(unittest.TestCase):

    def setUp(self):
        self.training_data = [(["彼", "は", "京都", "に", "行っ", "た"],
                               ["O", "O", "B", "O", "O", "O"])]

    def test_prepare_sequence_for(self):
        util = Util()
        sentence = self.training_data[0][0]
        word_to_index = util.make_index_from(self.training_data)
        tag_to_idnex = {"B": 0, "I": 1, "O": 2, "<START>": 3, "<STOP>": 4}
        model = Model(10, tag_to_idnex, 10, 5)
        sequence = model.prepare_sequence_for(sentence, word_to_index)
        expected_sequence = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
        self.assertTrue(torch.equal(sequence, expected_sequence))


if __name__ == "__main__":
    unittest.main()
