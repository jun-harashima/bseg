import unittest
from bseg.dataset.one_input_dataset import OneInputDataset


class TestOneInputDataset(unittest.TestCase):

    def setUp(self):
        self.examples = [(("人参", "を", "切る"), ("名詞", "助詞", "動詞")),
                         (("葱", "を", "切る"), ("名詞", "助詞", "動詞"))]
        self.dataset = OneInputDataset(self.examples)

    def test__make_index(self):
        word_to_index, tag_to_index = self.dataset._make_index(self.examples)
        self.assertEqual(word_to_index,
                         {"<PAD>": 0, "<UNK>": 1, "人参": 2, "を": 3,
                          "切る": 4, "葱": 5})
        self.assertEqual(tag_to_index,
                         {"<PAD>": 0, "<UNK>": 1, "名詞": 2, "助詞": 3,
                          "動詞": 4})

    def test__degitize(self):
        word_to_index, tag_to_index = self.dataset._make_index(self.examples)
        X, Y = self.dataset._degitize(self.examples)
        self.assertTrue(X[0], [2, 3, 4])
        self.assertTrue(Y[0], [2, 3, 4])
        self.assertTrue(X[1], [5, 3, 4])
        self.assertTrue(Y[1], [2, 3, 4])


if __name__ == "__main__":
    unittest.main()
