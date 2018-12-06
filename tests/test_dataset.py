import unittest
from bseg.dataset import Dataset


class TestDataset(unittest.TestCase):

    def setUp(self):
        self.examples = [(("人参", "を", "切る"), ("名詞", "助詞", "動詞")),
                         (("葱", "を", "切る"), ("名詞", "助詞", "動詞"))]
        self.dataset = Dataset(self.examples)

    def test__make_index(self):
        word_to_index, tag_to_index = self.dataset._make_index(self.examples)
        self.assertEqual(word_to_index,
                         {"<PAD>": 0, "人参": 1, "を": 2, "切る": 3, "葱": 4})
        self.assertEqual(tag_to_index,
                         {"<PAD>": 0, "名詞": 1, "助詞": 2, "動詞": 3})

    def test__degitize(self):
        word_to_index, tag_to_index = self.dataset._make_index(self.examples)
        X, Y = self.dataset._degitize(self.examples)
        self.assertTrue(X[0], [1, 2, 3])
        self.assertTrue(Y[0], [1, 2, 3])
        self.assertTrue(X[1], [4, 2, 3])
        self.assertTrue(Y[1], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
