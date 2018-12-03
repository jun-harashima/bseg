import unittest
import torch
from bseg.model import Model
from bseg.dataset import Dataset


class TestModel(unittest.TestCase):

    def test___init__(self):
        model = Model(200, 100, 10000, 10)
        expected_hidden = (torch.zeros(1, 1, 100), torch.zeros(1, 1, 100))
        self.assertTrue(model.hidden, expected_hidden)

    def test__split(self):
        examples = [(("人参", "を", "切る"), ("名詞", "助詞", "動詞")),
                    (("大根", "を", "切る"), ("名詞", "助詞", "動詞")),
                    (("白菜", "は", "蒸す"), ("名詞", "助詞", "動詞")),
                    (("牛蒡", "は", "削ぐ"), ("名詞", "助詞", "動詞"))]
        dataset = Dataset(examples)
        model = Model(200, 100, 10000, 10, 2)
        batch = list(model._split(dataset))[0]
        expected_batch = (([1, 2, 3], [4, 2, 3]), ([1, 2, 3], [1, 2, 3]))
        self.assertEqual(batch, expected_batch)


if __name__ == "__main__":
    unittest.main()
