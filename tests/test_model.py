import unittest
import torch
from bseg.model import Model
from bseg.dataset import Dataset


class TestModel(unittest.TestCase):

    def setUp(self):
        self.X1 = [[1, 2, 3], [1, 2], [1, 2, 3, 4]]
        self.X2 = [[1, 2, 3, 4], [1, 2, 3], [1, 2]]
        self.X3 = [[1, 2, 3, 4], [1, 2, 3, 0], [1, 2, 0, 0]]

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

    def test__sort(self):
        model = Model(200, 100, 10000, 10, 3)
        X2 = model._sort(self.X1)
        self.assertEqual(X2, self.X2)

    def test__pad(self):
        model = Model(200, 100, 10000, 10, 3)
        X3 = model._pad(self.X2, 0)
        self.assertEqual(X3, self.X3)

    def test__pack(self):
        model = Model(200, 100, 10000, 10, 3)
        X4 = model._pack(torch.tensor(self.X3))
        self.assertTrue(torch.equal(X4.data,
                                    torch.tensor([1, 1, 1, 2, 2, 2, 3, 3, 4])))
        self.assertTrue(torch.equal(X4.batch_sizes,
                                    torch.tensor([3, 3, 2, 1])))


if __name__ == "__main__":
    unittest.main()
