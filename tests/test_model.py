import unittest
from unittest.mock import patch
import torch
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import PackedSequence
from bseg.model import Model
from bseg.dataset import Dataset


class TestModel(unittest.TestCase):

    def setUp(self):
        self.model = Model(2, 4, 9, 6, 3)  # 9 = 8 + 1, 6 = 5 + 1 (for <PAD>)
        self.embeddings_weight = Parameter(torch.tensor([[0, 0],  # for <PAD>
                                                         [1, 2],
                                                         [3, 4],
                                                         [5, 6],
                                                         [7, 8],
                                                         [9, 10],
                                                         [11, 12],
                                                         [13, 14],
                                                         [15, 16]],
                                                        dtype=torch.float))
        self.X1 = ([1, 2, 3], [4, 3], [5, 6, 7, 8])
        self.X2 = [[5, 6, 7, 8], [1, 2, 3], [4, 3]]
        self.X3 = [[5, 6, 7, 8], [1, 2, 3, 0], [4, 3, 0, 0]]
        self.X4 = torch.tensor([[[9, 10], [11, 12], [13, 14], [15, 16]],
                                [[1, 2], [3, 4], [5, 6], [0, 0]],
                                [[7, 8], [5, 6], [0, 0], [0, 0]]],
                               dtype=torch.float)
        self.X5 = PackedSequence(torch.tensor([[9, 10],
                                               [1, 2],
                                               [7, 8],
                                               [11, 12],
                                               [3, 4],
                                               [5, 6],
                                               [13, 14],
                                               [5, 6],
                                               [15, 16]],
                                              dtype=torch.float),
                                 torch.tensor([3, 3, 2, 1]))
        self.Y = ([1, 2, 3], [4, 3], [1, 2, 5, 3])
        self.lengths = [len(x) for x in self.X2]

    def test___init__(self):
        zeros = torch.zeros(1, 3, 4)
        self.assertTrue(torch.equal(self.model.hidden[0], zeros))
        self.assertTrue(torch.equal(self.model.hidden[1], zeros))

    def test__split(self):
        examples = [
            (("人参", "を", "切る"), ("名詞", "助詞", "動詞")),
            (("ざっくり", "切る"), ("副詞", "動詞")),
            (("葱", "は", "細く", "刻む"), ("名詞", "助詞", "形容詞", "動詞"))
        ]
        dataset = Dataset(examples)
        batches = self.model._split(dataset)
        self.assertEqual(batches[0], (self.X1, self.Y))

    def test__sort(self):
        X2 = self.model._sort(self.X1)
        self.assertEqual(X2, self.X2)

    def test__pad(self):
        X3 = self.model._pad(self.X2, self.lengths, 0)
        self.assertEqual(X3, self.X3)

    def test__embed(self):
        with patch.object(self.model.embeddings, 'weight',
                          self.embeddings_weight):
            X4 = self.model._embed(torch.tensor(self.X3), 0)
            self.assertTrue(torch.equal(X4, self.X4))

    def test__pack(self):
        X5 = self.model._pack(self.X4, self.lengths)
        self.assertTrue(torch.equal(X5.data, self.X5.data))
        self.assertTrue(torch.equal(X5.batch_sizes, self.X5.batch_sizes))

    # def test__unpack(self):
    #     X6 = self.model._unpack(self.X5)
    #     print(X6)
    #     self.assertTrue(torch.equal(X6[0], torch.tensor(self.X3)))
    #     self.assertTrue(torch.equal(X6[1], torch.tensor(self.lengths)))


if __name__ == "__main__":
    unittest.main()
