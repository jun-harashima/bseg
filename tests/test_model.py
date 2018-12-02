import unittest
import torch
from bseg.model import Model


class TestModel(unittest.TestCase):

    def test___init__(self):
        model = Model(200, 100, 10000, 10)
        expected_hidden = (torch.zeros(1, 1, 100), torch.zeros(1, 1, 100))
        self.assertTrue(model.hidden, expected_hidden)


if __name__ == "__main__":
    unittest.main()
