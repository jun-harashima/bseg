import unittest
from bseg.dataset.one_input_dataset import OneInputDataset


class TestOneInputDataset(unittest.TestCase):

    def setUp(self):
        self.examples = [(('名詞', '助詞', '動詞'), ('人参', 'を', '切る')),
                         (('名詞', '助詞', '動詞'), ('葱', 'を', '切る'))]
        self.dataset = OneInputDataset(self.examples)

    def test__make_index(self):
        tags = [tag for example in self.examples for tag in example[0]]
        words = [word for example in self.examples for word in example[1]]
        tag_to_index = self.dataset._make_index(tags)
        word_to_index = self.dataset._make_index(words)
        self.assertEqual(tag_to_index,
                         {'<PAD>': 0, '<UNK>': 1, '名詞': 2, '助詞': 3,
                          '動詞': 4})
        self.assertEqual(word_to_index,
                         {'<PAD>': 0, '<UNK>': 1, '人参': 2, 'を': 3,
                          '切る': 4, '葱': 5})

    def test__degitize(self):
        tags = [tag for example in self.examples for tag in example[0]]
        words = [word for example in self.examples for word in example[1]]
        self.dataset.tag_to_index = self.dataset._make_index(tags)
        self.dataset.word_to_index = self.dataset._make_index(words)
        X, Y = self.dataset._degitize(self.examples)
        self.assertTrue(X[0], [2, 3, 4])
        self.assertTrue(Y[0], [2, 3, 4])
        self.assertTrue(X[1], [5, 3, 4])
        self.assertTrue(Y[1], [2, 3, 4])


if __name__ == '__main__':
    unittest.main()
