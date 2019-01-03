import unittest
from bseg.dataset.one_input_dataset import OneInputDataset


class TestOneInputDataset(unittest.TestCase):

    def test__init__(self):
        examples = [{'Xs': [['人参', 'を', '切る']],
                     'Y': ['名詞', '助詞', '動詞']},
                    {'Xs': [['葱', 'を', '切る']],
                     'Y': ['名詞', '助詞', '動詞']}]
        dataset = OneInputDataset(examples)
        self.assertEqual(dataset.x_to_index,
                         [{'<PAD>': 0, '<UNK>': 1, '人参': 2, 'を': 3,
                           '切る': 4, '葱': 5}])
        self.assertEqual(dataset.y_to_index,
                         {'<PAD>': 0, '<UNK>': 1, '名詞': 2, '助詞': 3,
                          '動詞': 4})
        self.assertTrue(dataset.Xs[0][0], [2, 3, 4])
        self.assertTrue(dataset.Xs[0][1], [5, 3, 4])
        self.assertTrue(dataset.Y[0], [2, 3, 4])
        self.assertTrue(dataset.Y[1], [2, 3, 4])


if __name__ == '__main__':
    unittest.main()
