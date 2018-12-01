import unittest
import torch
from bseg.pos_tagging_dataset import PosTaggingDataset


class TestPosTaggingDataset(unittest.TestCase):

    def setUp(self):
        self.examples = [(("人参", "を", "切る"), ("名詞", "助詞", "動詞")),
                         (("大根", "も", "切る"), ("名詞", "助詞", "動詞"))]
        self.dataset = PosTaggingDataset(self.examples)

    def test__make_index(self):
        word_to_index, tag_to_index = self.dataset._make_index(self.examples)
        self.assertEqual(word_to_index,
                         {"人参": 0, "を": 1, "切る": 2, "大根": 3, "も": 4})
        self.assertEqual(tag_to_index,
                         {"BOS": 0, "EOS": 1, "名詞": 2, "助詞": 3, "動詞": 4})

    def test__degitize_all(self):
        word_to_index, tag_to_index = self.dataset._make_index(self.examples)
        degitized_examples = self.dataset._degitize_all(self.examples)
        self.assertTrue(torch.equal(degitized_examples[0][0],
                                    torch.tensor([0, 1, 2], dtype=torch.long)))
        self.assertTrue(torch.equal(degitized_examples[0][1],
                                    torch.tensor([2, 3, 4], dtype=torch.long)))
        self.assertTrue(torch.equal(degitized_examples[1][0],
                                    torch.tensor([3, 4, 2], dtype=torch.long)))
        self.assertTrue(torch.equal(degitized_examples[1][1],
                                    torch.tensor([2, 3, 4], dtype=torch.long)))


if __name__ == "__main__":
    unittest.main()
