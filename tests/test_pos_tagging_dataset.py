import unittest
import torch
from bseg.pos_tagging_dataset import PosTaggingDataset


class TestPosTaggingDataset(unittest.TestCase):

    def setUp(self):
        self.examples = [(("人参", "を", "切る"), ("名詞", "助詞", "動詞")),
                         (("大根", "を", "切る"), ("名詞", "助詞", "動詞"))]
        self.dataset = PosTaggingDataset(self.examples)

    def test__make_index(self):
        word_to_index, tag_to_index = self.dataset._make_index(self.examples)
        self.assertEqual(word_to_index,
                         {"PAD": 0, "人参": 1, "を": 2, "切る": 3, "大根": 4})
        self.assertEqual(tag_to_index,
                         {"PAD": 0, "名詞": 1, "助詞": 2, "動詞": 3})

    def test__degitize_all(self):
        word_to_index, tag_to_index = self.dataset._make_index(self.examples)
        degitized_examples = self.dataset._degitize_all(self.examples)
        self.assertTrue(torch.equal(degitized_examples[0][0],
                                    torch.tensor([1, 2, 3], dtype=torch.long)))
        self.assertTrue(torch.equal(degitized_examples[0][1],
                                    torch.tensor([1, 2, 3], dtype=torch.long)))
        self.assertTrue(torch.equal(degitized_examples[1][0],
                                    torch.tensor([4, 2, 3], dtype=torch.long)))
        self.assertTrue(torch.equal(degitized_examples[1][1],
                                    torch.tensor([1, 2, 3], dtype=torch.long)))


if __name__ == "__main__":
    unittest.main()
