import unittest
from bseg.morphology import Morphology


class TestMorphology(unittest.TestCase):

    def test___init__(self):
        line = "今日	名詞,副詞可能,*,*,*,*,今日,キョウ,キョー"
        morp = Morphology(line)
        self.assertEqual(morp.surface, "今日")
        self.assertEqual(morp.part_of_speech, "名詞")
        self.assertEqual(morp.part_of_speech1, "副詞可能")


if __name__ == "__main__":
    unittest.main()
