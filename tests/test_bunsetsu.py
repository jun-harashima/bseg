import unittest
from bseg.morphology import Morphology
from bseg.bunsetsu import Bunsetsu


class TestBunsetsu(unittest.TestCase):

    def setUp(self):
        morp1 = Morphology("天気	名詞,一般,*,*,*,*,天気,テンキ,テンキ")
        morp2 = Morphology("が	助詞,格助詞,一般,*,*,*,が,ガ,ガ")
        self.bnst1 = Bunsetsu([morp1, morp2])
        morp3 = Morphology("良い	形容詞,自立,*,*,形容詞・アウオ段,\
                           基本形,良い,ヨイ,ヨイ")
        morp4 = Morphology("。	記号,句点,*,*,*,*,。,。,。")
        self.bnst2 = Bunsetsu([morp3, morp4])

    def test___init__(self):
        self.assertEqual(self.bnst1.surface, "天気が")

    def test_ispredicate(self):
        self.assertFalse(self.bnst1.ispredicate())
        self.assertTrue(self.bnst2.ispredicate())


if __name__ == "__main__":
    unittest.main()
