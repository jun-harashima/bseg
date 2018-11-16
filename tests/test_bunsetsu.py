import unittest
import textwrap
from bseg.morphology import Morphology
from bseg.bunsetsu import Bunsetsu


class TestBunsetsu(unittest.TestCase):

    def test___init__(self):
        morp1 = Morphology("今日	名詞,副詞可能,*,*,*,*,今日,キョウ,キョー")
        morp2 = Morphology("は	助詞,係助詞,*,*,*,*,は,ハ,ワ")
        bnst = Bunsetsu([morp1, morp2])
        self.assertEqual(bnst.surface, "今日は")


if __name__ == "__main__":
    unittest.main()
