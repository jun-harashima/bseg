import unittest
from bseg.bseg import Bseg


class TestBseg(unittest.TestCase):

    def test_segment(self):
        analysis_result = """
        今日	名詞,副詞可能,*,*,*,*,今日,キョウ,キョー
        は	助詞,係助詞,*,*,*,*,は,ハ,ワ
        天気	名詞,一般,*,*,*,*,天気,テンキ,テンキ
        が	助詞,格助詞,一般,*,*,*,が,ガ,ガ
        良い	形容詞,自立,*,*,形容詞・アウオ段,基本形,良い,ヨイ,ヨイ
        。	記号,句点,*,*,*,*,。,。,。
        EOS
        """

        bseg = Bseg()
        bnsts = bseg.segment(analysis_result)
        self.assertEqual(bnsts, [])
