import unittest
import textwrap
from bseg.bseg import Bseg


class TestBseg(unittest.TestCase):

    def setUp(self):
        self.analysis_result = textwrap.dedent("""
        今日	名詞,副詞可能,*,*,*,*,今日,キョウ,キョー
        は	助詞,係助詞,*,*,*,*,は,ハ,ワ
        天気	名詞,一般,*,*,*,*,天気,テンキ,テンキ
        が	助詞,格助詞,一般,*,*,*,が,ガ,ガ
        良い	形容詞,自立,*,*,形容詞・アウオ段,基本形,良い,ヨイ,ヨイ
        。	記号,句点,*,*,*,*,。,。,。
        """)[1:-1]

    def test_segment(self):
        bseg = Bseg()
        bnsts = bseg.segment(self.analysis_result)
        self.assertEqual(bnsts[0].surface, "今日は")

    def test__construct_morphology_from(self):
        bseg = Bseg()
        morps = bseg._construct_morphology_from(self.analysis_result)
        self.assertEqual(morps[0].surface, "今日")
        self.assertEqual(morps[1].surface, "は")
        self.assertEqual(morps[2].surface, "天気")
        self.assertEqual(morps[3].surface, "が")
        self.assertEqual(morps[4].surface, "良い")
        self.assertEqual(morps[5].surface, "。")

    def test__construct_bunsetsu_from(self):
        bseg = Bseg()
        morps = bseg._construct_morphology_from(self.analysis_result)
        bnsts = bseg._construct_bunsetsu_from(morps)
        self.assertEqual(bnsts[0].surface, "今日は")
        self.assertEqual(bnsts[1].surface, "天気が")
        self.assertEqual(bnsts[2].surface, "良い。")
