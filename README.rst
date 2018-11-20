====
bseg
====


.. image:: https://img.shields.io/pypi/v/bseg.svg
        :target: https://pypi.python.org/pypi/bseg

.. image:: https://img.shields.io/travis/jun-harashima/bseg.svg
        :target: https://travis-ci.org/jun-harashima/bseg

bseg is a tool for segmenting ipadic-based analysis results into bunsetsu.

Quick Start
===========

To install bseg, run this command in your terminal:

.. code-block:: bash

   $ pip install bseg

Using bseg, you can segment ipadic-based analysis results into bunsetsu as follows:

.. code-block:: python

   import textwrap
   from bseg.bseg import Bseg

   analysis_result = textwrap.dedent("""
   今日	名詞,副詞可能,*,*,*,*,今日,キョウ,キョー
   は	助詞,係助詞,*,*,*,*,は,ハ,ワ
   天気	名詞,一般,*,*,*,*,天気,テンキ,テンキ
   が	助詞,格助詞,一般,*,*,*,が,ガ,ガ
   良い	形容詞,自立,*,*,形容詞・アウオ段,基本形,良い,ヨイ,ヨイ
   。	記号,句点,*,*,*,*,。,。,。
   """)[1:-1]

   bseg = Bseg()
   bnsts = bseg.segment(analysis_result)
   print(bnsts[0].surface)  # => 今日は
